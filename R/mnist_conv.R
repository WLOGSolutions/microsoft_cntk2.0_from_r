# Detect proper script_path (you cannot use args yet as they are build with tools in set_env.r)
script_path <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- dirname(sub("--file=", "", args[grep("--file=", args)]))
  if (!length(script_path)) { return(".") }
  return(normalizePath(script_path))
})()

# Setting .libPaths() to point to libs folder
source(file.path(script_path, "set_env.r"), chdir = T)

config <- load_config()
args <- args_parser()

data_path <- normalizePath(file.path(script_path, "..", "data"))
work_path <- normalizePath(file.path(script_path, "..", "work"))

device <- args$get("device", required = FALSE, default = "gpu")

library(cntkR)

cntk_initialize(python_path = config$python)

create_reader <- function(path, is_training, input_dim, label_dim) {
  max_sweeps <- if (is_training) 10000000L else 1L

  cntk_io()$StreamDefs(features  = cntk_io()$StreamDef(field="features",
                                                       shape=input_dim,
                                                       is_sparse=FALSE),
                       labels    = cntk_io()$StreamDef(field="labels",
                                                       shape=label_dim,
                                                       is_sparse=FALSE)) %>%
    cntk_io()$CTFDeserializer(path, .) %>%
    cntk_io()$MinibatchSource(randomize = is_training,
                              max_sweeps = max_sweeps)
}

conv_mnist <- function(train_path, test_path, work_path, debug_output=FALSE,
                       epoch_size=60000L, minibatch_size=64L, max_epochs=40L) {
  logging_set_trace_level("Error")
  try_set_default_device(switch(device,
                                cpu = cpu(),
                                gpu = gpu(0)))

  image_height <- 28L
  image_width <- 28L
  num_channels <- 1L
  input_dim <- image_width * image_height * num_channels
  num_output_classes <- 10L

  # Input variables denoting the features and label data
  feature <- input_variable(c(num_channels, image_height, image_width), float32())
  label <- input_variable(num_output_classes, float32())

  # Instantiate the feedforward classification model
  scaled_input <- scale_input(0.00390625, feature)

  layers_default_options(activation = activation_relu(), pad = FALSE)
  z <- scaled_input %>%
    layer_convolution2D(c(5,5), 32, pad = TRUE) %>%
    layer_max_pooling(c(3,3), c(2,2)) %>%
    layer_convolution2D(c(3,3), 48) %>%
    layer_max_pooling(c(3,3), c(2,2)) %>%
    layer_convolution2D(c(3,3), 64) %>%
    layer_dense(96) %>%
    layer_dropout(0.5) %>%
    layer_dense(num_output_classes, activation = activation_softmax())

  ce <- loss_cross_entropy_with_softmax(z, label)
  pe <- metric_classification_error(z, label)

  reader_train <- create_reader(path = train_path,
                                is_training = TRUE,
                                input_dim = input_dim,
                                label_dim = num_output_classes)

  # Set learning parameters
  lr_per_sample    <- 0.001*10 + 0.0005*10 + 0.0001
  lr_schedule      <- learning_rate_schedule(lr_per_sample, learners_unit_type$sample(), epoch_size)
  mm_time_constant <- 0*5 + 1024
  mm_schedule      <- momentum_as_time_constant_schedule(mm_time_constant, epoch_size)

  learner <- momentum_sgd(z$parameters, lr_schedule, mm_schedule)

  progress_printer <- progress_printer(log_to_file = file.path(script_path, "../logs/train.log"),
                                       tag="Training",
                                       num_epochs=max_epochs)
  trainer <- create_trainer(model = z,
                            loss = ce,
                            metric = pe,
                            learner,
                            progress_printer)

  input_map <- create_input_map(feature = feature, label = label,
                                features = reader_train$streams$features,
                                labels = reader_train$streams$labels)

  log_number_of_parameters(z)
  # Get minibatches of images to train with and perform model training
  for (epoch in 1:max_epochs) {       # loop over epochs
    sample_count <- 0
    while (sample_count < epoch_size) {  # loop over minibatches in the epoch
      data <- reader_train$next_minibatch(as.integer(min(minibatch_size, epoch_size - sample_count)),
                                          input_map=input_map) # fetch minibatch.
      trainer$train_minibatch(create_input_map(feature = feature,
                                               label = label,
                                               features = data[[1]],
                                               labels = data[[2]])) # update model with it
      sample_count <- sample_count + data[[1]]$num_samples                     # count samples processed so far
    }
    trainer$summarize_training_progress()
    z$save(file.path(work_path, sprintf("ConvNet_MNIST_%s.dnn", epoch)))
    loginfo("Epoch = %s done", epoch)
  }

  # Load test data
  reader_test <- create_reader(path = test_path,
                               is_training = FALSE,
                               input_dim = input_dim,
                               label_dim = num_output_classes)

  input_map <- create_input_map(feature,
                                label,
                                features = reader_test$streams$features,
                                labels = reader_test$streams$labels)

  ## # Test data for trained model
  epoch_size <- 10000L
  minibatch_size <- 1024

  # process minibatches and evaluate the model
  metric_numer <- 0
  metric_denom <- 0
  sample_count <- 0
  minibatch_index <- 0

  test_result <- 0.0

  while (sample_count < epoch_size) {
    current_minibatch <- as.integer(min(minibatch_size, epoch_size - sample_count))

    # Fetch next test min batch.
    data <- reader_test$next_minibatch(current_minibatch,
                                       input_map=input_map)

    # minibatch data to be trained with
    metric_numer <- metric_numer + trainer$test_minibatch(create_input_map(feature,
                                                                           label,
                                                                           features = data[[1]],
                                                                           labels = data[[2]])) * current_minibatch
    metric_denom <- metric_denom + current_minibatch

    # Keep track of the number of samples processed so far.
    sample_count <- sample_count + data[[1]]$num_samples
    minibatch_index <- minibatch_index + 1
  }

  loginfo("Final Results: Minibatch[1-%s]: errs = %0.2f%% * %s",
          minibatch_index + 1,
          (metric_numer * 100.0) / metric_denom,
          metric_denom)


  metric_numer / metric_denom
}


train_path <- normalizePath(file.path(data_path, "Train-28x28_cntk_text.txt"))
test_path <- normalizePath(file.path(data_path, "Test-28x28_cntk_text.txt"))

error <- conv_mnist(train_path = train_path,
                    test_path = test_path,
                    work_path = work_path)

loginfo("Error: %s", error)
