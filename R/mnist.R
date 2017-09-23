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

device <- args$get("device", required = FALSE, default = "cpu")

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

simple_mnist <- function(train_path, test_path, tensorboard_logdir = NULL) {
    logging_set_trace_level("Error")
    try_set_default_device(switch(device,
                                  cpu = cpu(),
                                  gpu = gpu(0)))

    input_dim <- 784L
    num_output_classes <- 10L
    num_hidden_layers <- 2L
    hidden_layers_dim <- 200L

    # Input variables denoting the features and label data
    feature <- input_variable(input_dim, float32())
    label <- input_variable(num_output_classes, float32())

    # Instantiate the feedforward classification model
    scaled_input <- scale_input(0.00390625, feature)

    z <- layers_seq(num_hidden_layers,
                    layer = partial(layer_dense,
                                    shape = hidden_layers_dim,
                                    activation = activation_relu())) %>%
        append(partial(layer_dense,
                   shape = num_output_classes,
                   activation = activation_softmax())) %>%
        layer_sequential(scaled_input)

    ce <- loss_cross_entropy_with_softmax(z, label)

    pe <- metric_classification_error(z, label)

    reader_train <- create_reader(path = train_path,
                                  is_training = TRUE,
                                  input_dim = input_dim,
                                  label_dim = num_output_classes)

    input_map <- create_input_map(
        feature,
        label,
        features = reader_train$streams$features,
        labels = reader_train$streams$labels)

    minibatch_size <- 64L
    num_samples_per_sweep <- 60000L
    num_sweeps_to_train_with <- 20L

    # Instantiate progress writers.
    #training_progress_output_freq = 100
    progress_writers <- list(
        progress_printer(
            log_to_file = file.path(script_path, "../logs/train.log"),
            tag="Training",
            num_epochs = num_sweeps_to_train_with))

    lr <- learning_rate_schedule(1,
                                 learners_unit_type$sample())
    trainer <- create_trainer(model = z,
                              loss = ce,
                              metric = pe,
                              learner = adadelta(z$parameters, lr),
                              progress_printer = progress_writers)

    create_training_session(
        trainer = trainer,
        mb_source = reader_train,
        mb_size = minibatch_size,
        model_inputs_to_streams = input_map,
        max_samples = num_samples_per_sweep * num_sweeps_to_train_with,
        progress_frequency = num_samples_per_sweep
    )$train()

    reader_test <- create_reader(path = test_path,
                                 is_training = FALSE,
                                 input_dim = input_dim,
                                 label_dim = num_output_classes)

    input_map <- create_input_map(
        feature,
        label,
        features = reader_test$streams$features,
        labels = reader_test$streams$labels)

    # Test data for trained model
    test_minibatch_size <- 1024L
    num_samples <- 10000L
    num_minibatches_to_test <- num_samples / test_minibatch_size
    test_result <- 0.0
    for (i in 0:as.integer(num_minibatches_to_test)) {
        mb <- reader_test$next_minibatch(test_minibatch_size)
        eval_error <- trainer$test_minibatch(create_input_map(feature, label,
                                                             features = mb[[1]],
                                                             labels = mb[[2]]))
        test_result <- test_result + eval_error
    }

    test_result / num_minibatches_to_test
}

train_path <- normalizePath(file.path(data_path, "Train-28x28_cntk_text.txt"))
test_path <- normalizePath(file.path(data_path, "Test-28x28_cntk_text.txt"))

error <- simple_mnist(train_path = train_path,
                 test_path = test_path)

loginfo("Error: %s", error)
