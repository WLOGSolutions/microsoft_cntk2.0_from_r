.libPaths("lib")

library(magrittr)
library(purrr)
library(reticulate)

#Change to your Python installation path
my_python <- "~/.conda/envs/cntk2.0/bin"

reticulate::use_python(my_python, required = FALSE)

import_cntk <- function() {
    list(
        C = import("cntk"),
        io = import("cntk.io"),
        ops = import("cntk.ops"),
        layers = import("cntk.layers"),
        train = import("cntk.train"),
        training_session = import("cntk.train.training_session"),
        learners = import("cntk.learners"),
        losses = import("cntk.losses"),
        metrics = import("cntk.metrics"),
        logging = import("cntk.logging"),
        device = import("cntk.device"),
        np = import("numpy", as="np"))
}


create_reader <- function(cntk_io, path, is_training, input_dim, label_dim) {
    max_sweeps <- if (is_training) 10000000L else 1L

    cntk_io$StreamDefs(features  = cntk_io$StreamDef(field="features",
                                                     shape=input_dim,
                                                     is_sparse=FALSE),
                       labels    = cntk_io$StreamDef(field="labels",
                                                     shape=label_dim,
                                                     is_sparse=FALSE)) %>%
        cntk_io$CTFDeserializer(path, .) %>%
        cntk_io$MinibatchSource(randomize = is_training,
                                max_sweeps = max_sweeps)
}

simple_mnist <- function(cntk, train_path, test_path, tensorboard_logdir = NULL) {
    cntk$logging$set_trace_level(cntk$logging$TraceLevel$Error)
    cntk$device$try_set_default_device(cntk$device$gpu(0L))

    input_dim <- 784L
    num_output_classes <- 10L
    num_hidden_layers <- 2L
    hidden_layers_dim <- 200L

    # Input variables denoting the features and label data
    feature <- cntk$C$input_variable(input_dim, cntk$np$float32)
    label <- cntk$C$input_variable(num_output_classes, cntk$np$float32)

    # Instantiate the feedforward classification model
    scaled_input <- cntk$ops$constant(0.00390625) %>%
        cntk$ops$element_times(feature)

    Dense <- cntk$layers$Dense
    Sequential <- cntk$layers$Sequential
    relu <- cntk$ops$relu
    softmax <- cntk$ops$softmax

    z <- (c(rerun(num_hidden_layers,
                 Dense(hidden_layers_dim,
                       activation = relu)),
           Dense(num_output_classes,
                 activation = softmax)) %>%
          Sequential())(scaled_input)
    
    ce <- cntk$losses$cross_entropy_with_softmax(z, label)
    pe <- cntk$metrics$classification_error(z, label)

    reader_train <- create_reader(cntk$io,
                                  path = train_path,
                                  is_training = TRUE,
                                  input_dim = input_dim,
                                  label_dim = num_output_classes)

    input_map <- reticulate:::py_dict(
                                 list(feature, label),
                                 list(reader_train$streams$features,
                                      reader_train$streams$labels),
                                  convert = FALSE)

    minibatch_size <- 64L
    num_samples_per_sweep <- 60000L
    num_sweeps_to_train_with <- 20L

    # Instantiate progress writers.
    #training_progress_output_freq = 100
    progress_writers <- list(
        cntk$logging$ProgressPrinter(
                                        #freq=training_progress_output_freq,
                         tag="Training",
                         num_epochs=num_sweeps_to_train_with))

    if (!is.null(tensorboard_logdir)) {
        progress_writers <- c(progress_writers,
                              cntk$logging$TensorBoardProgressWriter(freq=10L,
                                                                     log_dir=tensorboard_logdir,
                                                                     model=z))
    }

    lr <- cntk$learners$learning_rate_schedule(1, cntk$learners$UnitType$sample)
    trainer <- cntk$train$Trainer(z,
                                  list(ce, pe),
                                  cntk$learners$adadelta(z$parameters, lr),
                                  progress_writers)

    cntk$training_session$training_session(
                              trainer = trainer,
                              mb_source = reader_train,
                              mb_size = minibatch_size,
                              model_inputs_to_streams = input_map,
                              max_samples = num_samples_per_sweep * num_sweeps_to_train_with,
                              progress_frequency = num_samples_per_sweep
                          )$train()

    reader_test <- create_reader(cntk$io,
                                 path = test_path,
                                 is_training = FALSE,
                                 input_dim = input_dim,
                                 label_dim = num_output_classes)

    input_map <- reticulate:::py_dict(
                                 list(feature, label),
                                 list(reader_test$streams$features,
                                      reader_test$streams$labels),
                                 convert = FALSE)
    # Test data for trained model
    test_minibatch_size <- 1024L
    num_samples <- 10000L
    num_minibatches_to_test <- num_samples / test_minibatch_size
    test_result <- 0.0
    for (i in 0:as.integer(num_minibatches_to_test)) {        
        mb <- reader_test$next_minibatch(test_minibatch_size)
        
        eval_error <- trainer$test_minibatch(reticulate:::py_dict(
                                                             list(feature, label),
                                                             list(mb[[1]],   
                                                                  mb[[2]]),
                                                             convert = FALSE))
        test_result <- test_result + eval_error
    }

    test_result / num_minibatches_to_test
}

data_path <- "data"
train_path <- normalizePath(file.path(data_path, "Train-28x28_cntk_text.txt"))
test_path <- normalizePath(file.path(data_path, "Test-28x28_cntk_text.txt"))

error <- import_cntk() %>%
    simple_mnist(train_path = train_path,
                 test_path = test_path)

print(sprintf("Error: %s", error))
