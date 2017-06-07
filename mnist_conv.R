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

conv_mnist <- function(cntk, train_path, test_path, work_path, debug_output=FALSE,
                       epoch_size=60000L, minibatch_size=64L, max_epochs=40L) {
    cntk$logging$set_trace_level(cntk$logging$TraceLevel$Error)
    cntk$device$try_set_default_device(cntk$device$gpu(0L))

    image_height <- 28L
    image_width <- 28L
    num_channels <- 1L
    input_dim <- image_width * image_height * num_channels
    num_output_classes <- 10L

    # Input variables denoting the features and label data
    feature <- cntk$C$input_variable(c(num_channels, image_height, image_width), cntk$np$float32)
    label <- cntk$C$input_variable(num_output_classes, cntk$np$float32)

    # Instantiate the feedforward classification model
    scaled_input <- cntk$ops$constant(0.00390625) %>%
        cntk$ops$element_times(feature)

    cntk$layers$default_options(activation = cntk$ops$relu, pad = FALSE)
    conv1 <- cntk$layers$Convolution2D(c(5,5), 32, pad = TRUE)(scaled_input)
    pool1 <- cntk$layers$MaxPooling(c(3,3), c(2,2))(conv1)
    conv2 <- cntk$layers$Convolution2D(c(3,3), 48)(pool1)
    pool2 <- cntk$layers$MaxPooling(c(3,3), c(2,2))(conv2)
    conv3 <- cntk$layers$Convolution2D(c(3,3), 64)(pool2)
    f4 <- cntk$layers$Dense(96)(conv3)
    drop4 <- cntk$layers$Dropout(0.5)(f4)
    z <- cntk$layers$Dense(num_output_classes, activation=cntk$ops$softmax)(drop4)
    
    ce <- cntk$losses$cross_entropy_with_softmax(z, label)
    pe <- cntk$metrics$classification_error(z, label)

    reader_train <- create_reader(cntk$io,
                                  path = train_path,
                                  is_training = TRUE,
                                  input_dim = input_dim,
                                  label_dim = num_output_classes)

    # Set learning parameters
    lr_per_sample    <- 0.001*10 + 0.0005*10 + 0.0001
    lr_schedule      <- cntk$learners$learning_rate_schedule(lr_per_sample, cntk$learners$UnitType$sample, epoch_size)
    mm_time_constant <- 0*5 + 1024
    mm_schedule      <- cntk$learners$momentum_as_time_constant_schedule(mm_time_constant, epoch_size)

    learner <- cntk$learners$momentum_sgd(z$parameters, lr_schedule, mm_schedule)
    progress_printer <- cntk$logging$ProgressPrinter(tag="Training", num_epochs=max_epochs)
    trainer <- cntk$train$Trainer(z, list(ce, pe), learner, progress_printer)
    
    input_map <- reticulate:::py_dict(
                                 list(feature, label),
                                 list(reader_train$streams$features,
                                      reader_train$streams$labels),
                                  convert = FALSE)

    cntk$logging$log_number_of_parameters(z)


    # Get minibatches of images to train with and perform model training
    for (epoch in 1:max_epochs) {       # loop over epochs
        sample_count <- 0
        while (sample_count < epoch_size) {  # loop over minibatches in the epoch
            data <- reader_train$next_minibatch(as.integer(min(minibatch_size, epoch_size - sample_count)),
                                               input_map=input_map) # fetch minibatch.
            trainer$train_minibatch(reticulate:::py_dict(
                                                     list(feature, label),
                                                     list(data[[1]], 
                                                          data[[2]]),
                                        convert = FALSE)) # update model with it
            sample_count <- sample_count + data[[1]]$num_samples                     # count samples processed so far               
        }
        trainer$summarize_training_progress()
        z$save(file.path(work_path, sprintf("ConvNet_MNIST_%s.dnn", epoch)))
    }
    
    ## minibatch_size <- 64L
    ## num_samples_per_sweep <- 60000L
    ## num_sweeps_to_train_with <- 20L

    ## # Instantiate progress writers.
    ## #training_progress_output_freq = 100
    ## progress_writers <- list(
    ##     cntk$logging$ProgressPrinter(
    ##                                     #freq=training_progress_output_freq,
    ##                      tag="Training",
    ##                      num_epochs=num_sweeps_to_train_with))

    ## if (!is.null(tensorboard_logdir)) {
    ##     progress_writers <- c(progress_writers,
    ##                           cntk$logging$TensorBoardProgressWriter(freq=10L,
    ##                                                                  log_dir=tensorboard_logdir,
    ##                                                                  model=z))
    ## }

    ## lr <- cntk$learners$learning_rate_schedule(1, cntk$learners$UnitType$sample)
    ## trainer <- cntk$train$Trainer(z,
    ##                               list(ce, pe),
    ##                               cntk$learners$adadelta(z$parameters, lr),
    ##                               progress_writers)

    ## cntk$training_session$training_session(
    ##                           trainer = trainer,
    ##                           mb_source = reader_train,
    ##                           mb_size = minibatch_size,
    ##                           model_inputs_to_streams = input_map,
    ##                           max_samples = num_samples_per_sweep * num_sweeps_to_train_with,
    ##                           progress_frequency = num_samples_per_sweep
    ##                       )$train()

    ## reader_test <- create_reader(cntk$io,
    ##                              path = test_path,
    ##                              is_training = FALSE,
    ##                              input_dim = input_dim,
    ##                              label_dim = num_output_classes)

    ## input_map <- reticulate:::py_dict(
    ##                              list(feature, label),
    ##                              list(reader_test$streams$features,
    ##                                   reader_test$streams$labels),
    ##                              convert = FALSE)
    ## # Test data for trained model
    ## test_minibatch_size <- 1024L
    ## num_samples <- 10000L
    ## num_minibatches_to_test <- num_samples / test_minibatch_size
    ## test_result <- 0.0
    ## for (i in 0:as.integer(num_minibatches_to_test)) {        
    ##     mb <- reader_test$next_minibatch(test_minibatch_size)
        
    ##     eval_error <- trainer$test_minibatch(reticulate:::py_dict(
    ##                                                          list(feature, label),
    ##                                                          list(mb[[1]],   
    ##                                                               mb[[2]]),
    ##                                                          convert = FALSE))
    ##     test_result <- test_result + eval_error
    ## }

        ## test_result / num_minibatches_to_test

    NA
}

data_path <- "data"
train_path <- normalizePath(file.path(data_path, "Train-28x28_cntk_text.txt"))
test_path <- normalizePath(file.path(data_path, "Test-28x28_cntk_text.txt"))
work_path <- "work"

error <- import_cntk() %>%
    conv_mnist(train_path = train_path,
               test_path = test_path,
               work_path = work_path)

print(sprintf("Error: %s", error))
