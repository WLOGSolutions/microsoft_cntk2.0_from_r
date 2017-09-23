
#' @export
layer_convolution2D <- function(input_layer, ...) {
    cntk()$layers$Convolution2D(...)(input_layer)
}

#' @export
layer_max_pooling <- function(input_layer, ...) {
    cntk()$layers$MaxPooling(...)(input_layer)
}

#' @export
layer_dense <- function(input_layer = NULL, ...) {
    cntk()$layers$Dense(...)(input_layer)
}


#' @export
layer_dropout <- function(input_layer, ...) {
    cntk()$layers$Dropout(...)(input_layer)
}

#' @export
activation_softmax <- function() {
    cntk()$ops$softmax
}

#' @export
activation_relu <- function() {
  cntk()$ops$relu  
}


#' @export
layers_default_options <- function(activation, pad) {
    cntk()$layers$default_options(activation = activation,
                                 pad = FALSE)
}

#' @export
layer_sequential <-  function(layers, input) {
    l <- cntk()$layers$Sequential(layers)(input)
    class(l) <- c(class(l), "cntk.layers.Sequential")
    return(l)
}

#' @export
layers_seq <- function(N, layer) {
    purrr::rerun(N,
                 layer)
}
