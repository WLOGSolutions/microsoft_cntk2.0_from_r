cntk_env <- new.env()

#' @export
cntk_initialize <- function(python_path) {
    reticulate::use_python(python_path, required = FALSE)

    cntk <- list(
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

    assign("cntk", value = cntk, envir = cntk_env)
    
    invisible(cntk)
}

cntk <- function() {
    get(x = "cntk", envir = cntk_env)
}

#' @export
cntk_io <- function() {
    cntk()$io
}


#' @export
float32 <- function() {
    cntk()$np$float32
}

#' @export
input_variable <- function(shape, type) {
    iv <- cntk()$C$input_variable(shape, type)
    class(iv) <- c("cntk.input_variable", class(iv))
    iv
}

#' @export
scale_input <- function(constant, input) {
    si <- cntk()$ops$constant(constant) %>%
               cntk()$ops$element_times(input)

    class(si) <- c("cntk.scaled_input", class(si))
    return(si)
}

