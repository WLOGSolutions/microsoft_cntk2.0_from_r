#' @export
cpu <- function() {
    cntk()$device$cpu()
}

#' @export
gpu <- function(id) {
    cntk()$device$gpu(as.integer(id))
}

#' @export
try_set_default_device <- function(dev = cpu()) {
    cntk()$device$try_set_default_device(dev)
}
