#' @export
metric_classification_error <- function(z, label) {
    cntk()$metrics$classification_error(z,
                                       label)
}
