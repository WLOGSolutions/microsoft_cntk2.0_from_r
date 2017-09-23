#' @export
loss_cross_entropy_with_softmax <- function(z, label) {
    cntk()$losses$cross_entropy_with_softmax(z,
                                            label)
}
