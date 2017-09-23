#' @export
create_input_map <- function(feature, label, features, labels) {
    reticulate:::py_dict(
                     list(feature, label),
                     list(features,
                          labels),
                     convert = FALSE)
}
