#' @export
create_trainer <- function(model,
                    loss,
                    metric,
                    learner,
                    progress_printer) {
    cntk()$train$Trainer(model,
                         list(loss,
                              metric),
                         learner,
                         progress_printer)
}
