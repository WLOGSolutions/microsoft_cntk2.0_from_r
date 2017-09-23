#' @export
create_training_session <- function(trainer,
                                    mb_source,
                                    mb_size,
                                    model_inputs_to_streams,
                                    max_samples,
                                    progress_frequency) {
    cntk()$training_session$training_session(
                              trainer = trainer,
                              mb_source = mb_source,
                              mb_size = mb_size,
                              model_inputs_to_streams = model_inputs_to_streams,
                              max_samples = max_samples,
                              progress_frequency = progress_frequency)
}
