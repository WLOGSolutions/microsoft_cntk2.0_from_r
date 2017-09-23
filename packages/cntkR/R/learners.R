#' @export
learning_rate_schedule <- function(lr_per_sample,
                                   unit_type,
                                   epoch_size = NULL) {
    cntk()$learners$learning_rate_schedule(lr_per_sample,
                                          unit_type,
                                          as.integer(epoch_size))
}

#' @export
learners_unit_type <- list(
    sample = function() { cntk()$learners$UnitType$sample })
    
#' @export
momentum_as_time_constant_schedule <- function(mm_time_constant, epoch_size) {
    cntk()$learners$momentum_as_time_constant_schedule(mm_time_constant,
                                                      as.integer(epoch_size))
}

#' @export
momentum_sgd <- function(parameters,
                         lr_schedule,
                         mm_schedule) {
    cntk()$learners$momentum_sgd(parameters,
                                lr_schedule,
                                mm_schedule)
}

#' @export
adadelta <- function(parameters, rate) {
    cntk()$learners$adadelta(parameters, rate)
}
