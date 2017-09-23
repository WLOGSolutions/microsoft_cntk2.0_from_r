#' @export
logging_set_trace_level <- function(level = "Error") {
    switch(level,
           Error = cntk()$logging$TraceLevel$Error,
           Info = cntk()$logging$TraceLevel$Info,
           Warning = cntk()$logging$TraceLevel$Warning,
           cntk()$logging$TraceLevel$Info)
}

#' @export
progress_printer <- function(freq = NULL,
                             first = 0L,
                             tag = '',
                             log_to_file = NULL,
                             rank = NULL,
                             gen_heartbeat = FALSE,
                             num_epochs = NULL,
                             test_freq = NULL,
                             test_first = 0L,
                             metric_is_pct = TRUE,
                             distributed_freq = NULL,
                             distributed_first = 0L) {
    
    cntk()$logging$ProgressPrinter(freq = freq,
                                   first = first,
                                   tag = tag,
                                   log_to_file = log_to_file,
                                   rank = rank,
                                   gen_heartbeat = gen_heartbeat,
                                   num_epochs = as.integer(num_epochs),
                                   test_freq = test_freq,
                                   test_first = test_first,
                                   metric_is_pct = metric_is_pct,
                                   distributed_freq = distributed_freq,
                                   distributed_first = distributed_first)
}

#' @export
log_number_of_parameters <- function(model) {
    cntk()$logging$log_number_of_parameters(model)
}
