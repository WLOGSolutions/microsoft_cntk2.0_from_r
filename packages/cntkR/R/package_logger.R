pkg_env <- new.env()
for (lev in c("info", "warn", "debug", "error")) {
  logger_lev <- logging::getLogger("cntk")
  logging::setLevel(level = "FINEST", logger_lev)
  assign(lev, value = logger_lev, envir = pkg_env)
}
pkg_logger <- function(x) {
  get(x = x, envir = pkg_env)
}

pkg_loginfo <- function(msg, ...) { logging::loginfo(msg,  logger = pkg_logger("info"), ...)}
pkg_logdebug <- function(msg, ...) { logging::logdebug(msg,  logger = pkg_logger("debug"), ...)}
pkg_logerror <- function(msg, ...) { logging::logerror(msg,  logger = pkg_logger("error"), ...)}
pkg_logwarn <- function(msg, ...) { logging::logwarn(msg,  logger = pkg_logger("warn"), ...)}
