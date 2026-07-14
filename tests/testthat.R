# Entry point for testthat. Run via:
#   Rscript tests/testthat.R
# or for selective filtering:
#   Rscript -e "testthat::test_dir('tests/testthat', filter='spearman')"

library(testthat)

# Resolve project root via several fallbacks so the script works whether run
# from the project root, the tests/ dir, or anywhere via absolute path.
resolve_project_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[1])
  if (!is.na(file_arg) && nzchar(file_arg)) {
    script <- normalizePath(file_arg, mustWork = FALSE)
    return(normalizePath(file.path(dirname(script), ".."), mustWork = FALSE))
  }
  cwd <- normalizePath(getwd(), mustWork = FALSE)
  if (basename(cwd) == "tests") return(normalizePath(file.path(cwd, ".."), mustWork = FALSE))
  if (file.exists(file.path(cwd, "tests", "testthat"))) return(cwd)
  cwd
}
project_root <- resolve_project_root()
Sys.setenv(HOBOTNICA_PROJECT_ROOT = project_root)

test_dir(file.path(project_root, "tests", "testthat"), reporter = "summary")
