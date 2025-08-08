#' @title Adaptive Bayesian estimation for \pkg{yuima} objects
#'
#' @importFrom methods new setClass setGeneric setMethod
#' 
#' @description
#' Generates Stan code on-the-fly and performs Bayesian inference for
#' (possibly multivariate) stochastic differential equations stored in a
#' \pkg{yuima} object.
#' Each model parameter is declared individually in Stan, so posterior draws
#' keep their symbolic names.
#'
#' @details
#' The function parses the drift and diffusion specified in \code{object@model}
#' and builds a Gaussian Euler-type one–step transition.
#' It auto-generates a Stan program with one \emph{unconstrained} (by default)
#' parameter per model symbol, optionally applying Stan bounds when requested
#' via \code{bounds}. Any parameters not listed in \code{bounds} remain
#' unconstrained.
#'
#' @section Parameter bounds:
#' Supply \code{bounds} as a named list such as
#' \code{list(sigma = c(0, Inf), rho = c(-1, 1))}. Each entry is interpreted as
#' \eqn{(\mathrm{lower}, \mathrm{upper})}. Use \code{Inf}, \code{-Inf}, or
#' \code{NULL} for “no bound” on that side. Bounds are translated into Stan
#' constraints, e.g. \code{real<lower=0>} or \code{real<lower=0, upper=1>}.
#'
#' @section Priors:
#' You may pass \code{priors} as a named list mapping parameter names to a Stan
#' distribution string (without the trailing semicolon), e.g.
#' \code{list(sigma = "student_t(3, 0, 0.5)", rho = "uniform(-1, 1)")}.
#' Any parameter not listed in \code{priors} receives \code{default_prior}.
#' Set \code{default_prior = NULL} to omit priors entirely (i.e., use only the
#' likelihood with Stan bounds as constraints).
#'
#' @name bayes
#' @aliases bayes,yuima-method
#' @docType methods
#'
#' @param object A \code{yuima} object containing model + data.
#' @param chains Number of MCMC chains (forwarded to Stan). Default \code{4}.
#' @param iter   Number of iterations \emph{per chain}. Default \code{2000}.
#' @param bounds Named list like \code{list(sigma = c(0, Inf), rho = c(-1, 1))}
#'               giving \eqn{(\mathrm{lower},\,\mathrm{upper})} for any subset
#'               of parameters. Use \code{Inf}, \code{-Inf}, or \code{NULL}
#'               for “no bound”.
#' @param priors Named list mapping parameter names to a Stan distribution
#'               string (no semicolon). Use \code{FALSE} or \code{NULL} for a
#'               specific parameter to suppress its prior even if
#'               \code{default_prior} is set.
#' @param default_prior Single distribution string applied to all parameters not
#'               listed in \code{priors}. Default \code{"normal(0, 5)"}.
#'               Set \code{NULL} to apply no default prior.
#' @param ...    Further arguments passed to \code{\link[rstan]{sampling}}.
#'
#' @return
#' An S4 object of class \code{"bayes.yuima"} with slots:
#' \itemize{
#'   \item \code{fit}: the fitted Stan object returned by \code{rstan::sampling()}.
#'   \item \code{call}: the original function call.
#'   \item \code{yuima}: the input \code{yuima} object.
#' }
#'
#' @seealso
#' \pkg{yuima};
#' \code{\link[yuima:setModel]{setModel}},
#' \code{\link[yuima:setYuima]{setYuima}},
#' \code{\link[yuima:simulate]{simulate}};
#' \code{\link[zoo:index]{index}};
#' \code{\link[rstan]{sampling}}, \code{\link[rstan]{stan_model}}
#'
#' @examples
#' \donttest{
#' if (requireNamespace("yuima", quietly = TRUE) &&
#'     requireNamespace("rstan",  quietly = TRUE)) {
#'   library(yuima)
#'
#'   ## 1D OU model
#'   mod <- setModel(
#'     drift          = "theta * (mu - x)",
#'     diffusion      = "sigma",
#'     time.variable  = "t",
#'     state.variable = "x",
#'     solve.variable = "x"
#'   )
#'
#'   set.seed(1)
#'   samp <- setSampling(Initial = 0, n = 200, delta = 0.01)
#'   yui  <- setYuima(model = mod, sampling = samp)
#'   sim  <- simulate(yui, xinit = 0,
#'                    true.parameter = list(theta = 1, mu = 0, sigma = 0.3))
#'
#'   # Fast, stable example: weakly-informative priors + adapt_delta
#'   fit <- bayes(
#'     sim,
#'     bounds        = list(sigma = c(0, Inf)),
#'     priors        = list(sigma = "student_t(3, 0, 0.5)"),
#'     default_prior = "student_t(3, 0, 2.5)",
#'     chains        = 2,
#'     iter          = 800,
#'     control       = list(adapt_delta = 0.95)
#'   )
#'
#'   print(fit@fit)
#' }
#' }
#'
#' \dontrun{
#' # Custom priors (heavier; not run on CRAN and not by --run-donttest):
#' fit2 <- bayes(
#'   sim,
#'   bounds        = list(sigma = c(0, Inf)),
#'   priors        = list(
#'     sigma = "student_t(3, 0, 0.5)",
#'     theta = "normal(0, 3)",  # slightly tighter than default
#'     mu    = "normal(0, 3)"
#'   ),
#'   default_prior = "student_t(3, 0, 2.5)",
#'   chains        = 2,
#'   iter          = 1000,
#'   control       = list(adapt_delta = 0.98)
#' )
#'
#' # Likelihood-dominant (no priors): may need higher adapt_delta and more iters
#' fit3 <- bayes(sim, default_prior = NULL,
#'               chains = 2, iter = 1200,
#'               control = list(adapt_delta = 0.99))
#' }
#'
#' @import            yuima
#' @importClassesFrom yuima  yuima
#' @importFrom        zoo     index
#' @importFrom        rstan   sampling stan_model
#' @export
NULL


setGeneric("bayes", function(object, ...) standardGeneric("bayes"))

setClass("bayes.yuima",
         slots = c(fit   = "ANY",
                   call  = "language",
                   yuima = "yuima"))

#' @rdname bayes
#' @export
setMethod(
  "bayes", "yuima",
  function(object,
           chains         = 4,
           iter           = 2000,
           bounds         = list(),
           priors         = list(),
           default_prior  = "normal(0, 5)",
           ...) {

    if (!requireNamespace("rstan", quietly = TRUE))
      stop("Package 'rstan' is required but not installed.", call. = FALSE)

    ## 1 ─ data ----------------------------------------------------------------
    Y  <- zoo::coredata(object@data@original.data)
    ts <- zoo::index(object@data@original.data)
    if (anyNA(Y) || anyNA(ts))
      stop("Data or index contains NA.", call. = FALSE)

    N  <- nrow(Y) - 1L
    D  <- ncol(Y)
    dt <- as.numeric(diff(ts))

    if (length(dt) != N)
      stop("Length of dt must equal N.", call. = FALSE)

    Y_prev <- Y[-nrow(Y), , drop = FALSE]
    Y_next <- Y[-1,       , drop = FALSE]

    ## 2 ─ helper: expression → string ----------------------------------------
    extract_txt <- function(slot) {
      if (is.list(slot))
        return(unlist(lapply(slot, extract_txt), use.names = FALSE))
      if (inherits(slot, "expression"))
        return(vapply(slot, \(e) paste(deparse(e), collapse = ""), character(1)))
      if (is.call(slot) || is.name(slot))
        return(paste(deparse(slot), collapse = ""))
      stop("Cannot parse slot class ", class(slot)[1])
    }

    ## 3 ─ drift / diffusion raw strings --------------------------------------
    drift_raw <- extract_txt(object@model@drift)
    diff_raw  <- extract_txt(object@model@diffusion)
    if (length(diff_raw) == 1L && D > 1L) {
      inner <- sub("^c\\((.*)\\)$", "\\1", diff_raw)
      diffusion_exprs <- trimws(strsplit(inner, ",")[[1]])
    } else {
      diffusion_exprs <- diff_raw
    }
    if (length(drift_raw) != D)
      stop("Drift length must equal state dimension D.")
    if (length(diffusion_exprs) != D * D)
      stop("Diffusion entry count must equal D^2.")

    ## 4 ─ parameter names -----------------------------------------------------
    all_txt     <- c(drift_raw, diffusion_exprs)
    syms        <- unique(unlist(lapply(all_txt, \(t) all.vars(parse(text = t)))))
    state_vars  <- object@model@state.variable
    time_var    <- object@model@time.variable
    param_names <- setdiff(syms, c(state_vars, time_var))

    ## 4a ─ bounds; NA means “no bound” ---------------------------------------
    lower_vec <- rep(NA_character_, length(param_names))
    upper_vec <- rep(NA_character_, length(param_names))

    if (length(bounds)) {
      bad <- setdiff(names(bounds), param_names)
      if (length(bad))
        stop("bounds names not parameters: ", paste(bad, collapse = ", "))
      for (nm in names(bounds)) {
        idx <- match(nm, param_names)
        lo  <- bounds[[nm]][1]; up <- bounds[[nm]][2]
        if (!is.null(lo) && is.finite(lo)) lower_vec[idx] <- as.character(lo)
        if (!is.null(up) && is.finite(up)) upper_vec[idx] <- as.character(up)
      }
    }

    ## 4b ─ collision-proof work names ----------------------------------------
    reserved <- c(param_names, state_vars, time_var, "pi", "e", "N", "D")
    make_safe <- function(base) {
      nm <- base
      while (nm %in% reserved) nm <- paste0(nm, "_tmp")
      reserved <<- c(reserved, nm); nm
    }
    x_work     <- make_safe("x")
    mu_work    <- make_safe("mu_pred")
    G_work     <- make_safe("G")
    Sigma_work <- make_safe("Sigma")

    ## 4c ─ substitute state vars only ----------------------------------------
    subst_state <- function(txt) {
      s <- txt
      for (j in seq_along(state_vars))
        s <- gsub(paste0("\\b", state_vars[j], "\\b"),
                  paste0(x_work, "[", j, "]"), s, perl = TRUE)
      s
    }

    ## 5 ─ parameters block ----------------------------------------------------
    param_lines <- vapply(seq_along(param_names), function(i) {
      lo <- lower_vec[i]; up <- upper_vec[i]; nm <- param_names[i]
      if (is.na(lo) && is.na(up)) {
        sprintf("  real %s;", nm)
      } else if (is.na(lo)) {
        sprintf("  real<upper=%s> %s;", up, nm)
      } else if (is.na(up)) {
        sprintf("  real<lower=%s> %s;", lo, nm)
      } else {
        sprintf("  real<lower=%s, upper=%s> %s;", lo, up, nm)
      }
    }, character(1))

    ## 5a ─ priors (optional) --------------------------------------------------
    if (length(priors)) {
      bad <- setdiff(names(priors), param_names)
      if (length(bad))
        stop("priors names not parameters: ", paste(bad, collapse = ", "))
    }
    build_prior_lines <- function(param_names, priors, default_prior) {
      out <- character(0)
      for (nm in param_names) {
        pr <- if (length(priors) && !is.null(priors[[nm]])) priors[[nm]] else default_prior
        if (is.null(pr) || identical(pr, FALSE)) next
        if (!is.character(pr) || length(pr) != 1L)
          stop("Each prior must be a length-1 character like \"normal(0, 5)\" or FALSE.")
        out <- c(out, sprintf("  %s ~ %s;", nm, pr))
      }
      out
    }
    prior_lines <- build_prior_lines(param_names, priors, default_prior)

    ## 6 ─ Stan program --------------------------------------------------------
    lines <- c(
      "// auto-generated by bayes.yuima",
      "data {",
      "  int<lower=1> N;",
      "  int<lower=1> D;",
      "  matrix[N, D] y_prev;",
      "  matrix[N, D] y_next;",
      "  vector[N] dt;",
      "}",
      "parameters {",
      param_lines,
      "}",
      "model {",
      prior_lines,      # zero or more lines like: param ~ normal(0, 5);
      "",
      "  for (i in 1:N) {",
      sprintf("    vector[D] %s = y_prev[i]';", x_work),
      sprintf("    vector[D] %s;", mu_work),
      sprintf("    matrix[D, D] %s;", G_work),
      sprintf("    matrix[D, D] %s;", Sigma_work)
    )

    for (j in seq_len(D)) {
      code <- subst_state(drift_raw[j])
      lines <- c(lines,
        sprintf("    %s[%d] = %s[%d] + (%s) * dt[i];",
                mu_work, j, x_work, j, code))
    }
    for (j in seq_len(D)) for (k in seq_len(D)) {
      idx  <- (j - 1) * D + k
      code <- subst_state(diffusion_exprs[idx])
      lines <- c(lines,
        sprintf("    %s[%d, %d] = %s;", G_work, j, k, code))
    }
    lines <- c(lines,
      sprintf("    %s = %s * %s' * dt[i];", Sigma_work, G_work, G_work),
      sprintf("    y_next[i] ~ multi_normal(%s, %s);", mu_work, Sigma_work),
      "  }",
      "}"
    )
    stan_code <- paste(lines, collapse = "\n")

    ## 7 ─ compile & sample ----------------------------------------------------
    sm  <- rstan::stan_model(model_code = stan_code, model_name = "yuima_bayes")
    fit <- rstan::sampling(
      sm,
      data    = list(N = N, D = D,
                     y_prev = Y_prev,
                     y_next = Y_next,
                     dt     = as.vector(dt)),
      chains  = chains,
      iter    = iter,
      refresh = 0,
      ...
    )

    ## 8 ─ return --------------------------------------------------------------
    new("bayes.yuima",
        fit   = fit,
        call  = match.call(),
        yuima = object)
  }
)
