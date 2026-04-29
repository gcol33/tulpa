#' Formula parsing for tulpa models
#'
#' @description
#' Parses mixed-model formulas by walking the formula's abstract syntax tree.
#' R formulas are already parse trees — we do structural recursion to find
#' random effect terms (`|` nodes) and separate them from fixed effects.
#'
#' @name tulpa_formula
NULL

# ============================================================================
# Core AST operations: findbars / nobars
#
# A formula like y ~ x + (1 | g) + (x || g2) is a nested call object:
#   ~(y, +(+(x, (|(1, g))), (||(x, g2))))
#
# findbars: collect all | and || nodes (random effects)
# nobars:  rewrite the tree with bar terms removed (fixed effects)
# ============================================================================

#' Find all bar terms in a formula's parse tree
#'
#' Recursively walks the formula AST and collects all `|` and `||` nodes
#' found inside parentheses. These are the random effect specifications.
#'
#' @param term A language object (formula term)
#' @return A list of language objects, each a `|` or `||` call
#' @keywords internal
#' @export
findbars <- function(term) {
  if (is.name(term) || !is.language(term)) return(NULL)

  if (term[[1]] == as.name("(")) {
    inner <- term[[2]]
    if (is.call(inner) && (inner[[1]] == as.name("|") || inner[[1]] == as.name("||"))) {
      return(list(inner))
    }
    return(findbars(inner))
  }

  if (term[[1]] == as.name("|") || term[[1]] == as.name("||")) {
    return(list(term))
  }

  if (length(term) == 2) {
    return(findbars(term[[2]]))
  }

  c(findbars(term[[2]]), findbars(term[[3]]))
}

#' Remove all bar terms from a formula's parse tree
#'
#' Recursively rewrites the formula AST, removing any `|` or `||` nodes
#' found inside parentheses. Returns the fixed-effects-only formula.
#'
#' @param term A language object (formula term)
#' @return A language object with all bar terms removed, or NULL if nothing remains
#' @keywords internal
#' @export
nobars <- function(term) {
  if (is.name(term) || !is.language(term)) return(term)

  if (term[[1]] == as.name("(")) {
    inner <- term[[2]]
    if (is.call(inner) && (inner[[1]] == as.name("|") || inner[[1]] == as.name("||"))) {
      return(NULL)
    }
    nb <- nobars(inner)
    if (is.null(nb)) return(NULL)
    return(call("(", nb))
  }

  if (length(term) == 2) {
    nb <- nobars(term[[2]])
    if (is.null(nb)) return(NULL)
    return(call(deparse(term[[1]]), nb))
  }

  nb_left  <- nobars(term[[2]])
  nb_right <- nobars(term[[3]])

  if (is.null(nb_left) && is.null(nb_right)) return(NULL)
  if (is.null(nb_left))  return(nb_right)
  if (is.null(nb_right)) return(nb_left)

  call(deparse(term[[1]]), nb_left, nb_right)
}

# ============================================================================
# Bar term parsing: structured RE specification with || expansion
# ============================================================================

#' Parse a single random effect bar term
#'
#' Takes a `|` or `||` language object and extracts the grouping variable(s),
#' effect terms (intercept, slopes), and correlation structure. The bar
#' operator drives the `correlated` flag: `|` → `TRUE` (LKJ-Cholesky on the
#' joint slope vector), `||` → `FALSE` (diagonal covariance, one σ per
#' coefficient). This matches lme4 / glmmTMB and lets downstream packages
#' branch on `correlated` to choose between independent-σ and Cholesky
#' parameterizations.
#'
#' Nested grouping `(1 | a/b)` is expanded into one spec per level
#' (a, then a:b). `||` is preserved as a single spec rather than split into
#' multiple `|` bars, so the original user intent (one logical RE term with
#' diagonal covariance) round-trips through the parsed object.
#'
#' @param bar_term A language object: `|`(lhs, rhs) or `||`(lhs, rhs)
#' @return A list of RE specs (one per grouping level). Each spec has:
#'   - `group_var`: character, display label (colon-joined for nested)
#'   - `group_vars`: character vector, each element a column name
#'   - `group_expr`: language or NULL (set when grouping is a non-name expr)
#'   - `slope_terms`: list of language objects (slope LHS terms)
#'   - `has_intercept`: logical
#'   - `correlated`: logical (TRUE for `|`, FALSE for `||`)
#'   - `original`: the original bar language object
#' @keywords internal
#' @export
parse_bar_term <- function(bar_term) {
  stopifnot(is.call(bar_term))

  op <- bar_term[[1]]
  correlated <- identical(op, as.name("|"))

  lhs <- bar_term[[2]]
  rhs <- bar_term[[3]]

  lhs_decomp <- decompose_bar_lhs(lhs)
  group_specs <- resolve_group_rhs(rhs)

  out <- list()
  for (gs in group_specs) {
    out[[length(out) + 1L]] <- list(
      group_var     = gs$group_var,
      group_vars    = gs$group_vars,
      group_expr    = gs$group_expr,
      slope_terms   = lhs_decomp$slope_terms,
      has_intercept = lhs_decomp$has_intercept,
      correlated    = correlated,
      original      = bar_term
    )
  }
  out
}

#' Resolve the RHS of a bar term into one or more group specs
#'
#' Three cases:
#'   - simple name: `g` → one spec with `group_vars = "g"`.
#'   - nested `/` :   `a/b` → two specs (`"a"`, `c("a","b")`).
#'   - other call:    `factor(g)` or `g1:g2` → one spec carrying the
#'     language object in `group_expr` so we evaluate it at build time
#'     instead of guessing column names.
#'
#' @param rhs A language object (RHS of `|` or `||`).
#' @return List of group specs.
#' @keywords internal
resolve_group_rhs <- function(rhs) {
  # Nested grouping a/b
  if (is.call(rhs) && identical(rhs[[1]], as.name("/")) && is.name(rhs[[2]]) && is.name(rhs[[3]])) {
    parent_name <- as.character(rhs[[2]])
    child_name  <- as.character(rhs[[3]])
    return(list(
      list(
        group_var  = parent_name,
        group_vars = parent_name,
        group_expr = NULL
      ),
      list(
        group_var  = paste0(parent_name, ":", child_name),
        group_vars = c(parent_name, child_name),
        group_expr = NULL
      )
    ))
  }

  # Simple name
  if (is.name(rhs)) {
    nm <- as.character(rhs)
    return(list(list(
      group_var  = nm,
      group_vars = nm,
      group_expr = NULL
    )))
  }

  # General expression: evaluate at build time
  label <- paste(deparse(rhs, width.cutoff = 500L), collapse = "")
  list(list(
    group_var  = label,
    group_vars = character(0),
    group_expr = rhs
  ))
}

#' Check if a formula term includes an implicit intercept
#'
#' Pattern-matches on the formula AST directly — no deparse/regex.
#'
#' @param term A language object (LHS of a bar term)
#' @return logical
#' @keywords internal
has_implicit_intercept <- function(term) {
  if (is.numeric(term) && term == 0) return(FALSE)
  if (!is.call(term) || length(term) != 3) return(TRUE)
  op <- term[[1]]

  if (identical(op, as.name("+")) && is.numeric(term[[2]]) && term[[2]] == 0) return(FALSE)
  if (identical(op, as.name("+"))) {
    lhs <- term[[2]]
    if (is.call(lhs) && identical(lhs[[1]], as.name("-")) &&
        length(lhs) == 2 && is.numeric(lhs[[2]]) && lhs[[2]] == 1) return(FALSE)
  }
  TRUE
}

#' Decompose the LHS of a bar term into intercept flag + slope language objects
#'
#' Walks the additive chain of the LHS and separates numeric intercept
#' indicators (0, 1, -1) from slope term expressions. Returns language
#' objects, not deparsed strings.
#'
#' @param lhs A language object (LHS of a bar term)
#' @return A list with `has_intercept` (logical) and `slope_terms` (list of language objects)
#' @keywords internal
decompose_bar_lhs <- function(lhs) {
  terms <- collect_additive_terms(lhs)
  has_intercept <- TRUE
  slopes <- list()

  for (term in terms) {
    if (is.numeric(term)) {
      if (term == 0) has_intercept <- FALSE
      # 1 = intercept, skip from slopes
    } else if (is.call(term) && identical(term[[1]], as.name("-")) &&
               length(term) == 2 && is.numeric(term[[2]]) && term[[2]] == 1) {
      has_intercept <- FALSE
    } else {
      slopes <- c(slopes, list(term))
    }
  }

  list(has_intercept = has_intercept, slope_terms = slopes)
}

#' Collect additive terms from a + chain
#'
#' Recursively flattens `a + b + c` into `list(a, b, c)`.
#'
#' @param expr A language object
#' @return A list of language objects (individual terms)
#' @keywords internal
collect_additive_terms <- function(expr) {
  if (is.call(expr) && identical(expr[[1]], as.name("+"))) {
    c(collect_additive_terms(expr[[2]]), collect_additive_terms(expr[[3]]))
  } else {
    list(expr)
  }
}

#' Render a slope spec as a display string
#'
#' Used by the print method. Derives the textual form on demand from
#' the stored language objects so we never store deparsed slope text.
#'
#' @keywords internal
format_re_lhs <- function(re) {
  pieces <- character(0)
  if (re$has_intercept) pieces <- c(pieces, "1") else pieces <- c(pieces, "0")
  for (s in re$slope_terms) {
    pieces <- c(pieces, paste(deparse(s, width.cutoff = 500L), collapse = ""))
  }
  paste(pieces, collapse = " + ")
}

# ============================================================================
# Main formula parser
# ============================================================================

#' Parse a mixed-model formula
#'
#' Decomposes a formula into fixed effects and random effects by walking
#' the formula's abstract syntax tree. This is structural recursion with
#' pattern matching on bar terms — no string manipulation.
#'
#' @param formula A formula object (e.g., `y ~ x + (1 | group)`)
#' @return A list with:
#'   - `response`: character, the response variable label (`NULL` if none)
#'   - `response_expr`: language object, the LHS of `~` (`NULL` if none)
#'   - `fixed_formula`: formula, the fixed-effects-only formula
#'   - `random_effects`: list of parsed RE specifications
#'   - `n_re_terms`: integer, number of RE terms
#'   - `original`: the original formula
#'
#' @examples
#' pf <- tulpa_parse_formula(y ~ x1 + x2 + (1 | group) + (x1 || site))
#' pf$response        # "y"
#' pf$fixed_formula   # y ~ x1 + x2
#'
#' @export
tulpa_parse_formula <- function(formula) {
  stopifnot(inherits(formula, "formula"))

  formula_env <- environment(formula)

  if (length(formula) == 3) {
    response_expr <- formula[[2]]
    rhs <- formula[[3]]
  } else {
    response_expr <- NULL
    rhs <- formula[[2]]
  }
  response <- if (is.null(response_expr)) NULL else paste(deparse(response_expr, width.cutoff = 500L), collapse = "")

  bars <- findbars(rhs)

  random_effects <- list()
  if (length(bars) > 0) {
    for (bar in bars) {
      random_effects <- c(random_effects, parse_bar_term(bar))
    }
  }

  rhs_clean <- nobars(rhs)

  fixed_rhs <- if (is.null(rhs_clean)) 1 else rhs_clean
  fixed_formula <- if (is.null(response_expr)) {
    as.formula(call("~", fixed_rhs), env = formula_env)
  } else {
    as.formula(call("~", response_expr, fixed_rhs), env = formula_env)
  }

  structure(
    list(
      response       = response,
      response_expr  = response_expr,
      fixed_formula  = fixed_formula,
      random_effects = random_effects,
      n_re_terms     = length(random_effects),
      original       = formula
    ),
    class = "tulpa_parsed_formula"
  )
}

#' Build model matrices from a parsed formula
#'
#' Takes a parsed formula and data frame, and constructs:
#' - The fixed-effects design matrix X
#' - An offset vector (or NULL) extracted from `offset(...)` terms
#' - RE group index vectors
#' - RE slope matrices (if applicable)
#'
#' @param parsed A `tulpa_parsed_formula` object
#' @param data A data frame
#' @return A list with:
#'   - `y`: response vector (or NULL)
#'   - `X`: fixed-effects design matrix
#'   - `offset`: numeric vector or NULL
#'   - `re_terms`: list of RE data structures (group indices, slope matrices)
#'
#' @export
tulpa_build_model_data <- function(parsed, data) {
  stopifnot(inherits(parsed, "tulpa_parsed_formula"))

  formula_env <- environment(parsed$original) %||% parent.frame()

  # Response: evaluate the language object directly. Handles bare names,
  # cbind(succ, fail), backtick-quoted identifiers, all in one path.
  y <- NULL
  if (!is.null(parsed$response_expr)) {
    y <- tryCatch(
      eval(parsed$response_expr, envir = data, enclos = formula_env),
      error = function(e) NULL
    )
    if (is.null(y)) {
      stop("Response '", parsed$response, "' not found in data", call. = FALSE)
    }
  }

  # Fixed-effects design matrix + offset extraction.
  # model.frame parses offset() terms and exposes them via model.offset();
  # model.matrix excludes them from the design.
  mf <- model.frame(parsed$fixed_formula, data, na.action = na.pass)
  X <- model.matrix(parsed$fixed_formula, mf)
  off <- stats::model.offset(mf)

  # Random effects
  re_terms <- list()
  for (i in seq_along(parsed$random_effects)) {
    re_spec <- parsed$random_effects[[i]]

    group_factor <- resolve_group_factor(re_spec, data, formula_env)
    group_idx <- as.integer(group_factor)
    n_groups <- nlevels(group_factor)

    n_coefs <- if (re_spec$has_intercept) 1L else 0L
    slope_matrix <- NULL
    slope_names <- character(0)

    slope_lang <- re_spec$slope_terms
    if (length(slope_lang) > 0) {
      slope_rhs <- slope_lang[[1]]
      if (length(slope_lang) > 1) {
        for (s in slope_lang[-1]) slope_rhs <- call("+", slope_rhs, s)
      }
      slope_formula <- as.formula(call("~", slope_rhs), env = formula_env)
      slope_mf <- model.frame(slope_formula, data, na.action = na.pass)
      slope_mat <- model.matrix(slope_formula, slope_mf)
      intercept_col <- which(colnames(slope_mat) == "(Intercept)")
      if (length(intercept_col) > 0) {
        slope_mat <- slope_mat[, -intercept_col, drop = FALSE]
      }
      if (ncol(slope_mat) > 0) {
        slope_matrix <- slope_mat
        slope_names <- colnames(slope_mat)
        n_coefs <- n_coefs + ncol(slope_mat)
      }
    }

    re_terms[[i]] <- list(
      group_var     = re_spec$group_var,
      group_vars    = re_spec$group_vars,
      group_idx     = group_idx,
      n_groups      = n_groups,
      n_coefs       = n_coefs,
      has_intercept = re_spec$has_intercept,
      slope_matrix  = slope_matrix,
      slope_names   = slope_names,
      correlated    = re_spec$correlated,
      levels        = levels(group_factor)
    )
  }

  list(
    y           = y,
    X           = X,
    offset      = off,
    re_terms    = re_terms,
    n_obs       = nrow(X),
    n_fixed     = ncol(X),
    n_re_terms  = length(re_terms),
    fixed_names = colnames(X)
  )
}

#' Resolve a parsed RE spec to a grouping factor
#'
#' Three paths, mirroring `resolve_group_rhs`:
#'   - single column name: take `data[[name]]`.
#'   - multiple column names (nested `a/b`): `interaction(data[names])`.
#'   - language expression: evaluate against `data` (e.g., `factor(g)`).
#'
#' @keywords internal
resolve_group_factor <- function(re_spec, data, env) {
  if (!is.null(re_spec$group_expr)) {
    val <- eval(re_spec$group_expr, envir = data, enclos = env)
    return(as.factor(val))
  }
  gvars <- re_spec$group_vars
  if (length(gvars) > 1L) {
    missing <- setdiff(gvars, names(data))
    if (length(missing)) {
      stop("Grouping variable(s) ", paste(shQuote(missing), collapse = ", "),
           " not found in data", call. = FALSE)
    }
    return(interaction(data[gvars], drop = TRUE))
  }
  if (!gvars %in% names(data)) {
    stop("Grouping variable '", gvars, "' not found in data", call. = FALSE)
  }
  as.factor(data[[gvars]])
}

#' @export
print.tulpa_parsed_formula <- function(x, ...) {
  cat("tulpa parsed formula\n")
  cat("  Response:", x$response %||% "(none)", "\n")
  cat("  Fixed:", paste(deparse(x$fixed_formula), collapse = ""), "\n")
  cat("  Random effects:", x$n_re_terms, "term(s)\n")
  for (re in x$random_effects) {
    bar <- if (re$correlated) "|" else "||"
    cat("    (", format_re_lhs(re), bar, re$group_var, ")\n")
  }
  invisible(x)
}
