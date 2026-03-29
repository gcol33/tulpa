#' Formula parsing for tulpa models
#'
#' @description
#' Parses mixed-model formulas by walking the formula's abstract syntax tree.
#' R formulas are already parse trees — we do structural recursion to find
#' random effect terms (`|` nodes) and separate them from fixed effects.
#'
#' This is the "reduction tree" approach: pattern-match on bar terms,
#' collect them, and rewrite the tree without them.
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
  # Leaf: symbol or literal — no bars here

if (is.name(term) || !is.language(term)) return(NULL)

  # Parenthesized expression: check if it wraps a bar term
  if (term[[1]] == as.name("(")) {
    inner <- term[[2]]
    if (is.call(inner) && (inner[[1]] == as.name("|") || inner[[1]] == as.name("||"))) {
      return(list(inner))
    }
    # Not a bar term — recurse into the parenthesized content
    return(findbars(inner))
  }

  # Bar term found directly (shouldn't happen at top level, but handle it)
  if (term[[1]] == as.name("|") || term[[1]] == as.name("||")) {
    return(list(term))
  }

  # Unary operator (e.g., `-x`): recurse into operand
  if (length(term) == 2) {
    return(findbars(term[[2]]))
  }

  # Binary operator (e.g., `+`, `*`, `~`): recurse both sides
  c(findbars(term[[2]]), findbars(term[[3]]))
}

#' Remove all bar terms from a formula's parse tree
#'
#' Recursively rewrites the formula AST, removing any `|` or `||` nodes
#' found inside parentheses. Returns the fixed-effects-only formula.
#'
#' This is structural term rewriting: we pattern-match on bar terms
#' and replace them with NULL, then clean up the tree.
#'
#' @param term A language object (formula term)
#' @return A language object with all bar terms removed, or NULL if nothing remains
#' @keywords internal
#' @export
nobars <- function(term) {
  # Leaf: unchanged
  if (is.name(term) || !is.language(term)) return(term)

  # Parenthesized expression: drop if it wraps a bar term
  if (term[[1]] == as.name("(")) {
    inner <- term[[2]]
    if (is.call(inner) && (inner[[1]] == as.name("|") || inner[[1]] == as.name("||"))) {
      return(NULL)  # Remove this entire term
    }
    nb <- nobars(inner)
    if (is.null(nb)) return(NULL)
    return(call("(", nb))
  }

  # Unary operator: recurse
  if (length(term) == 2) {
    nb <- nobars(term[[2]])
    if (is.null(nb)) return(NULL)
    return(call(deparse(term[[1]]), nb))
  }

  # Binary operator: recurse both sides, handle NULL removal
  nb_left  <- nobars(term[[2]])
  nb_right <- nobars(term[[3]])

  if (is.null(nb_left) && is.null(nb_right)) return(NULL)
  if (is.null(nb_left))  return(nb_right)
  if (is.null(nb_right)) return(nb_left)

  call(deparse(term[[1]]), nb_left, nb_right)
}

# ============================================================================
# Parse a single bar term into a structured RE specification
# ============================================================================

#' Parse a single random effect bar term
#'
#' Takes a `|` or `||` language object and extracts the grouping variable,
#' effect terms (intercept, slopes), and correlation structure.
#'
#' @param bar_term A language object: `|`(lhs, rhs) or `||`(lhs, rhs)
#' @return A list with:
#'   - `group_var`: character, the grouping variable name
#'   - `terms`: character vector of effect term expressions
#'   - `has_intercept`: logical
#'   - `correlated`: logical (TRUE for `|`, FALSE for `||`)
#'   - `original`: the original language object
#' @keywords internal
#' @export
parse_bar_term <- function(bar_term) {
  stopifnot(is.call(bar_term))

  op <- bar_term[[1]]
  correlated <- identical(op, as.name("|"))

  lhs <- bar_term[[2]]  # Effect specification (e.g., 1 + x)
  rhs <- bar_term[[3]]  # Grouping variable (e.g., group)

  # Decompose LHS into intercept + slope language objects
  lhs_decomp <- decompose_bar_lhs(lhs)

  # Handle nested grouping: (1 | a/b) → AST check for `/` call
  if (is.call(rhs) && identical(rhs[[1]], as.name("/"))) {
    parent_name <- as.character(rhs[[2]])
    child_name <- as.character(rhs[[3]])
    nested_group <- paste0(parent_name, ":", child_name)
    make_re <- function(gvar) {
      list(
        group_var = gvar,
        terms = deparse(lhs),
        slope_terms = lhs_decomp$slope_terms,
        has_intercept = lhs_decomp$has_intercept,
        correlated = correlated,
        original = bar_term
      )
    }
    return(list(make_re(parent_name), make_re(nested_group)))
  }

  # Simple grouping variable
  group_var <- if (is.name(rhs)) as.character(rhs) else deparse(rhs)

  list(list(
    group_var = group_var,
    terms = deparse(lhs),
    slope_terms = lhs_decomp$slope_terms,
    has_intercept = lhs_decomp$has_intercept,
    correlated = correlated,
    original = bar_term
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
  # bare 0 → no intercept
  if (is.numeric(term) && term == 0) return(FALSE)
  if (!is.call(term) || length(term) != 3) return(TRUE)
  op <- term[[1]]
  # 0 + x → call("+", 0, x) → no intercept

  if (identical(op, as.name("+")) && is.numeric(term[[2]]) && term[[2]] == 0) return(FALSE)
  # -1 + x → call("+", call("-", 1), x) → no intercept
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
#'   - `response`: character, the response variable name
#'   - `fixed_formula`: formula, the fixed-effects-only formula
#'   - `random_effects`: list of parsed RE specifications
#'   - `n_re_terms`: integer, number of RE terms
#'   - `original`: the original formula
#'
#' @examples
#' pf <- tulpa_parse_formula(y ~ x1 + x2 + (1 | group) + (x1 || site))
#' pf$response        # "y"
#' pf$fixed_formula   # y ~ x1 + x2
#' pf$n_re_terms      # 2
#'
#' @export
tulpa_parse_formula <- function(formula) {
  stopifnot(inherits(formula, "formula"))

  # Extract response (LHS of ~)
  if (length(formula) == 3) {
    response <- deparse(formula[[2]])
    rhs <- formula[[3]]
  } else {
    response <- NULL
    rhs <- formula[[2]]
  }

  # Find all bar terms (random effects) via AST recursion
  bars <- findbars(rhs)

  # Parse each bar term into structured RE spec
  random_effects <- list()
  if (length(bars) > 0) {
    for (bar in bars) {
      parsed <- parse_bar_term(bar)
      random_effects <- c(random_effects, parsed)
    }
  }

  # Remove bar terms to get fixed-effects formula
  rhs_clean <- nobars(rhs)

  # Reconstruct fixed-effects formula
  if (is.null(rhs_clean)) {
    # All terms were random effects — intercept-only fixed effects
    if (!is.null(response)) {
      response_expr <- parse(text = response)[[1]]
      fixed_formula <- call("~", response_expr, 1)
      fixed_formula <- as.formula(fixed_formula, env = environment(formula))
    } else {
      fixed_formula <- ~ 1
    }
  } else {
    if (!is.null(response)) {
      response_expr <- parse(text = response)[[1]]
      fixed_formula <- call("~", response_expr, rhs_clean)
      fixed_formula <- as.formula(fixed_formula, env = environment(formula))
    } else {
      fixed_formula <- call("~", rhs_clean)
      fixed_formula <- as.formula(fixed_formula, env = environment(formula))
    }
  }

  structure(
    list(
      response = response,
      fixed_formula = fixed_formula,
      random_effects = random_effects,
      n_re_terms = length(random_effects),
      original = formula
    ),
    class = "tulpa_parsed_formula"
  )
}

#' Build model matrices from a parsed formula
#'
#' Takes a parsed formula and data frame, and constructs:
#' - The fixed-effects design matrix X
#' - RE group index vectors
#' - RE slope matrices (if applicable)
#'
#' @param parsed A `tulpa_parsed_formula` object
#' @param data A data frame
#' @return A list with:
#'   - `y`: response vector
#'   - `X`: fixed-effects design matrix
#'   - `re_terms`: list of RE data structures (group indices, slope matrices)
#'
#' @export
tulpa_build_model_data <- function(parsed, data) {
  stopifnot(inherits(parsed, "tulpa_parsed_formula"))

  # Response — evaluate the LHS expression in the data environment.
  # This handles both simple names (y) and calls (cbind(succ, fail), c(...)).
  y <- NULL
  if (!is.null(parsed$response)) {
    y <- tryCatch(
      eval(parse(text = parsed$response)[[1]], envir = data, enclos = parent.frame()),
      error = function(e) NULL
    )
    if (is.null(y)) {
      stop("Response variable '", parsed$response, "' not found in data", call. = FALSE)
    }
  }

  # Fixed-effects design matrix
  mf <- model.frame(parsed$fixed_formula, data, na.action = na.pass)
  X <- model.matrix(parsed$fixed_formula, mf)

  # Random effects
  re_terms <- list()
  for (i in seq_along(parsed$random_effects)) {
    re_spec <- parsed$random_effects[[i]]
    gvar <- re_spec$group_var

    # Handle interaction grouping (a:b)
    if (grepl(":", gvar)) {
      parts <- strsplit(gvar, ":")[[1]]
      group_factor <- interaction(data[parts], drop = TRUE)
    } else {
      if (!gvar %in% names(data)) {
        stop("Grouping variable '", gvar, "' not found in data", call. = FALSE)
      }
      group_factor <- as.factor(data[[gvar]])
    }

    group_idx <- as.integer(group_factor)
    n_groups <- nlevels(group_factor)

    # Parse slope terms from language objects (no deparse/regex)
    n_coefs <- if (re_spec$has_intercept) 1L else 0L
    slope_matrix <- NULL
    slope_names <- character(0)

    slope_lang <- re_spec$slope_terms  # list of language objects from decompose_bar_lhs
    if (length(slope_lang) > 0) {
      # Build slope formula from language objects: ~ term1 + term2 + ...
      slope_rhs <- slope_lang[[1]]
      if (length(slope_lang) > 1) {
        for (s in 2:length(slope_lang)) {
          slope_rhs <- call("+", slope_rhs, slope_lang[[s]])
        }
      }
      slope_formula <- as.formula(call("~", slope_rhs), env = parent.frame())
      slope_mf <- model.frame(slope_formula, data, na.action = na.pass)
      slope_mat <- model.matrix(slope_formula, slope_mf)
      # Remove intercept column from slope matrix (already counted)
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
      group_var = gvar,
      group_idx = group_idx,
      n_groups = n_groups,
      n_coefs = n_coefs,
      has_intercept = re_spec$has_intercept,
      slope_matrix = slope_matrix,
      slope_names = slope_names,
      correlated = re_spec$correlated,
      levels = levels(group_factor)
    )
  }

  list(
    y = y,
    X = X,
    re_terms = re_terms,
    n_obs = nrow(X),
    n_fixed = ncol(X),
    n_re_terms = length(re_terms),
    fixed_names = colnames(X)
  )
}

#' @export
print.tulpa_parsed_formula <- function(x, ...) {
  cat("tulpa parsed formula\n")
  cat("  Response:", x$response %||% "(none)", "\n")
  cat("  Fixed:", deparse(x$fixed_formula), "\n")
  cat("  Random effects:", x$n_re_terms, "term(s)\n")
  for (i in seq_along(x$random_effects)) {
    re <- x$random_effects[[i]]
    bar <- if (re$correlated) "|" else "||"
    cat("    (", re$terms, bar, re$group_var, ")\n")
  }
  invisible(x)
}
