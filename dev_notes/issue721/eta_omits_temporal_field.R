# Does the synthesised linear predictor that feeds WAIC / LOO and predict()
# carry a nested fit's latent temporal field?
#
# Run from the tulpa repo root:  Rscript dev_notes/issue721/eta_omits_temporal_field.R
#
# Uses vignettes/temporal-models.Rmd's own data and fits, executed up to its
# evidence chunk, so the fixture is the one the package documents.
#
# If the field is carried, mean(eta) - X beta_hat tracks the simulated trend over
# time. If it is not, it is Monte Carlo noise around zero and predict(type =
# "link") equals X beta_hat.

suppressMessages({ library(tulpa); library(knitr) })

code <- tempfile(fileext = ".R")
purl("vignettes/temporal-models.Rmd", output = code, quiet = TRUE, documentation = 0)
src <- readLines(code)
cut <- grep("evidence_temporal <- lse", src, fixed = TRUE)[1L]
stopifnot(!is.na(cut))
env <- new.env()
invisible(capture.output(eval(parse(text = src[seq_len(cut + 1L)]), envir = env)))
fit  <- env$fit    # nested fit with temporal = temporal_rw1("time")
m_nt <- env$m_nt   # same formula, no temporal field, mode = "laplace"
df   <- env$df

cat("tulpa", as.character(packageVersion("tulpa")), "\n\n")

eta <- tulpa:::.tulpa_eta_draws(fit, synth_seed = 285603L)
X   <- model.matrix(~ x, data = df)
Xb  <- as.vector(X %*% coef(fit)[colnames(X)])
em  <- colMeans(eta)
by_t <- tapply(em - Xb, df$time, mean)

cat(sprintf("max |mean eta - X beta_hat|        = %.4f\n", max(abs(em - Xb))))
cat(sprintf("sd over time of (mean eta - X beta) = %.4f  (%d time points)\n",
            sd(by_t), length(by_t)))
cat(sprintf("cor(that, simulated trend)          = %.3f\n",
            cor(as.numeric(by_t), as.numeric(env$trend))))
cat(sprintf("max |predict(link) - X beta_hat|    = %.4g\n\n",
            max(abs(predict(fit, type = "link") - Xb))))

cat(sprintf("evidence, logLik(): temporal %.2f vs no temporal %.2f\n",
            as.numeric(logLik(fit)), as.numeric(logLik(m_nt))))
print(compare_models(no_temporal = m_nt, temporal = fit, criterion = "waic"))
