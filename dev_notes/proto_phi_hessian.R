# Standalone check of the dispersion (log phi) Hessian blocks derived in
# dev_notes/laplace_phi_hessian.md. Base R, no tulpa code: agreement with a
# central difference of the analytic gradient (itself checked against the
# objective) is evidence about the MATH of the phi column and phi diagonal.
#
# Two families, chi = (theta, psi), scalar RE block:
#   gaussian  W = 1/phi constant in eta (dwde = 0): exercises L1/L2, DW/DW2,
#             Sc1/Sc2, the mode's phi-motion, dV/dpsi, dR/dpsi -- NOT the through-
#             mode weight terms. Analytic family derivatives, so it checks to 1e-10.
#   nb2       working == observed, dwde != 0: adds the C / dr / d2x-dphi2 terms and
#             the mixed DWde. First phi-derivatives analytic (the .FAMILY_DPHI
#             forms); the four NEW second-order derivatives (L2, Sc2, DW2, DWde) by
#             one central difference of those, so it pins the ASSEMBLY to ~1e-6
#             (the closed forms are FD-checked separately when wired).

options(digits = 10)

# ---- families -------------------------------------------------------------
gaussian_fam <- function() list(
  kind = "gaussian",
  gen  = function(eta) rnorm(length(eta), eta, sqrt(0.5)),
  ll   = function(y, eta, phi) dnorm(y, eta, sqrt(phi), log = TRUE),
  score= function(y, eta, phi) (y - eta) / phi,
  W    = function(y, eta, phi) rep(1 / phi, length(eta)),
  dwde = function(y, eta, phi) rep(0, length(eta)),
  d2wde= function(y, eta, phi) rep(0, length(eta)),
  L1   = function(y, eta, phi) -0.5 / phi + 0.5 * (y - eta)^2 / phi^2,
  Sc1  = function(y, eta, phi) -(y - eta) / phi^2,
  DW   = function(y, eta, phi) rep(-1 / phi^2, length(eta)),
  phi0 = 0.55)

nb2_fam <- function() {
  ll <- function(y, eta, phi) { mu <- exp(eta)
    lgamma(y + phi) - lgamma(phi) - lgamma(y + 1) + phi * log(phi) -
      phi * log(phi + mu) + y * log(mu) - y * log(phi + mu) }
  score <- function(y, eta, phi) { mu <- exp(eta); phi * (y - mu) / (mu + phi) }
  W  <- function(y, eta, phi) { mu <- exp(eta); mu * phi * (y + phi) / (mu + phi)^2 }
  dwde <- function(y, eta, phi) { mu <- exp(eta); s <- mu + phi
    mu * phi * (y + phi) * (phi - mu) / s^3 }
  d2wde <- function(y, eta, phi) { mu <- exp(eta); s <- mu + phi
    phi * (y + phi) * mu * (mu^2 - 4 * mu * phi + phi^2) / s^4 }
  L1 <- function(y, eta, phi) { mu <- exp(eta)
    digamma(y + phi) - digamma(phi) + log(phi) - log(phi + mu) + 1 - (phi + y)/(phi + mu) }
  Sc1 <- function(y, eta, phi) { mu <- exp(eta); (y - mu) * mu / (mu + phi)^2 }
  DW  <- function(y, eta, phi) { mu <- exp(eta)
    mu * (y * (mu - phi) + 2 * phi * mu) / (mu + phi)^3 }
  list(kind = "nb2", gen = function(eta) rnbinom(length(eta), size = 2.5, mu = exp(eta)),
       ll = ll, score = score, W = W, dwde = dwde, d2wde = d2wde,
       L1 = L1, Sc1 = Sc1, DW = DW, phi0 = 2.5)
}

# Second-order family derivatives: analytic where cheap (gaussian), else one
# central difference of the analytic first-order forms (nb2).
attach_second <- function(fam) {
  if (fam$kind == "gaussian") {
    fam$L2   <- function(y, eta, phi)  0.5 / phi^2 - (y - eta)^2 / phi^3
    fam$Sc2  <- function(y, eta, phi)  2 * (y - eta) / phi^3
    fam$DW2  <- function(y, eta, phi) rep(2 / phi^3, length(eta))
    fam$DWde <- function(y, eta, phi) rep(0, length(eta))
  } else {
    hp <- 1e-5; he <- 1e-5
    fam$L2  <- function(y, eta, phi) (fam$L1(y, eta, phi + hp) - fam$L1(y, eta, phi - hp)) / (2*hp)
    fam$Sc2 <- function(y, eta, phi) (fam$Sc1(y, eta, phi + hp) - fam$Sc1(y, eta, phi - hp)) / (2*hp)
    fam$DW2 <- function(y, eta, phi) (fam$DW(y, eta, phi + hp) - fam$DW(y, eta, phi - hp)) / (2*hp)
    fam$DWde<- function(y, eta, phi) (fam$DW(y, eta + he, phi) - fam$DW(y, eta - he, phi)) / (2*he)
  }
  fam
}

# ---- one model + all checks for a family ----------------------------------
run_family <- function(fam, seed = 11) {
  set.seed(seed)
  G <- 8L; per <- 8L; n <- G * per; grp <- rep(seq_len(G), each = per)
  p_fix <- 2L
  X <- cbind(1, rnorm(n)); Z <- outer(seq_len(n), seq_len(G), function(i,g) as.numeric(grp[i]==g))
  A <- cbind(X, Z); nx <- ncol(A); re_idx <- p_fix + seq_len(G)
  P_beta <- diag(1e-4, p_fix)
  eta_true <- as.numeric(X %*% c(0.3, 0.5)) + rnorm(G, 0, 0.7)[grp]
  y <- fam$gen(eta_true)

  Pmat <- function(theta) as.matrix(Matrix::bdiag(P_beta, diag(exp(-2*theta), G)))
  inner <- function(theta, phi) {                       # Newton on the joint mode
    P <- Pmat(theta); x <- rep(0, nx)
    for (it in 1:200) {
      eta <- as.numeric(A %*% x)
      g <- as.numeric(crossprod(A, fam$score(y, eta, phi))) - as.numeric(P %*% x)
      H <- crossprod(A, fam$W(y, eta, phi) * A) + P
      step <- solve(H, g); x <- x + step
      if (max(abs(step)) < 1e-13) break
    }
    eta <- as.numeric(A %*% x)
    list(x = x, eta = eta, H = crossprod(A, fam$W(y, eta, phi) * A) + P, P = P)
  }
  logmarg <- function(chi) {
    theta <- chi[1]; phi <- exp(chi[2]); f <- inner(theta, phi)
    ldH <- as.numeric(determinant(f$H, TRUE)$modulus); ldP <- as.numeric(determinant(f$P, TRUE)$modulus)
    sum(fam$ll(y, f$eta, phi)) - 0.5 * sum(f$x * (f$P %*% f$x)) + 0.5 * ldP - 0.5 * ldH
  }
  base_q <- function(chi) {
    theta <- chi[1]; phi <- exp(chi[2]); f <- inner(theta, phi)
    H <- f$H; Hinv <- solve(H); x <- f$x; eta <- f$eta
    Om <- exp(-2*theta); Sig <- exp(2*theta); dSigma <- 2*Sig
    s <- rowSums((A %*% Hinv) * A); dw <- fam$dwde(y, eta, phi)
    u <- as.numeric(Hinv %*% crossprod(A, dw * s)); b <- x[re_idx]
    R <- sum(b^2); V <- sum(diag(Hinv)[re_idx]); C <- sum(b * u[re_idx]); M0 <- R + V - C
    list(theta=theta, phi=phi, H=H, Hinv=Hinv, x=x, eta=eta, Om=Om, Sig=Sig,
         dSigma=dSigma, s=s, dw=dw, u=u, b=b, M0=M0, S=0.5*(Om*M0*Om - G*Om))
  }
  grad_an <- function(chi) {
    q <- base_q(chi); phi <- q$phi
    L1 <- fam$L1(y,q$eta,phi); DW <- fam$DW(y,q$eta,phi); Sc1 <- fam$Sc1(y,q$eta,phi)
    dxdphi <- as.numeric(q$Hinv %*% crossprod(A, Sc1)); deta_dphi <- as.numeric(A %*% dxdphi)
    g_pre <- sum(L1) - 0.5 * sum(q$s * (DW + q$dw * deta_dphi))
    c(q$dSigma * q$S, phi * g_pre)
  }
  hess_an <- function(chi) {
    q <- base_q(chi); phi <- q$phi; eta <- q$eta; Hinv <- q$Hinv; Om <- q$Om; s <- q$s; dw <- q$dw
    L1<-fam$L1(y,eta,phi); L2<-fam$L2(y,eta,phi); Sc1<-fam$Sc1(y,eta,phi); Sc2<-fam$Sc2(y,eta,phi)
    DW<-fam$DW(y,eta,phi); DW2<-fam$DW2(y,eta,phi); DWde<-fam$DWde(y,eta,phi); d2wde<-fam$d2wde(y,eta,phi)
    v_r <- crossprod(A, dw*s)
    col <- function(dP, J) {  # shared inner assembly for a column given dP, mode Jacobian J
      etad <- as.numeric(A %*% J); dW <- dw*etad + 0
      list(etad=etad)
    }
    # theta column
    dSig<-q$dSigma; dOm <- -Om*dSig*Om; dP_t<-matrix(0,nx,nx); diag(dP_t)[re_idx]<-dOm
    J_t <- -as.numeric(Hinv %*% (dP_t %*% q$x)); etad_t<-as.numeric(A%*%J_t)
    dH_t <- crossprod(A,(dw*etad_t)*A)+dP_t; dHi_t <- -Hinv%*%dH_t%*%Hinv
    ds_t <- rowSums((A%*%dHi_t)*A); dr_t <- (d2wde*etad_t)*s + dw*ds_t
    du_t <- as.numeric(dHi_t%*%v_r)+as.numeric(Hinv%*%crossprod(A,dr_t))
    dR_t<-2*sum(q$b*J_t[re_idx]); dV_t<-sum(diag(dHi_t)[re_idx])
    dC_t<-sum(J_t[re_idx]*q$u[re_idx]+q$b*du_t[re_idx]); dM0_t<-dR_t+dV_t-dC_t
    dS_t<-0.5*(dOm*q$M0*Om+Om*q$M0*dOm-G*dOm)+0.5*(Om*dM0_t*Om)
    H_tt <- dSig*dS_t + (4*exp(2*chi[1]))*q$S
    # psi column (dP=0)
    J_p <- phi*as.numeric(Hinv%*%crossprod(A,Sc1)); etad_p<-as.numeric(A%*%J_p)
    dW_p <- dw*etad_p + phi*DW; dH_p <- crossprod(A,dW_p*A); dHi_p <- -Hinv%*%dH_p%*%Hinv
    ds_p <- rowSums((A%*%dHi_p)*A); dr_p <- (d2wde*etad_p + phi*DWde)*s + dw*ds_p
    du_p <- as.numeric(dHi_p%*%v_r)+as.numeric(Hinv%*%crossprod(A,dr_p))
    dR_p<-2*sum(q$b*J_p[re_idx]); dV_p<-sum(diag(dHi_p)[re_idx])
    dC_p<-sum(J_p[re_idx]*q$u[re_idx]+q$b*du_p[re_idx]); dM0_p<-dR_p+dV_p-dC_p
    H_tp <- dSig*(0.5*(Om*dM0_p*Om))
    # psi diagonal
    dxdphi<-as.numeric(Hinv%*%crossprod(A,Sc1)); deta_dphi<-as.numeric(A%*%dxdphi)
    grad_phi <- phi*(sum(L1)-0.5*sum(s*(DW+dw*deta_dphi)))
    dA_term <- sum(Sc1*etad_p + phi*L2)
    d_ddw_dpsi <- d2wde*etad_p + phi*DWde                      # d(dw)/dpsi
    rhs_Sc1 <- (-DW)*etad_p + phi*Sc2                          # dSc1/dpsi (Sc1_de = -DW)
    d_detadphi_dpsi <- as.numeric(A %*% (as.numeric(dHi_p%*%crossprod(A,Sc1)) +
                                         as.numeric(Hinv%*%crossprod(A,rhs_Sc1))))
    dB_term <- sum(ds_p*(DW+dw*deta_dphi) +
                   s*(DWde*etad_p + phi*DW2 + d_ddw_dpsi*deta_dphi + dw*d_detadphi_dpsi))
    H_pp <- grad_phi + phi*(dA_term - 0.5*dB_term)
    matrix(c(H_tt,H_tp,H_tp,H_pp),2,2)
  }
  fd_grad <- function(chi,h=1e-5) vapply(seq_along(chi),function(j){cp<-chi;cp[j]<-cp[j]+h;cm<-chi;cm[j]<-cm[j]-h
    (logmarg(cp)-logmarg(cm))/(2*h)},numeric(1))
  fd_hess <- function(chi,h=1e-5){k<-length(chi);Hf<-matrix(0,k,k)
    for(j in 1:k){cp<-chi;cp[j]<-cp[j]+h;cm<-chi;cm[j]<-cm[j]-h;Hf[,j]<-(grad_an(cp)-grad_an(cm))/(2*h)}
    (Hf+t(Hf))/2}

  chi0 <- c(log(0.7), log(fam$phi0))
  ga<-grad_an(chi0); gf<-fd_grad(chi0); Ha<-hess_an(chi0); Hf<-fd_hess(chi0)
  cat(sprintf("%-9s grad vs obj-FD = %.2e   Hessian vs grad-FD = %.2e   (H_tp = %.4f)\n",
              fam$kind, max(abs(ga-gf)/pmax(abs(gf),1e-8)),
              max(abs(Ha-Hf)/pmax(abs(Hf),1e-6)), Ha[1,2]))
}

cat("phi-Hessian proto (chi = theta, psi)\n====================================\n")
run_family(attach_second(gaussian_fam()))
run_family(attach_second(nb2_fam()))
