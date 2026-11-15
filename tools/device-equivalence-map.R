device_equivalence_map <- function() {
  entries <- list(
    nngp_cholesky=c('device-nngp','NNGP moments agree across the measured size gates'),
    axpy=c('device-runtime','NVRTC compiles shipped source and launches a driver kernel'),
    sparse_cholesky=c('device-sparse-fits','complete sparse fits agree below and above the automatic factor gate'),
    sparse_selected_inverse=c('device-sparse','cuDSS refactors and selects the inverse on both sides of the measured gate'),
    likelihood=c('device-likelihood','automatic family routing agrees on both sides of every measured crossover'),
    likelihood_weights=c('device-likelihood','automatic family routing agrees on both sides of every measured crossover'),
    dense_gp=c('device-dense','dense CPU and CUDA agree across their measured gates'),
    temporal_gp=c('device-dense','temporal GP and TVC values and gradients agree on CUDA with binding nuggets'),
    rsr_gibbs=c('device-dense','projected RSR conditionals agree with R and CUDA'),
    gp_rsr_gibbs=c('device-dense','projected RSR conditionals agree with R and CUDA'),
    dense_spec=c('device-dense','dense CPU and CUDA agree across their measured gates'),
    dense_precision=c('device-dense','dense CPU and CUDA agree across their measured gates'),
    hsgp_product=c('device-dense','HSGP products agree on both sides of each basis-size gate'),
    hsgp_transpose=c('device-dense','HSGP products agree on both sides of each basis-size gate'),
    hsgp_crossprod=c('device-dense','HSGP products agree on both sides of each basis-size gate'),
    hsgp_cross=c('device-dense','rectangular HSGP products use their own measured shape gate'),
    dense_adjoint=c('device-dense','Cholesky adjoints agree with finite differences and CUDA across the gate'),
    laplace_dense=c('device-laplace-batch','dense Laplace factors agree on both sides of every worker gate'),
    laplace_full_csc=c('device-laplace-batch','full CSC Hessians use their own CPU crossover'),
    laplace_dense_inverse=c('device-laplace-batch','dense inverse selection follows its own measured crossover'),
    laplace_scatter=c('device-scatter','weighted assembly selects the measured route on both sides of each gate'),
    sparse_uniform=c('device-laplace-batch','uniform sparse factors agree across every worker threshold'))
  data.frame(operation=names(entries),
    file=vapply(entries,function(x)paste0('test-',x[[1L]],'.R'),''),
    test=vapply(entries,`[[`,'',2L),row.names=NULL)
}

device_test_names <- function(file) {
  names <- character()
  walk <- function(node) {
    if (!is.call(node) && !is.expression(node)) return(invisible(NULL))
    if (is.call(node) && identical(node[[1L]],as.name('test_that')) &&
        length(node)>1L && is.character(node[[2L]])) names <<- c(names,node[[2L]])
    for (j in seq_along(node)) {
      if (identical(node[[j]],quote(expr=))) next
      walk(node[[j]])
    }
  }
  walk(parse(file))
  names
}
