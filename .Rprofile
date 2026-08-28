## devtools::load_all() delegates to pkgbuild, which by default overrides the
## compiler flags with `-g -O0`. At -O0 gcc emits one comdat section per
## template instantiation with no merging, and the heavy inference kernels pass
## the 32767-section limit of the Windows COFF object format: `aghq_re.cpp`
## assembles 37080 sections and the assembler stops with "file too big". The
## same file at R's own -O2 fits well inside the limit.
##
## Switching the override off leaves R's Makeconf flags in place, so the DLL
## load_all() compiles is built the way `R CMD INSTALL` builds it. It also
## leaves R_MAKEVARS_USER pointing at ~/.R/Makevars.win, whose ccache wrappers
## pkgbuild's replacement makevars would otherwise drop.
options(pkg.build_extra_flags = FALSE)
