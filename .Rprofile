## devtools::load_all() delegates to pkgbuild, which by default replaces R's own
## compiler flags with `-UNDEBUG -Wall -pedantic -g -O0`. At -O0 gcc emits one
## comdat section per template instantiation and merges none of them, so the
## inference kernels assemble tens of thousands of sections and the objects run
## to tens of megabytes each -- src/Makevars.win's -Wa,-mbig-obj is what keeps
## that inside the COFF object format, but the build is still slow and the DLL
## unoptimized.
##
## Switching the override off leaves R's Makeconf flags in place, so the DLL
## load_all() compiles is built the way `R CMD INSTALL` builds it. It also
## leaves R_MAKEVARS_USER pointing at ~/.R/Makevars.win, whose ccache wrappers
## pkgbuild's replacement makevars would otherwise drop.
##
## R reads this file only when the session STARTS in the package root and
## startup files are not skipped, so a build driven from another directory, or
## under --vanilla, gets pkgbuild's debug flags. The package has to build under
## those too; that is what the Makevars.win flag is for.
options(pkg.build_extra_flags = FALSE)
