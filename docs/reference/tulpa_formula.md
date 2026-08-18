# Formula parsing for tulpa models

Parses mixed-model formulas by walking the formula's abstract syntax
tree. R formulas are already parse trees – we do structural recursion to
find random effect terms (`|` nodes) and separate them from fixed
effects.

## Value

The formula helpers documented in this family return parsed-formula
structures (lists describing the fixed effects, random-effect terms, and
latent blocks); see each function's own help page.
