# Internal validation helpers

Small shared helpers used by `validate_*()` functions across the spatial
/ temporal / SVC / TVC specs. Centralised here to keep the per-spec
validators thin and prevent drift between near-identical
column-existence checks and coordinate preparation blocks.
