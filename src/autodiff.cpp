// autodiff.cpp
// Implementation of global tape (deprecated) and backward-compatible functions

#include "autodiff.h"

namespace tulpa {
namespace ad {

// Define the global tape (deprecated - for backward compatibility only)
// New code should use TapeScope or create_tape()/delete_tape()
Tape* global_tape = nullptr;

// Backward-compatible constructor using current tape
// Uses thread-local current_tape() which is set by TapeScope
// Falls back to global_tape for legacy code
Var::Var(double value) : tape(current_tape() ? current_tape() : global_tape) {
  if (tape != nullptr) {
    idx = tape->add_node(value);
  } else {
    idx = 0;
  }
}

} // namespace ad
} // namespace tulpa
