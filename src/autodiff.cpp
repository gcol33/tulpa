// autodiff.cpp
// Global-tape definition and the tape-inferring Var constructors.

#include "autodiff.h"

namespace tulpa {
namespace ad {

// Fallback tape used when no TapeScope is active on the current thread.
Tape* global_tape = nullptr;

// Constructs a Var on the thread-local current_tape() when a TapeScope is
// active, otherwise on global_tape.
Var::Var(double value) : tape(current_tape() ? current_tape() : global_tape) {
  if (tape != nullptr) {
    idx = tape->add_node(value);
  } else {
    idx = 0;
  }
}

} // namespace ad
} // namespace tulpa
