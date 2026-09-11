// Convenience aggregate header for the Phase 0 execution foundation.
// Prefer including the specific headers you need in performance-sensitive
// translation units; this is meant for tests, benchmarks, and tutorials.
#pragma once

#include "cfe/backend/cpu/serial.hpp"
#include "cfe/backend/cpu/threaded.hpp"
#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/component_counts.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/field/layout.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/math/operations.hpp"
