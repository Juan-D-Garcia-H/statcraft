// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file mcglm.hpp
/// @brief Umbrella header of the MCGLM engine (header-only, C++20, Eigen3).
///
/// Typical use: build the response specifications, obtain starting values with
/// mcglm::initial_values() and fit with mcglm::fit(); the result is carried by
/// mcglm::FitResult and the starting values by mcglm::Params.
#pragma once
#include "extras.hpp"
#include "fit.hpp"
#include "initial.hpp"
