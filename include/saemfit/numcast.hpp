// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file numcast.hpp
/// @brief Explicit numeric conversions (no -Wold-style-cast / -Wsign-conversion noise; values are not range-checked).
#pragma once
#include <Eigen/Core>
#include <cstddef>

namespace mcmckl {

/// Converts an integer to @c std::size_t (index of a standard container).
template <class T> constexpr std::size_t uz(T v) noexcept { return static_cast<std::size_t>(v); }
/// Converts a number to @c int.
template <class T> constexpr int toi(T v) noexcept { return static_cast<int>(v); }
/// Converts a number to @c double.
template <class T> constexpr double tod(T v) noexcept { return static_cast<double>(v); }
/// Converts an integer to @c Eigen::Index (index of an Eigen object).
template <class T> constexpr Eigen::Index tix(T v) noexcept { return static_cast<Eigen::Index>(v); }

} // namespace mcmckl
