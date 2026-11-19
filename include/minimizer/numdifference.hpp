// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

#ifndef NUMERICAL_DIFFERENCE_HPP
#define NUMERICAL_DIFFERENCE_HPP

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <string_view>
#include <optional>
#include <iostream>
#include "common.hpp"

namespace min {

/**
 * @brief Internal helpers for finite-difference step computation.
 */
namespace detail {

/**
 * @brief Effective positive perturbation applied to @p x for nominal step @p h.
 *
 * Returns the rounded, actually-applied step `(x + h) - x` (exact subtraction,
 * so the denominator always matches the perturbation actually applied to @p x).
 * If the nominal step is so small that `x + h == x` (that is, @p h is below
 * half an ulp of @p x), falls back to one ulp so that the perturbation never
 * silently collapses to zero.
 */
template <typename FP>
FP effective_step(const FP x, const FP h) {
    const FP x_plus = x + h;
    FP h_eff = x_plus - x;
    if (h_eff == FP(0))
        h_eff = std::nextafter(x, std::numeric_limits<FP>::infinity()) - x;
    return h_eff;
}

/**
 * @brief Effective negative perturbation magnitude for @p x and nominal @p h.
 *
 * Returns `x - (x - h)`, i.e. the rounded magnitude of the step actually
 * applied on the negative side of @p x. Falls back to one ulp (downward) if
 * the perturbation collapses to zero.
 */
template <typename FP>
FP effective_step_down(const FP x, const FP h) {
    const FP x_minus = x - h;
    FP h_eff = x - x_minus;
    if (h_eff == FP(0))
        h_eff = x - std::nextafter(x, -std::numeric_limits<FP>::infinity());
    return h_eff;
}

} // namespace detail

/**
 * @brief Forward-difference approximation of the Jacobian of @p f at @p x.
 *
 * For each coordinate i, perturbs x(i) by a step scaled with the magnitude of
 * x(i), evaluates @p f at the perturbed point, and forms the finite-difference
 * quotient against the base evaluation f(x).
 *
 * The step is computed as @p step * max(1, |x_i|) and then rounded to the
 * nearest representable value (via `x_plus - x_orig`) so that the effective
 * perturbation matches what was actually applied.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Callable evaluating the vector-valued function; must be
 *              invocable as `Vec<FP>(const Vec<FP>&)`.
 *
 * @param f    Vector-valued function to differentiate.
 * @param x    Point at which to evaluate the Jacobian.
 * @param step Relative step size (default: sqrt(eps)).
 *
 * @return Jacobian matrix of size m x n, where m = f(x).size() and n = x.size().
 */
template <typename FP = double, typename Func>
Mat<FP> jacobian_forward(const Func& f, const Vec<FP>& x,
    FP step = std::sqrt(std::numeric_limits<FP>::epsilon())) {

    const int n = static_cast<int>(x.size());
    const Vec<FP> y0 = f(x);
    Mat<FP> J(y0.size(), n);

    Vec<FP> x_temp = x;

    for (int i = 0; i < n; i++) {
        const FP x_orig = x(i);
        const FP h = detail::effective_step(
            x_orig, step * std::max(FP(1.0), std::abs(x_orig)));

        x_temp(i) = x_orig + h;
        J.col(i) = (f(x_temp) - y0) / h;

        x_temp(i) = x_orig;
    }
    return J;
}

/**
 * @brief Central-difference approximation of the Jacobian of @p f at @p x.
 *
 * For each coordinate i, evaluates @p f at `x(i) + h⁺` and `x(i) − h⁻` and
 * forms the central quotient. The step is scaled with the magnitude of x(i)
 * and rounded to the nearest representable value before use; each side is
 * rounded independently (so the quotient is divided by `h⁺ + h⁻`, which may
 * differ from `2h` by an ulp). If the rounded step collapses to zero it falls
 * back to one ulp.
 *
 * Central differences achieve O(h²) accuracy (versus O(h) for forward
 * differences), which is why the default step is eps^(1/3) rather than
 * sqrt(eps).
 *
 * @note The output dimension is inferred from the first evaluation of @p f;
 * this avoids the redundant base evaluation `f(x)`.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Callable evaluating the vector-valued function; must be
 *              invocable as `Vec<FP>(const Vec<FP>&)`.
 *
 * @param f    Vector-valued function to differentiate.
 * @param x    Point at which to evaluate the Jacobian.
 * @param step Relative step size (default: eps^(1/3)).
 *
 * @return Jacobian matrix of size m x n.
 */
template <typename FP = double, typename Func>
Mat<FP> jacobian_central(const Func& f, const Vec<FP>& x,
    FP step = std::pow(std::numeric_limits<FP>::epsilon(), FP(1.0 / 3.0))) {

    const int n = static_cast<int>(x.size());
    Mat<FP> J;
    Vec<FP> x_temp = x;

    for (int i = 0; i < n; i++) {
        const FP x_orig = x(i);
        const FP h_nom = step * std::max(FP(1.0), std::abs(x_orig));
        const FP h_plus  = detail::effective_step(x_orig, h_nom);
        const FP h_minus = detail::effective_step_down(x_orig, h_nom);

        x_temp(i) = x_orig + h_plus;
        Vec<FP> f_plus = f(x_temp);
        if (i == 0) J.resize(static_cast<int>(f_plus.size()), n);

        x_temp(i) = x_orig - h_minus;
        Vec<FP> f_minus = f(x_temp);

        J.col(i) = (f_plus - f_minus) / (h_plus + h_minus);

        x_temp(i) = x_orig;
    }
    return J;
}

/**
 * @brief Forward-difference approximation of the gradient of a scalar @p f at @p x.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Scalar-valued callable invocable as `FP(const Vec<FP>&)`.
 *
 * @param f    Scalar function to differentiate.
 * @param x    Point at which to evaluate the gradient.
 * @param step Relative step size (default: sqrt(eps)).
 *
 * @return Gradient vector of length n = x.size().
 */
template <typename FP = double, typename Func>
Vec<FP> grad_forward(const Func& f, const Vec<FP>& x,
    FP step = std::sqrt(std::numeric_limits<FP>::epsilon())) {
    FP y0 = f(x);
    const int n = static_cast<int>(x.size());
    Vec<FP> g(n);
    for (int i = 0; i < n; i++) {
        const FP h = detail::effective_step(
            x(i), step * std::max(FP(1.0), std::abs(x(i))));
        Vec<FP> x_p = x; x_p(i) += h;
        g(i) = (f(x_p) - y0) / h;
    }
    return g;
}

/**
 * @brief Central-difference approximation of the gradient of a scalar @p f at @p x.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Scalar-valued callable invocable as `FP(const Vec<FP>&)`.
 *
 * @param f    Scalar function to differentiate.
 * @param x    Point at which to evaluate the gradient.
 * @param step Relative step size (default: eps^(1/3)).
 *
 * @return Gradient vector of length n = x.size().
 */
template <typename FP = double, typename Func>
Vec<FP> grad_central(const Func& f, const Vec<FP>& x,
    FP step = std::pow(std::numeric_limits<FP>::epsilon(), FP(1.0/3.0))) {
    const int n = static_cast<int>(x.size());
    Vec<FP> g(n);
    for (int i = 0; i < n; i++) {
        const FP h_nom = step * std::max(FP(1.0), std::abs(x(i)));
        const FP h_plus  = detail::effective_step(x(i), h_nom);
        const FP h_minus = detail::effective_step_down(x(i), h_nom);
        Vec<FP> x_p = x, x_m = x;
        x_p(i) += h_plus; x_m(i) -= h_minus;
        g(i) = (f(x_p) - f(x_m)) / (h_plus + h_minus);
    }
    return g;
}

/**
 * @brief Richardson-extrapolated forward-difference gradient approximation.
 *
 * Combines two forward-difference estimates at step sizes h and h/2 via the
 * standard Richardson extrapolation formula
 * `g ≈ 2·D(h/2) − D(h)`, which cancels the leading O(h) error term and
 * yields second-order accuracy at the cost of a slightly smaller optimal
 * step size.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Scalar-valued callable invocable as `FP(const Vec<FP>&)`.
 *
 * @param f    Scalar function to differentiate.
 * @param x    Point at which to evaluate the gradient.
 * @param step Reference step size (default: 1e-5).
 *
 * @return Gradient vector of length n = x.size().
 */
template <typename FP = double, typename Func>
Vec<FP> grad_richardson_forward(const Func& f, const Vec<FP>& x,
    FP step = FP(1e-5)) {
    FP y0 = f(x);
    const int n = static_cast<int>(x.size());
    Vec<FP> g(n);
    for (int i = 0; i < n; i++) {
        FP h = step * std::max(FP(1.0), std::abs(x(i)));
        auto fwd = [&](FP s) {
            const FP s_eff = detail::effective_step(x(i), s);
            Vec<FP> x_p = x; x_p(i) += s_eff;
            return (f(x_p) - y0) / s_eff;
        };
        g(i) = FP(2.0) * fwd(h / FP(2.0)) - fwd(h);
    }
    return g;
}

/**
 * @brief Richardson-extrapolated central-difference gradient approximation.
 *
 * Combines two central-difference estimates at step sizes h and h/2 via the
 * standard Richardson extrapolation formula
 * `g ≈ (4·D(h/2) − D(h)) / 3`, which cancels the leading O(h²) error term
 * and yields fourth-order accuracy.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Scalar-valued callable invocable as `FP(const Vec<FP>&)`.
 *
 * @param f    Scalar function to differentiate.
 * @param x    Point at which to evaluate the gradient.
 * @param step Reference step size (default: 1e-3).
 *
 * @return Gradient vector of length n = x.size().
 */
template <typename FP = double, typename Func>
Vec<FP> grad_richardson_central(const Func& f, const Vec<FP>& x,
    FP step = FP(1e-3)) {
    const int n = static_cast<int>(x.size());
    Vec<FP> g(n);
    for (int i = 0; i < n; i++) {
        FP h = step * std::max(FP(1.0), std::abs(x(i)));
        auto central = [&](FP s) {
            const FP s_plus  = detail::effective_step(x(i), s);
            const FP s_minus = detail::effective_step_down(x(i), s);
            Vec<FP> x_p = x, x_m = x;
            x_p(i) += s_plus; x_m(i) -= s_minus;
            return (f(x_p) - f(x_m)) / (s_plus + s_minus);
        };
        g(i) = (FP(4.0) * central(h / FP(2.0)) - central(h)) / FP(3.0);
    }
    return g;
}

/**
 * @brief Default finite-difference step size for a given differentiation method.
 *
 * Primary template is intentionally undefined; each supported method
 * provides an explicit specialization with a static `value()` function.
 *
 * @tparam FP     Floating-point scalar type.
 * @tparam Method One of the differentiation method tags from @ref method.
 */
template <typename FP, typename Method> struct DefaultStep;

/** @brief Default step for forward differences: sqrt(eps). */
template <typename FP> struct DefaultStep<FP, method::Forward> {
    static FP value() { return std::sqrt(std::numeric_limits<FP>::epsilon()); }
};
/** @brief Default step for central differences: eps^(1/3). */
template <typename FP> struct DefaultStep<FP, method::Central> {
    static FP value() { return std::pow(std::numeric_limits<FP>::epsilon(), FP(1.0/3.0)); }
};
/** @brief Default step for Richardson-extrapolated forward differences: 1e-5. */
template <typename FP> struct DefaultStep<FP, method::RichardsonFwd> {
    static FP value() { return FP(1e-5); }
};
/** @brief Default step for Richardson-extrapolated central differences: 1e-3. */
template <typename FP> struct DefaultStep<FP, method::RichardsonCen> {
    static FP value() { return FP(1e-3); }
};

/**
 * @brief Compute the gradient of a scalar function using a compile-time method tag.
 *
 * Dispatches on the method tag at compile time and calls the corresponding
 * finite-difference routine. This overload is disabled when @p Method is
 * convertible to `std::string_view` so that the runtime string-based
 * overload below can be selected instead.
 *
 * @tparam FP     Floating-point scalar type.
 * @tparam Func   Scalar-valued callable invocable as `FP(const Vec<FP>&)`.
 * @tparam Method Differentiation method tag (see @ref method).
 *
 * @param f    Scalar function to differentiate.
 * @param x    Point at which to evaluate the gradient.
 * @param      Ignored tag parameter (used only for overload resolution).
 * @param step Step size; defaults to @ref DefaultStep for @p Method.
 *
 * @return Gradient vector of length n = x.size().
 */
template <typename FP = double, typename Func, typename Method = method::Central>
requires (!std::convertible_to<Method, std::string_view>)
Vec<FP> Gradient(const Func& f, const Vec<FP>& x,
                 Method = method::central,
                 FP step = DefaultStep<FP, Method>::value()) {
    if constexpr (std::is_same_v<Method, method::Forward>)
        return min::grad_forward(f, x, step);
    else if constexpr (std::is_same_v<Method, method::RichardsonFwd>)
        return min::grad_richardson_forward(f, x, step);
    else if constexpr (std::is_same_v<Method, method::RichardsonCen>)
        return min::grad_richardson_central(f, x, step);
    else if constexpr (std::is_same_v<Method, method::Central>)
        return min::grad_central(f, x, step);
    else
        static_assert(!std::is_same_v<Method, Method>,
            "min::Gradient: unsupported Method tag; use method::Forward, "
            "method::Central, method::RichardsonFwd or method::RichardsonCen");
}

/**
 * @brief Compute the gradient of a scalar function using a runtime method name.
 *
 * Recognized names:
 *  - `"forward"`            -> forward differences
 *  - `"central"`            -> central differences (default)
 *  - `"richardson_forward"` -> Richardson-extrapolated forward differences
 *  - `"richardson_central"` -> Richardson-extrapolated central differences
 *
 * Any other string is treated as `"central"` and, unless @p verbose is false,
 * a warning is written to `std::cerr`.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Scalar-valued callable invocable as `FP(const Vec<FP>&)`.
 *
 * @param f          Scalar function to differentiate.
 * @param x          Point at which to evaluate the gradient.
 * @param method_str Name of the differentiation method.
 * @param step       Optional step size; when empty, the method's default is used.
 * @param verbose    When false, suppresses the std::cerr warning for
 *                   unrecognized method names.
 *
 * @return Gradient vector of length n = x.size().
 */
template <typename FP = double, typename Func>
Vec<FP> Gradient(const Func& f, const Vec<FP>& x,
                 std::string_view method_str,
                 std::optional<FP> step = std::nullopt,
                 bool verbose = true) {
    if (method_str == "forward")
        return min::Gradient(f, x, method::forward,
                        step.value_or(DefaultStep<FP, method::Forward>::value()));
    if (method_str == "richardson_forward")
        return min::Gradient(f, x, method::richardson_forward,
                        step.value_or(DefaultStep<FP, method::RichardsonFwd>::value()));
    if (method_str == "richardson_central")
        return min::Gradient(f, x, method::richardson_central,
                        step.value_or(DefaultStep<FP, method::RichardsonCen>::value()));
    if (method_str != "central" && verbose)
        std::cerr << "[Gradient] unknown method '" << method_str << "'; falling back to central.\n";
    return min::Gradient(f, x, method::central,
                    step.value_or(DefaultStep<FP, method::Central>::value()));
}

} // namespace min

#endif // NUMERICAL_DIFFERENCE_HPP