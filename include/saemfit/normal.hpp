// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file normal.hpp
 * @brief Dependency-free utilities for the standard normal distribution.
 *
 * The header provides:
 *  - @ref normPdf, @ref normCdf, @ref normLogCdf (stable in the tails), and
 *    @ref normLogProbInterval;
 *  - @ref normQuantile (Acklam's approximation followed by one Halley step);
 *  - @ref sampleTruncStdNormal for truncated normal sampling (CDF inversion
 *    plus Robert's exponential rejection sampler for distant tails).
 */

#pragma once
#include <cmath>
#include <limits>
#include <numbers>
#include <random>

namespace mcmckl {

/**
 * @brief Standard normal probability density function.
 * @param z Evaluation point.
 * @return @f$\varphi(z)@f$.
 */
inline double normPdf(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::numbers::pi); }

/**
 * @brief Standard normal cumulative distribution function.
 * @param z Evaluation point.
 * @return @f$\Phi(z)@f$.
 */
inline double normCdf(double z) { return 0.5 * std::erfc(-z / std::numbers::sqrt2); }

/**
 * @brief Numerically stable @f$\log\Phi(z)@f$.
 *
 * Uses the complementary error function for moderate arguments, a
 * @c log1p formulation for positive @f$z@f$, and a tail asymptotic expansion
 * for @f$z < -20@f$.
 *
 * @param z Evaluation point.
 * @return @f$\log\Phi(z)@f$.
 */
inline double normLogCdf(double z) {
    if (z > 0.0) return std::log1p(-0.5 * std::erfc(z / std::numbers::sqrt2));
    if (z > -20.0) return std::log(0.5 * std::erfc(-z / std::numbers::sqrt2));
    const double z2 = z * z;
    return -0.5 * z2 - std::log(-z) - 0.5 * std::log(2.0 * std::numbers::pi) +
           std::log1p(-1.0 / z2 + 3.0 / (z2 * z2) - 15.0 / (z2 * z2 * z2));
}

/**
 * @brief Computes @f$\log P(\text{lo} < Z < \text{hi})@f$.
 *
 * The bounds may be infinite. The computation is performed in the left tail
 * for numerical accuracy.
 *
 * @param lo Lower bound.
 * @param hi Upper bound.
 * @return The log probability, or @c -infinity when @p hi <= @p lo.
 */
inline double normLogProbInterval(double lo, double hi) {
    const double inf = std::numeric_limits<double>::infinity();
    if (!(hi > lo)) return -inf;
    if (lo > 0.0) { const double t = -lo; lo = -hi; hi = t; }   // Symmetry: work in the left tail.
    const double lh = normLogCdf(hi);
    if (std::isinf(lo) && lo < 0.0) return lh;
    const double ll = normLogCdf(lo);
    return lh + std::log1p(-std::exp(ll - lh));
}

/**
 * @brief Standard normal quantile function.
 *
 * Uses Acklam's rational approximation refined by one Halley step, giving a
 * relative accuracy of about @f$10^{-15}@f$.
 *
 * @param p Probability.
 * @return @f$\Phi^{-1}(p)@f$; @c -infinity for @f$p \le 0@f$ and @c +infinity
 *         for @f$p \ge 1@f$.
 */
inline double normQuantile(double p) {
    const double inf = std::numeric_limits<double>::infinity();
    if (!(p > 0.0)) return -inf;
    if (!(p < 1.0)) return inf;
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    const double plow = 0.02425;
    double x;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - plow) {
        const double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log1p(-p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    // Halley refinement.
    const double e = normCdf(x) - p;
    const double u = e * std::sqrt(2.0 * std::numbers::pi) * std::exp(0.5 * x * x);
    return x - u / (1.0 + 0.5 * x * u);
}

/**
 * @brief Samples a standard normal truncated to the interval @f$(a, b)@f$.
 *
 * Uses CDF inversion in the central region and Robert's shifted-exponential
 * rejection sampler for distant tails.
 *
 * @tparam RNG Random number generator type.
 * @param a Lower truncation bound (may be @c -infinity).
 * @param b Upper truncation bound (may be @c +infinity).
 * @param rng Random number generator.
 * @return A draw from @f$N(0,1)@f$ conditioned on @f$a < Z < b@f$.
 * @pre @f$a < b@f$.
 */
template <class RNG>
inline double sampleTruncStdNormal(double a, double b, RNG& rng) {
    const double inf = std::numeric_limits<double>::infinity();
    std::uniform_real_distribution<double> U(0.0, 1.0);
    bool flip = false;
    if (a > 0.0) { const double t = -b; b = -a; a = t; flip = true; }   // Work where the CDF is accurate.
    double z;
    if (b < -6.0) {
        // Distant tail (reflected interval [lo, hi] with lo > 6): shifted
        // exponential proposal with rejection (Robert, 1995).
        const double lo = -b, hi = -a;
        const double alpha = 0.5 * (lo + std::sqrt(lo * lo + 4.0));
        double zz = lo;
        for (int it = 0; it < 100000; ++it) {
            zz = lo - std::log(1.0 - U(rng)) / alpha;
            if (zz <= hi && U(rng) < std::exp(-0.5 * (zz - alpha) * (zz - alpha))) break;
        }
        z = -std::min(zz, std::isinf(hi) ? zz : hi);
    } else {
        const double pa = std::isinf(a) ? 0.0 : normCdf(a), pb = normCdf(b);
        double u = pa + (pb - pa) * U(rng);
        u = std::min(std::max(u, std::nextafter(pa, 1.0)), std::nextafter(pb, 0.0));
        z = normQuantile(u);
        if (!(z > a)) z = std::isinf(a) ? b - 1e-9 : a + 1e-12;
        if (!(z < b)) z = b - 1e-12;
    }
    (void)inf;
    return flip ? -z : z;
}

} // namespace mcmckl
