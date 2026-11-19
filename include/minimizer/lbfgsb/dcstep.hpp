// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_DCSTEP_HPP
#define LBFGSB_DCSTEP_HPP

/**
 * @file dcstep.hpp
 * @ingroup lbfgsb_linesearch
 * @brief One safeguarded interval-update step of the More–Thuente line search.
 *
 * @details
 * Given the bracketing endpoints \f$(\mathrm{stx},\mathrm{sty})\f$, the
 * current trial step @p stp and the function/derivative triplets at each, this
 * routine computes a new safeguarded trial step and tightens the bracket that
 * is guaranteed to contain a point satisfying the (strong) Wolfe conditions.
 * The update logic uses the standard cubic (`stpc`) and quadratic (`stpq`)
 * interpolations and forces the new step to stay inside the bracket once a
 * minimizer has been bracketed.
 *
 * This is the low-level worker called by @ref dcsrch on every line-search
 * iteration.
 *
 * @see dcsrch.hpp (caller)
 */


#include <algorithm>
#include <cmath>
#include "../common.hpp"

namespace min {

template<typename FP = double>
/**
 * @brief Compute a safeguarded trial step and update the bracketing interval.
 *
 * @details
 * The routine classifies the local picture using the sign of the derivative
 * product \f$\mathrm{dp}\cdot\mathrm{dx}\f$ and the magnitudes of the
 * derivatives (four canonical cases, matching MINPACK-2 `dcstep`):
 * - \f$\mathrm{fp} > \mathrm{fx}\f$: the trial point is worse; the next step
 *   lies between the cubic and quadratic minimizers, and a minimizer is
 *   declared bracketed;
 * - \f$\mathrm{dp}\,\mathrm{dx} < 0\f$: opposite derivative signs bracket the
 *   minimum; the new step is the cubic/quartic solution, bracket closes;
 * - \f$|\mathrm{dp}| < |\mathrm{dx}|\f$: same signs, smaller derivative —
 *   interpolation with bound safeguards, bracket kept or tightened;
 * - otherwise: derivative not decreasing; step is thrown to the nearest bound
 *   (`stpmin`/`stpmax`) unless already bracketed, in which case a one-sided
 *   cubic step toward `sty` is used.
 *
 * The bracket endpoints are then updated (`stx` becomes the best point; `sty`
 * the other endpoint, tracking the interval that contains the minimizer).
 *
 * @tparam FP     Floating-point scalar type.
 * @param stx     In/out: step with the least function value; updated best step.
 * @param fx      Function value at @p stx (in/out).
 * @param dx      Derivative at @p stx (in/out; negative in the step direction).
 * @param sty     In/out: second endpoint of the bracketing interval.
 * @param fy      Function value at @p sty (in/out).
 * @param dy      Derivative at @p sty (in/out).
 * @param stp     In/out: current trial step; the new safeguarded step on exit.
 * @param fp      Function value at @p stp (in; not modified).
 * @param dp      Derivative at @p stp (in; not modified).
 * @param brackt  In: false initially. Out: true once a minimizer has been
 *                bracketed in \f$[\min(\mathrm{stx},\mathrm{sty}),
 *                \max(\mathrm{stx},\mathrm{sty})]\f$.
 * @param stpmin  Lower bound for the step (not modified).
 * @param stpmax  Upper bound for the step (not modified).
 */
inline void dcstep(FP& stx, FP& fx, FP& dx,
                   FP& sty, FP& fy, FP& dy,
                   FP& stp, FP  fp, FP  dp,
                   bool& brackt,
                   FP  stpmin, FP  stpmax)
{
    constexpr FP zero = FP(0.0), p66 = FP(0.66), two = FP(2.0), three = FP(3.0);
    const FP sgnd = dp * (dx / std::abs(dx));
    FP stpf, theta, s, gamma, p, q, r, stpc, stpq;

    if (fp > fx) {
        theta = three * (fx - fp) / (stp - stx) + dx + dp;
        s     = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
        gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp < stx) gamma = -gamma;
        p = (gamma - dx) + theta;
        q = ((gamma - dx) + gamma) + dp;
        r = p / q;
        stpc = stx + r * (stp - stx);
        stpq = stx + (dx / ((fx - fp) / (stp - stx) + dx) / two) * (stp - stx);
        stpf = (std::abs(stpc - stx) < std::abs(stpq - stx)) ? stpc : stpc + (stpq - stpc) / two;
        brackt = true;
    } else if (sgnd < zero) {
        theta = three * (fx - fp) / (stp - stx) + dx + dp;
        s     = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
        gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp > stx) gamma = -gamma;
        p = (gamma - dp) + theta;
        q = ((gamma - dp) + gamma) + dx;
        r = p / q;
        stpc = stp + r * (stx - stp);
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        stpf = (std::abs(stpc - stp) > std::abs(stpq - stp)) ? stpc : stpq;
        brackt = true;
    } else if (std::abs(dp) < std::abs(dx)) {
        theta = three * (fx - fp) / (stp - stx) + dx + dp;
        s     = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
        gamma = s * std::sqrt(std::max(zero, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
        if (stp > stx) gamma = -gamma;
        p = (gamma - dp) + theta;
        q = (gamma + (dx - dp)) + gamma;
        r = p / q;
        if      (r < zero && gamma != zero) stpc = stp + r * (stx - stp);
        else if (stp > stx)                 stpc = stpmax;
        else                                stpc = stpmin;
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (brackt) {
            stpf = (std::abs(stpc - stp) < std::abs(stpq - stp)) ? stpc : stpq;
            stpf = (stp > stx) ? std::min(stp + p66 * (sty - stp), stpf)
                               : std::max(stp + p66 * (sty - stp), stpf);
        } else {
            stpf = (std::abs(stpc - stp) > std::abs(stpq - stp)) ? stpc : stpq;
            stpf = std::min(stpmax, stpf);
            stpf = std::max(stpmin, stpf);
        }
    } else {
        if (brackt) {
            theta = three * (fp - fy) / (sty - stp) + dy + dp;
            s     = std::max({std::abs(theta), std::abs(dy), std::abs(dp)});
            gamma = s * std::sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
            if (stp > sty) gamma = -gamma;
            p = (gamma - dp) + theta;
            q = ((gamma - dp) + gamma) + dy;
            r = p / q;
            stpc = stp + r * (sty - stp);
            stpf = stpc;
        } else {
            stpf = (stp > stx) ? stpmax : stpmin;
        }
    }

    if (fp > fx) {
        sty = stp; fy = fp; dy = dp;
    } else {
        if (sgnd < zero) { sty = stx; fy = fx; dy = dx; }
        stx = stp; fx = fp; dx = dp;
    }
    stp = stpf;
}

} // namespace min

#endif // LBFGSB_DCSTEP_HPP
