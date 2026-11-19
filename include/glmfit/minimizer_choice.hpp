// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file minimizer_choice.hpp
 * @brief Definition of optimization minimizer choices for GLM solver backends.
 * @details Provides the enumeration class for selecting numerical optimization
 *          algorithms used by MleSolver and QuasiSolver implementations.
 */

#ifndef _GLMFIT_MINIMIZER_CHOICE_HPP
#define _GLMFIT_MINIMIZER_CHOICE_HPP

namespace glm {

/**
 * @brief Selection enumeration for numerical optimization algorithms.
 * @details Defines the optimization backend used by `run_optimization()` in both
 *          `MleSolver` and `QuasiSolver`. Shared within the `glm` namespace to prevent
 *          One Definition Rule (ODR) violations across translation units.
 */
enum class Minimizer {
    /**
     * @brief Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Box constraints (L-BFGS-B).
     * @details Default solver (`min::lbfgsb`). Utilizes analytical gradients and low-memory
     *          approximations of the Inverse Hessian matrix. Robust for high-dimensional parameter
     *          spaces and bounded optimization problems.
     */
    LBFGSB,

    /**
     * @brief Standard Broyden-Fletcher-Goldfarb-Shanno algorithm.
     * @details Unconstrained optimization (`min::bfgs`) approximating the full dense Hessian matrix.
     *          May converge in fewer iterations than L-BFGS-B for small-to-medium parameter sizes (low $p$),
     *          at the cost of $\mathcal{O}(p^2)$ memory storage per iteration.
     */
    BFGS,

    /**
     * @brief Derivative-free Nelder-Mead Simplex optimization algorithm.
     * @details Derivative-free optimization backend (`min::nelder_mead`). Useful as a fallback when
     *          analytical gradients are unavailable, ill-conditioned, or severe quasi-separability
     *          creates non-smooth loss surfaces.
     */
    NelderMead
};

} // namespace glm

#endif // _GLMFIT_MINIMIZER_CHOICE_HPP
