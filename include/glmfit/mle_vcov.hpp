// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file mle_vcov.hpp
 * @brief Asymptotic covariance matrix inversion and standard error estimation routines.
 * @details Implements robust numerical strategies for inverting the negative log-likelihood
 *          Hessian matrix, computing parameter standard errors, evaluating Hessian condition numbers,
 *          and handling ill-conditioned or singular information matrices.
 */

#ifndef _GLMFIT_MLE_VCOV_HPP
#define _GLMFIT_MLE_VCOV_HPP

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <iostream>

namespace glm {

/**
 * @brief Inverts the Hessian matrix of the negative log-likelihood to compute asymptotic variance-covariance.
 * @details Employs a multi-tier fallback scheme to handle ill-conditioned matrices:
 *          1. Direct Cholesky decomposition (`Eigen::LLT`) for positive-definite symmetric matrices.
 *          2. Robust $LDLT$ decomposition (`Eigen::LDLT`) for semi-definite matrices.
 *          3. Adaptive Tikhonov (Ridge) regularization to stabilize flat directions under quasi-separability.
 *          4. Truncated Singular Value Decomposition (`Eigen::JacobiSVD`) Moore-Penrose pseudo-inverse fallback.
 * @param H Symmetric Hessian matrix ($p \times p$) of negative log-likelihood.
 * @return Estimated asymptotic variance-covariance matrix ($p \times p$).
 */
inline Eigen::MatrixXd invert_hessian(const Eigen::MatrixXd& H) {
    const Eigen::Index m = H.rows();
    if (m == 0) return Eigen::MatrixXd::Zero(0, 0);

    // 1. Direct Cholesky decomposition (LLT)
    Eigen::LLT<Eigen::MatrixXd> llt(H.selfadjointView<Eigen::Lower>());
    if (llt.info() == Eigen::Success) {
        return llt.solve(Eigen::MatrixXd::Identity(m, m));
    }

    // 2. Fallback to LDLT decomposition
    Eigen::LDLT<Eigen::MatrixXd> ldlt(H.selfadjointView<Eigen::Lower>());
    if (ldlt.info() == Eigen::Success) {
        return ldlt.solve(Eigen::MatrixXd::Identity(m, m));
    }

    // 3. Fallback to Tikhonov (Ridge) adaptive regularization
    Eigen::MatrixXd H_stabilized = H;
    // Scale lambda to the actual magnitude of H's diagonal: a fixed constant
    // (e.g. 1e-5) is negligible against Hessians with entries of order 1e3-1e6
    // (typical of uncentered/unscaled covariates), making this level almost
    // indistinguishable from the plain LDLT of level 2.
    const double diag_scale = H.diagonal().cwiseAbs().mean();
    const double ridge_lambda = 1e-6 * std::max(diag_scale, 1.0);
    H_stabilized.diagonal().array() += ridge_lambda;

    Eigen::LDLT<Eigen::MatrixXd> ldlt_stab(H_stabilized.selfadjointView<Eigen::Lower>());
    if (ldlt_stab.info() == Eigen::Success) {
        return ldlt_stab.solve(Eigen::MatrixXd::Identity(m, m));
    }

    // 4. Absolute emergency fallback: SVD pseudo-inverse with truncated singular values
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // setThreshold() expects a RELATIVE factor: Eigen internally discards singular
    // values with sigma_i <= threshold() * sigma_max. Passing a value already
    // multiplied by sigma_max (as before) makes Eigen multiply it again, leaving an
    // effective threshold of order sigma_max^2 -- with badly scaled Hessians
    // (large sigma_max) this truncates far more directions than intended, precisely
    // in the case where this fallback needs to be reliable.
    svd.setThreshold(1e-9 * static_cast<double>(std::max(H.rows(), H.cols())));
    return svd.solve(Eigen::MatrixXd::Identity(m, m));
}

/**
 * @brief Extracts parameter standard errors as square roots of diagonal covariance terms.
 * @details Assigns NaN if diagonal variance terms are non-positive or non-finite, flagging
 *          "could not be estimated" rather than a fabricated numeric standard error. A large
 *          sentinel like 1e6 would silently make coef/se collapse toward 0 and its p-value
 *          toward 1, reading downstream as "clearly not significant" -- the opposite of the
 *          correct interpretation ("significance unknown").
 * @param vcov Asymptotic variance-covariance matrix ($p \times p$).
 * @return Eigen vector containing standard errors for each parameter (NaN where undefined).
 */
inline Eigen::VectorXd vcov_to_se(const Eigen::MatrixXd& vcov) {
    const Eigen::Index m = vcov.rows();
    Eigen::VectorXd se(m);
    for (Eigen::Index i = 0; i < m; ++i) {
        double var_i = vcov(i, i);

        if (var_i > 0.0 && std::isfinite(var_i)) {
            se(i) = std::sqrt(var_i);
        } else {
            se(i) = std::numeric_limits<double>::quiet_NaN();
        }
    }
    return se;
}

/**
 * @brief Memory-efficient standard error computation directly from Hessian matrix.
 * @details Solves for diagonal inverse entries column-by-column without retaining full $p \times p$
 *          covariance matrix in memory where applicable.
 * @param H Symmetric Hessian matrix ($p \times p$).
 * @return Eigen vector of standard error estimates.
 */
inline Eigen::VectorXd compute_all_se_efficient(const Eigen::MatrixXd& H) {
    const Eigen::Index m = H.rows();
    if (m == 0) return Eigen::VectorXd::Zero(0);

    Eigen::VectorXd se(m);

    // Column-by-column solve via LLT
    Eigen::LLT<Eigen::MatrixXd> llt(H.selfadjointView<Eigen::Lower>());
    if (llt.info() == Eigen::Success) {
        for (Eigen::Index i = 0; i < m; ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Unit(m, i);
            double var_i = llt.solve(e_i)(i);
            se(i) = (var_i > 0.0) ? std::sqrt(var_i) : std::numeric_limits<double>::quiet_NaN();
        }
        return se;
    }

    // Column-by-column solve via LDLT
    Eigen::LDLT<Eigen::MatrixXd> ldlt(H.selfadjointView<Eigen::Lower>());
    if (ldlt.info() == Eigen::Success) {
        for (Eigen::Index i = 0; i < m; ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Unit(m, i);
            double var_i = ldlt.solve(e_i)(i);
            se(i) = (var_i > 0.0) ? std::sqrt(var_i) : std::numeric_limits<double>::quiet_NaN();
        }
        return se;
    }

    // Robust fallback using invert_hessian
    Eigen::MatrixXd vcov = invert_hessian(H);
    return vcov_to_se(vcov);
}

/**
 * @brief Computes the 2-norm matrix condition number of the Hessian matrix.
 * @details Ratio of maximum eigenvalue to minimum eigenvalue ($\lambda_{\max} / \lambda_{\min}$).
 * @param H Symmetric Hessian matrix ($p \times p$).
 * @return Condition number as double, or infinity if matrix is singular or non-finite.
 */
inline double hessian_condition_number(const Eigen::MatrixXd& H) {
    const Eigen::Index m = H.rows();
    if (m == 0) return 1.0;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H.selfadjointView<Eigen::Lower>());
    if (es.info() != Eigen::Success) return std::numeric_limits<double>::infinity();

    const Eigen::VectorXd eigs = es.eigenvalues().cwiseAbs();
    const double max_eig = eigs.maxCoeff();
    const double min_eig = eigs.minCoeff();
    if (min_eig <= 0.0) return std::numeric_limits<double>::infinity();
    return max_eig / min_eig;
}

/**
 * @brief Attaches variance-covariance matrix and standard errors to a solver result object.
 * @tparam ResultT Result structure type containing `vcov` and `se` members.
 * @param res Result object modified in-place.
 * @param H Hessian matrix of negative log-likelihood.
 * @param warn_cond Condition number threshold above which warnings are logged (default = $10^8$).
 */
template <typename ResultT>
inline void attach_vcov_se(ResultT& res, const Eigen::MatrixXd& H, double warn_cond = 1e8) {
    Eigen::MatrixXd vcov = invert_hessian(H);
    Eigen::VectorXd se = vcov_to_se(vcov);

    const double cond = hessian_condition_number(H);
    if (!std::isfinite(cond) || cond > warn_cond) {
        if (!std::isfinite(cond)) {
            std::cerr << "[glm::attach_vcov_se] warning: Hessian is singular "
                      << "(non-finite condition number). Standard errors are derived "
                      << "from regularized fallback (Tikhonov or pseudo-inverse) and "
                      << "do not reflect true asymptotic variance. This typically indicates "
                      << "rank deficiency or severe quasi-separability.\n";
        } else {
            std::cerr << "[glm::attach_vcov_se] warning: Hessian condition number = "
                      << cond << " (> " << warn_cond << "). Standard errors may be "
                      << "unreliable if optimization did not converge with high precision.\n";
        }
    }

    res.vcov = std::move(vcov);
    res.se = std::move(se);
}

} // namespace glm

#endif // _GLMFIT_MLE_VCOV_HPP
