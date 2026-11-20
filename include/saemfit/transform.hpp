// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file transform.hpp
 * @brief Parameter transformations used to enforce domain constraints.
 *
 * The library optimizes unconstrained parameters @f$\phi@f$ and maps them to
 * constrained natural-scale parameters @f$\psi@f$:
 *
 * | Transform | Forward map @f$\phi = g(\psi)@f$ | Domain of @f$\psi@f$ |
 * |-----------|--------------------------------|---------------------|
 * | Identity  | @f$\phi = \psi@f$              | @f$\mathbb{R}@f$    |
 * | Log       | @f$\phi = \log\psi@f$          | @f$\psi > 0@f$      |
 * | Logit     | @f$\phi = \log(\psi/(1-\psi))@f$ | @f$0 < \psi < 1@f$ |
 * | Probit    | @f$\phi = \Phi^{-1}(\psi)@f$   | @f$0 < \psi < 1@f$  |
 *
 * The four elementary maps are:
 *  - transphi(psi) -> phi  (natural scale to unconstrained scale);
 *  - transpsi(phi) -> psi  (unconstrained scale to natural scale);
 *  - derivphi(psi) -> dphi/dpsi  (Jacobian, used by the FIM);
 *  - derivpsi(phi) -> dpsi/dphi  (inverse Jacobian, used by the delta method).
 *
 * Chain rule for the score and the Fisher information (theta on the phi scale):
 * @f[
 *   \frac{\partial \ell}{\partial \psi}
 *     = \frac{d\phi}{d\psi}\,\frac{\partial \ell}{\partial \phi},
 *   \qquad
 *   I_\psi = J^\top I_\phi J,\quad J = \operatorname{diag}(d\phi/d\psi).
 * @f]
 *
 * Requires: Eigen3 (header only), C++20.
 */

#pragma once
#include "numcast.hpp"

#include "normal.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Supported parameter transformations.
 *
 * @c Probit maps @f$\phi = \Phi^{-1}(\psi)@f$ with @f$\psi \in (0,1)@f$.
 */
enum class Transform { Identity, Log, Logit, Probit };

/**
 * @brief Returns a stable English label for a transformation.
 * @param t Transformation type.
 * @return Read-only label; @c "?" for an unrecognized value.
 */
inline const char* name(Transform t) {
    switch (t) {
        case Transform::Identity: return "identity";
        case Transform::Log:      return "log";
        case Transform::Logit:    return "logit";
        case Transform::Probit:   return "probit";
    }
    return "?";
}

/**
 * @brief Maps a natural-scale parameter to the unconstrained scale.
 * @param psi Natural-scale value.
 * @param t Transformation type.
 * @return @f$\phi = g(\psi)@f$.
 * @throws std::domain_error If @p psi lies outside the transformation domain.
 */
inline double transphi(double psi, Transform t) {
    switch (t) {
        case Transform::Identity: return psi;
        case Transform::Log:
            if (psi <= 0.0) throw std::domain_error("log: psi must be > 0");
            return std::log(psi);
        case Transform::Logit:
            if (psi <= 0.0 || psi >= 1.0) throw std::domain_error("logit: psi must be in (0, 1)");
            return std::log(psi / (1.0 - psi));
        case Transform::Probit:
            if (psi <= 0.0 || psi >= 1.0) throw std::domain_error("probit: psi must be in (0, 1)");
            return normQuantile(psi);
    }
    return psi;
}

/**
 * @brief Maps an unconstrained parameter back to the natural scale.
 * @param phi Unconstrained value.
 * @param t Transformation type.
 * @return @f$\psi = g^{-1}(\phi)@f$.
 */
inline double transpsi(double phi, Transform t) {
    switch (t) {
        case Transform::Identity: return phi;
        case Transform::Log:      return std::exp(phi);
        case Transform::Logit:    return 1.0 / (1.0 + std::exp(-phi));   // sigmoid
        case Transform::Probit:   return normCdf(phi);
    }
    return phi;
}

/**
 * @brief Evaluates the forward Jacobian @f$d\phi/d\psi@f$.
 *
 * Used to transform the Fisher information from the @f$\phi@f$ scale to the
 * @f$\psi@f$ scale. For the supported maps:
 *  - Log   -> @f$1/\psi@f$;
 *  - Logit -> @f$1/(\psi(1-\psi))@f$;
 *  - Probit -> @f$1/\varphi(\Phi^{-1}(\psi))@f$.
 *
 * @param psi Natural-scale value.
 * @param t Transformation type.
 * @return @f$d\phi/d\psi@f$.
 * @throws std::domain_error If @p psi lies outside the transformation domain.
 */
inline double derivphi(double psi, Transform t) {
    switch (t) {
        case Transform::Identity: return 1.0;
        case Transform::Log:
            if (psi <= 0.0) throw std::domain_error("log: psi must be > 0");
            return 1.0 / psi;
        case Transform::Logit:
            if (psi <= 0.0 || psi >= 1.0) throw std::domain_error("logit: psi must be in (0, 1)");
            return 1.0 / (psi * (1.0 - psi));
        case Transform::Probit:
            if (psi <= 0.0 || psi >= 1.0) throw std::domain_error("probit: psi must be in (0, 1)");
            return 1.0 / normPdf(normQuantile(psi));
    }
    return 1.0;
}

/**
 * @brief Evaluates the inverse Jacobian @f$d\psi/d\phi@f$.
 *
 * Used by the delta method for standard errors. For the supported maps:
 *  - Log   -> @f$\exp(\phi) = \psi@f$;
 *  - Logit -> @f$\psi(1-\psi)@f$;
 *  - Probit -> @f$\varphi(\phi)@f$.
 *
 * @param phi Unconstrained value.
 * @param t Transformation type.
 * @return @f$d\psi/d\phi@f$.
 */
inline double derivpsi(double phi, Transform t) {
    switch (t) {
        case Transform::Identity: return 1.0;
        case Transform::Log:      return std::exp(phi);
        case Transform::Logit: {
            const double s = 1.0 / (1.0 + std::exp(-phi));
            return s * (1.0 - s);
        }
        case Transform::Probit:   return normPdf(phi);
    }
    return 1.0;
}

/**
 * @brief Vectorized forward transformation using one type for every coordinate.
 * @param psi Natural-scale vector.
 * @param t Transformation type.
 * @return Transformed vector.
 * @throws std::domain_error If any coordinate lies outside the domain.
 */
inline Vec transphi(const Vec& psi, Transform t) {
    Vec out(psi.size());
    for (Eigen::Index i = 0; i < psi.size(); ++i) out[i] = transphi(psi[i], t);
    return out;
}

/**
 * @brief Vectorized inverse transformation using one type for every coordinate.
 * @param phi Unconstrained vector.
 * @param t Transformation type.
 * @return Natural-scale vector.
 */
inline Vec transpsi(const Vec& phi, Transform t) {
    Vec out(phi.size());
    for (Eigen::Index i = 0; i < phi.size(); ++i) out[i] = transpsi(phi[i], t);
    return out;
}

/**
 * @brief Vectorized forward Jacobian using one type for every coordinate.
 * @param psi Natural-scale vector.
 * @param t Transformation type.
 * @return Vector of @f$d\phi/d\psi@f$ values.
 * @throws std::domain_error If any coordinate lies outside the domain.
 */
inline Vec derivphi(const Vec& psi, Transform t) {
    Vec out(psi.size());
    for (Eigen::Index i = 0; i < psi.size(); ++i) out[i] = derivphi(psi[i], t);
    return out;
}

/**
 * @brief Vectorized inverse Jacobian using one type for every coordinate.
 * @param phi Unconstrained vector.
 * @param t Transformation type.
 * @return Vector of @f$d\psi/d\phi@f$ values.
 */
inline Vec derivpsi(const Vec& phi, Transform t) {
    Vec out(phi.size());
    for (Eigen::Index i = 0; i < phi.size(); ++i) out[i] = derivpsi(phi[i], t);
    return out;
}

/**
 * @brief Vectorized forward transformation with one type per coordinate.
 *
 * Useful when @f$\theta@f$ mixes log, logit, and other parameterizations.
 *
 * @param psi Natural-scale vector.
 * @param ts Transformation type for each coordinate.
 * @return Transformed vector.
 * @throws std::invalid_argument If @p ts and @p psi have different sizes.
 * @throws std::domain_error If any coordinate lies outside its domain.
 */
inline Vec transphi(const Vec& psi, const std::vector<Transform>& ts) {
    if (toi(ts.size()) != psi.size()) throw std::invalid_argument("transform types have incorrect dimension");
    Vec out(psi.size());
    for (Eigen::Index i = 0; i < psi.size(); ++i) out[i] = transphi(psi[i], ts[uz(i)]);
    return out;
}

/**
 * @brief Vectorized inverse transformation with one type per coordinate.
 * @param phi Unconstrained vector.
 * @param ts Transformation type for each coordinate.
 * @return Natural-scale vector.
 * @throws std::invalid_argument If @p ts and @p phi have different sizes.
 */
inline Vec transpsi(const Vec& phi, const std::vector<Transform>& ts) {
    if (toi(ts.size()) != phi.size()) throw std::invalid_argument("transform types have incorrect dimension");
    Vec out(phi.size());
    for (Eigen::Index i = 0; i < phi.size(); ++i) out[i] = transpsi(phi[i], ts[uz(i)]);
    return out;
}

/**
 * @brief Vectorized forward Jacobian with one type per coordinate.
 * @param psi Natural-scale vector.
 * @param ts Transformation type for each coordinate.
 * @return Vector of @f$d\phi/d\psi@f$ values.
 * @throws std::invalid_argument If @p ts and @p psi have different sizes.
 * @throws std::domain_error If any coordinate lies outside its domain.
 */
inline Vec derivphi(const Vec& psi, const std::vector<Transform>& ts) {
    if (toi(ts.size()) != psi.size()) throw std::invalid_argument("transform types have incorrect dimension");
    Vec out(psi.size());
    for (Eigen::Index i = 0; i < psi.size(); ++i) out[i] = derivphi(psi[i], ts[uz(i)]);
    return out;
}

/**
 * @brief Vectorized inverse Jacobian with one type per coordinate.
 * @param phi Unconstrained vector.
 * @param ts Transformation type for each coordinate.
 * @return Vector of @f$d\psi/d\phi@f$ values.
 * @throws std::invalid_argument If @p ts and @p phi have different sizes.
 */
inline Vec derivpsi(const Vec& phi, const std::vector<Transform>& ts) {
    if (toi(ts.size()) != phi.size()) throw std::invalid_argument("transform types have incorrect dimension");
    Vec out(phi.size());
    for (Eigen::Index i = 0; i < phi.size(); ++i) out[i] = derivpsi(phi[i], ts[uz(i)]);
    return out;
}

/**
 * @brief Changes the base of a Fisher information matrix.
 *
 * Computes @f$I_\psi = J^\top I_\phi J@f$ with
 * @f$J = \operatorname{diag}(d\phi/d\psi)@f$.
 *
 * @param fim_phi Fisher information on the unconstrained scale.
 * @param psi Natural-scale parameter vector.
 * @param ts Transformation type for each coordinate.
 * @return Fisher information on the natural scale.
 * @throws std::invalid_argument If @p ts and @p psi have different sizes.
 * @throws std::domain_error If any coordinate lies outside its domain.
 */
inline Mat fimTransform(const Mat& fim_phi, const Vec& psi, const std::vector<Transform>& ts) {
    const Vec j = derivphi(psi, ts);
    return j.asDiagonal() * fim_phi * j.asDiagonal();
}

} // namespace mcmckl
