/**
 * @file residual.hpp
 * @brief Residual error models describing the conditional distribution of
 *        observations given the model prediction.
 *
 * With eps ~ N(0, 1) i.i.d., the available models are:
 *
 *   Constant      : y = f + a*eps                 sd(y|f) = a
 *   Proportional  : y = f*(1 + b*eps)             sd(y|f) = b*|f|
 *   Exponential   : log y = log f + a*eps         sd(log y|log f) = a
 *   Combined      : y = f + (a + b*|f|)*eps       sd(y|f) = a + b*|f|
 *   Combined2     : y = f + sqrt(a^2 + b^2 f^2)*eps
 *                   sd(y|f) = sqrt(a^2 + b^2 f^2)
 *   Power         : y = f + (a + b*|f|^c)*eps
 *                   sd(y|f) = a + b*|f|^c
 *
 * Main functions, used to optimize the error parameters {a, b, c}:
 *   error(f, pres, kind)        -> residual standard deviation (model scale)
 *   errorObs(f, pres, kind)     -> standard deviation of y on the observation scale
 *   resid(y, f, kind)           -> residual (y-f, or log y - log f if exponential)
 *   stdResid(...)               -> standardized residual: resid / error
 *   ssq(y, f, pres, kind)       -> weighted sum of squares (LS objective)
 *   negloglik(y, f, pres, kind) -> Gaussian negative log-likelihood (ML objective)
 *
 * Header-only. Requires header-only Eigen3 and C++20.
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>

namespace resmod {

using Vec = Eigen::VectorXd;

/**
 * @brief Residual error model type.
 *
 * Combined2 uses g = sqrt(a^2 + b^2 f^2) (quadratic combination), while Power
 * uses g = a + b*|f|^c with a free exponent c.
 */
enum class ErrorKind { Constant, Proportional, Exponential, Combined, Combined2, Power };

/**
 * @brief Error-model parameters.
 *
 * @c a is the additive/exponential parameter, @c b is the proportional
 * parameter, and @c c is the exponent used only by the Power model.
 */
struct Pres {
    double a = 1.0;
    double b = 1.0;
    double c = 1.0;      // exponent (Power model only)
    Pres() = default;
    Pres(double a_, double b_) : a(a_), b(b_) {}
    Pres(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
};

/**
 * @brief Returns a stable English label for the error model type.
 * @param t Error model type.
 * @return Read-only label; "?" for an unrecognized value.
 */
inline const char* name(ErrorKind t) {
    switch (t) {
        case ErrorKind::Constant:     return "constant";
        case ErrorKind::Proportional: return "proportional";
        case ErrorKind::Exponential:  return "exponential";
        case ErrorKind::Combined:     return "combined";
        case ErrorKind::Combined2:    return "combined2";
        case ErrorKind::Power:        return "power";
    }
    return "?";
}

/**
 * @brief Number of error parameters used by a model type.
 * @param t Error model type.
 * @return 1 for single-parameter models, 2 for the combined models, and 3 for
 *         the Power model.
 */
inline int nParams(ErrorKind t) {
    switch (t) {
        case ErrorKind::Constant:    return 1;   // a
        case ErrorKind::Proportional: return 1;   // b
        case ErrorKind::Exponential:  return 1;   // a
        case ErrorKind::Combined:    return 2;   // a, b
        case ErrorKind::Combined2:   return 2;   // a, b
        case ErrorKind::Power:     return 3;   // a, b, c
    }
    return 0;
}

/**
 * @brief Residual standard deviation on the model scale.
 *
 * - Constant     -> a
 * - Proportional -> b*|f|
 * - Exponential  -> a (standard deviation of the residual on the log scale)
 * - Combined     -> a + b*|f|
 * - Combined2    -> sqrt(a^2 + b^2 f^2)
 * - Power        -> a + b*|f|^c
 *
 * @param f Model prediction.
 * @param pres Error parameters.
 * @param kind Error model type.
 * @return Residual standard deviation g(f).
 */
inline double error(double f, const Pres& pres, ErrorKind kind) {
    switch (kind) {
        case ErrorKind::Constant:    return pres.a;
        case ErrorKind::Proportional: return pres.b * std::abs(f);
        case ErrorKind::Exponential:  return pres.a;
        case ErrorKind::Combined:    return pres.a + pres.b * std::abs(f);
        case ErrorKind::Combined2:   return std::sqrt(pres.a * pres.a + pres.b * pres.b * f * f);
        case ErrorKind::Power:     return pres.a + pres.b * std::pow(std::abs(f), pres.c);
    }
    return pres.a;
}

/**
 * @brief Standard deviation of y on the observation scale.
 *
 * For the exponential model y = f*exp(a*eps):
 *   sd(y|f) = |f| * exp(a^2/2) * sqrt(exp(a^2) - 1)
 * (approximately a*|f| for small a). For all other models this is equal to
 * error().
 *
 * @param f Model prediction.
 * @param pres Error parameters.
 * @param kind Error model type.
 * @return Standard deviation of y given f.
 */
inline double errorObs(double f, const Pres& pres, ErrorKind kind) {
    if (kind == ErrorKind::Exponential) {
        const double a2 = pres.a * pres.a;
        return std::abs(f) * std::exp(0.5 * a2) * std::sqrt(std::expm1(a2));
    }
    return error(f, pres, kind);
}

/**
 * @brief Residual on the model scale.
 *
 * Returns y - f, or log y - log f for the exponential model.
 *
 * @param y Observed value.
 * @param f Model prediction.
 * @param kind Error model type.
 * @return The residual.
 * @throws std::domain_error If the exponential model is selected and y <= 0 or
 *         f <= 0.
 */
inline double resid(double y, double f, ErrorKind kind) {
    if (kind == ErrorKind::Exponential) {
        if (y <= 0.0 || f <= 0.0)
            throw std::domain_error("exponential model requires y > 0 and f > 0");
        return std::log(y) - std::log(f);
    }
    return y - f;
}

/**
 * @brief Standardized residual, assumed to follow N(0, 1).
 * @param y Observed value.
 * @param f Model prediction.
 * @param pres Error parameters.
 * @param kind Error model type.
 * @return resid(y, f, kind) / error(f, pres, kind).
 */
inline double stdResid(double y, double f, const Pres& pres, ErrorKind kind) {
    return resid(y, f, kind) / error(f, pres, kind);
}

/**
 * @brief Weighted sum of squares for Eigen vectors.
 *
 * Computes sum_i (r_i^2 / g_i^2), where r_i is the residual and g_i the error
 * scale (on the log scale for the exponential model).
 *
 * @warning Degeneracy in scale: ssq() is not identifiable in {a, b} for the
 *          proportional/combined models because it tends to 0 as the error
 *          scale tends to infinity. It is intended as a least-squares objective
 *          with the scale fixed, or as a diagnostic (at the ML estimate, ssq is
 *          approximately n). To estimate {a, b}, use negloglik(), whose
 *          log(g^2) term breaks the degeneracy.
 *
 * @param y Observations.
 * @param f Predictions.
 * @param pres Error parameters.
 * @param kind Error model type.
 * @return The weighted sum of squares.
 * @throws std::invalid_argument If y and f have different sizes.
 */
inline double ssq(const Vec& y, const Vec& f, const Pres& pres, ErrorKind kind) {
    if (y.size() != f.size()) throw std::invalid_argument("y and f must have the same size");
    double s = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double g = error(f[i], pres, kind);
        const double r = resid(y[i], f[i], kind);
        s += (r * r) / (g * g);
    }
    return s;
}

/**
 * @brief Weighted sum of squares for standard vectors.
 *
 * Same computation as the Eigen overload.
 *
 * @param y Observations.
 * @param f Predictions.
 * @param pres Error parameters.
 * @param kind Error model type.
 * @return The weighted sum of squares.
 * @throws std::invalid_argument If y and f have different sizes.
 */
inline double ssq(const std::vector<double>& y, const std::vector<double>& f,
                  const Pres& pres, ErrorKind kind) {
    if (y.size() != f.size()) throw std::invalid_argument("y and f must have the same size");
    double s = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double g = error(f[i], pres, kind);
        const double r = resid(y[i], f[i], kind);
        s += (r * r) / (g * g);
    }
    return s;
}

/**
 * @brief Gaussian negative log-likelihood (ML objective).
 *
 * Computes 0.5 * sum_i [ r_i^2/g_i^2 + log(g_i^2) ], plus the Jacobian term
 * for the exponential model. It is preferable to ssq() when estimating a and b
 * by maximum likelihood because it avoids the scale bias of least squares in
 * the proportional/combined models.
 *
 * @param y Observations.
 * @param f Predictions.
 * @param pres Error parameters.
 * @param kind Error model type.
 * @return The negative log-likelihood, or +infinity if any error scale is
 *         non-positive.
 * @throws std::invalid_argument If y and f have different sizes.
 */
inline double negloglik(const Vec& y, const Vec& f, const Pres& pres, ErrorKind kind) {
    if (y.size() != f.size()) throw std::invalid_argument("y and f must have the same size");
    double s = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double g = error(f[i], pres, kind);
        if (g <= 0.0) return std::numeric_limits<double>::infinity();
        const double r = resid(y[i], f[i], kind);
        s += 0.5 * ((r * r) / (g * g) + std::log(g * g));
        if (kind == ErrorKind::Exponential) s -= std::log(y[i]);   // Jacobian d(log y)/dy
    }
    return s;
}

/**
 * @brief Builds error parameters from convenience scalar inputs.
 *
 * For single-parameter models the value is stored in the field used by that
 * model; for the proportional model the user-supplied @p a is stored as the
 * proportional parameter @c b.
 *
 * @param a Additive parameter, or the single parameter for one-parameter models.
 * @param b Proportional parameter.
 * @param kind Error model type.
 * @return Parameters initialized for the requested model.
 */
inline Pres paramsFrom(double a, double b, ErrorKind kind) {
    // single-parameter models store the value in the field they use
    switch (kind) {
        case ErrorKind::Constant:    return Pres(a, 0.0);
        case ErrorKind::Proportional: return Pres(0.0, a);     // user 'a' becomes b
        case ErrorKind::Exponential:  return Pres(a, 0.0);
        case ErrorKind::Combined:    return Pres(a, b);
        case ErrorKind::Combined2:   return Pres(a, b);
        case ErrorKind::Power:     return Pres(a, b, 1.0);
    }
    return Pres(a, b);
}

} // namespace resmod
