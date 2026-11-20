// ============================================================================
// @file expr.hpp
// @brief Evaluator for elementwise expression graphs ("compiled" from Python).
//
// Python traces the user's structural function (see saem/_trace.py) and turns
// it into a directed graph of nodes; this header evaluates that graph per
// observation without the GIL. The evaluator is thread-safe and performs no
// per-call allocations except for the initial growth of the per-thread
// workspace.
//
//   * Nodes that do not depend on any x column (psi, cov, constants) are
//     evaluated once per call (hoisting); all other nodes are evaluated once
//     per observation.
//   * Nodes are stored in topological order: every argument of a node has a
//     lower index than the node itself.
//   * Operations replicate NumPy semantics (NaN propagation, division by zero
//     yielding inf, maximum/minimum propagating NaN, np.sign, np.remainder
//     taking the sign of the divisor, booleans represented as 0.0/1.0).
//
// The order of ExprOp MUST match the OPS table in saem/_trace.py (verified via
// EXPR_NOPS).
// ============================================================================
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ode.hpp"

namespace mcmckl {

/**
 * @brief Opcodes for the elementwise expression evaluator.
 *
 * @warning The numeric order of these values is part of the wire format shared
 *          with saem/_trace.py. Do not reorder, insert, or remove entries
 *          without updating both sides; EXPR_NOPS verifies the table length.
 */
enum ExprOp : int {
    E_CONST = 0, E_PSI, E_X, E_COV,
    E_ADD, E_SUB, E_MUL, E_DIV, E_POW, E_NEG,
    E_EXP, E_LOG, E_LOG10, E_LOG2, E_LOG1P, E_EXPM1, E_EXP2,
    E_SQRT, E_CBRT, E_ABS, E_SIGN,
    E_SIN, E_COS, E_TAN, E_ASIN, E_ACOS, E_ATAN,
    E_SINH, E_COSH, E_TANH, E_ASINH, E_ACOSH, E_ATANH,
    E_FLOOR, E_CEIL, E_SQUARE, E_RECIP,
    E_MAX, E_MIN, E_ATAN2, E_MOD,
    E_LT, E_LE, E_GT, E_GE, E_EQ, E_NE,
    E_AND, E_OR, E_NOT,
    E_WHERE,
    E_ERF, E_ERFC, E_EXPIT, E_GAMMALN,
    // --- per-row / per-dose inputs and reduction over doses ---
    E_YTYPE,                                   ///< Response type (last column of x).
    E_Y,                                       ///< Observation y (custom likelihood).
    E_EXTRA,                                   ///< extra[a] (custom likelihood).
    E_DT, E_DAMT, E_DDUR, E_DCMT,              ///< Per dose: t - dose_time, amount, duration, cmt (1-based).
    E_SUMDOSE,                                 ///< a = core (per dose), b = time: sum over doses with t >= dose_time.
    E_COUNT                                    ///< Number of supported operations.
};

/**
 * @brief Immutable, topologically ordered expression graph compiled from Python.
 *
 * The program stores one opcode and up to three argument indices per node,
 * together with a constant operand and one or more output node indices. During
 * construction it validates the graph, classifies nodes as scalar or
 * vector-valued, and identifies per-dose ("level 2") subgraphs used by
 * E_SUMDOSE.
 *
 * Evaluation is thread-safe: each thread owns its scratch buffer, so the same
 * program may be evaluated concurrently. No external allocations occur after
 * the first evaluation on a given thread.
 */
class ExprProgram {
public:
    /**
     * @brief Builds and validates a multi-output expression program.
     * @param op Opcode per node (see ExprOp).
     * @param a First argument index per node.
     * @param b Second argument index per node.
     * @param c Third argument index per node.
     * @param cst Constant value per node (used by E_CONST).
     * @param outs Indices of the output nodes.
     * @throws std::invalid_argument If the arrays have different lengths,
     *         are empty, reference an unknown opcode, contain a forward or
     *         out-of-range argument, use a per-dose value outside
     *         E_SUMDOSE, or have an invalid output index.
     */
    ExprProgram(std::vector<int> op, std::vector<int> a, std::vector<int> b, std::vector<int> c,
                std::vector<double> cst, std::vector<int> outs)
        : op_(std::move(op)), a_(std::move(a)), b_(std::move(b)), c_(std::move(c)), cst_(std::move(cst)), outs_(std::move(outs)) {
        const int N = static_cast<int>(op_.size());
        if (N == 0 || a_.size() != op_.size() || b_.size() != op_.size() || c_.size() != op_.size() || cst_.size() != op_.size())
            throw std::invalid_argument("expr: arrays have different lengths or are empty");
        if (outs_.empty()) throw std::invalid_argument("expr: no output nodes");
        for (int o : outs_) if (o < 0 || o >= N) throw std::invalid_argument("expr: output node out of range");
        isVec_.assign(N, 0);
        lvl2_.assign(N, 0);
        subOf_.assign(N, -1);
        for (int k = 0; k < N; ++k) {
            const int o = op_[k];
            if (o < 0 || o >= E_COUNT) throw std::invalid_argument("expr: unknown operation " + std::to_string(o));
            const int na = nargs(o);
            const int args[3] = {a_[k], b_[k], c_[k]};
            bool vec = (o == E_X || o == E_YTYPE || o == E_Y || o == E_EXTRA || o == E_SUMDOSE);
            bool l2 = (o == E_DT || o == E_DAMT || o == E_DDUR || o == E_DCMT);
            for (int t = 0; t < na; ++t) {
                if (args[t] < 0 || args[t] >= k) throw std::invalid_argument("expr: argument is out of topological order");
                if (o == E_SUMDOSE && t == 0) continue;                 // the per-dose core is level 2
                if (lvl2_[args[t]]) l2 = true;
                if (isVec_[args[t]]) vec = true;
            }
            if (o == E_SUMDOSE) {
                if (lvl2_[b_[k]]) throw std::invalid_argument("expr: the time argument of doses.superpose cannot depend on the dose");
                l2 = false; vec = true;
            } else if (!l2) {
                for (int t = 0; t < na; ++t)
                    if (lvl2_[args[t]]) throw std::invalid_argument("expr: per-dose value used outside doses.superpose");
            }
            if (o == E_PSI)   { if (a_[k] < 0) throw std::invalid_argument("expr: psi index < 0");   maxPsi_ = std::max(maxPsi_, a_[k]); }
            if (o == E_COV)   { if (a_[k] < 0) throw std::invalid_argument("expr: cov index < 0");   maxCov_ = std::max(maxCov_, a_[k]); }
            if (o == E_X)     { if (a_[k] < 0) throw std::invalid_argument("expr: x index < 0");     maxX_   = std::max(maxX_, a_[k]); }
            if (o == E_EXTRA) { if (a_[k] < 0) throw std::invalid_argument("expr: extra index < 0"); maxExtra_ = std::max(maxExtra_, a_[k]); }
            if (o == E_DT) dtNode_ = k;
            if (o == E_DAMT) damtNode_ = k;
            if (o == E_DDUR) ddurNode_ = k;
            if (o == E_DCMT) dcmtNode_ = k;
            if (o == E_YTYPE) usesRow_ = true;
            if (o == E_X) usesRow_ = true;
            lvl2_[k] = l2 ? 1 : 0;
            isVec_[k] = vec ? 1 : 0;
            if (!l2) (vec ? vecIdx_ : scalIdx_).push_back(k);
        }
        for (int o : outs_) if (lvl2_[o]) throw std::invalid_argument("expr: output depends on a dose value outside doses.superpose");
        // Level-2 nodes reachable from the core of each E_SUMDOSE (topological order).
        for (int k = 0; k < N; ++k) {
            if (op_[k] != E_SUMDOSE) continue;
            std::vector<char> seen(N, 0);
            std::vector<int> st{a_[k]}, sub;
            while (!st.empty()) {
                const int u = st.back(); st.pop_back();
                if (!lvl2_[u] || seen[u]) continue;
                seen[u] = 1;
                const int o = op_[u];
                if (!(o == E_DT || o == E_DAMT || o == E_DDUR || o == E_DCMT)) sub.push_back(u);
                const int na = nargs(o);
                const int ar[3] = {a_[u], b_[u], c_[u]};
                for (int t = 0; t < na; ++t) st.push_back(ar[t]);
            }
            std::sort(sub.begin(), sub.end());
            subOf_[k] = static_cast<int>(subs_.size());
            subs_.push_back(std::move(sub));
        }
    }

    /**
     * @brief Builds a single-output program (Gaussian prediction).
     * @param op Opcode per node.
     * @param a First argument index per node.
     * @param b Second argument index per node.
     * @param c Third argument index per node.
     * @param cst Constant value per node.
     * @param out Index of the single output node.
     * @throws std::invalid_argument Under the same conditions as the
     *         multi-output constructor.
     */
    ExprProgram(std::vector<int> op, std::vector<int> a, std::vector<int> b, std::vector<int> c,
                std::vector<double> cst, int out)
        : ExprProgram(std::move(op), std::move(a), std::move(b), std::move(c), std::move(cst), std::vector<int>{out}) {}

    /**
     * @brief Returns the number of operand indices consumed by an opcode.
     * @param o Opcode to query.
     * @return 0, 1, 2, or 3 depending on the operation.
     */
    static int nargs(int o) {
        switch (o) {
            case E_CONST: case E_PSI: case E_X: case E_COV: return 0;
            case E_ADD: case E_SUB: case E_MUL: case E_DIV: case E_POW:
            case E_MAX: case E_MIN: case E_ATAN2: case E_MOD:
            case E_LT: case E_LE: case E_GT: case E_GE: case E_EQ: case E_NE:
            case E_AND: case E_OR: return 2;
            case E_WHERE: return 3;
            case E_YTYPE: case E_Y: case E_EXTRA: case E_DT: case E_DAMT: case E_DDUR: case E_DCMT: return 0;
            case E_SUMDOSE: return 2;
            default: return 1;
        }
    }

    /** @brief Returns the number of nodes in the program. */
    int size() const { return static_cast<int>(op_.size()); }
    /** @brief Returns the number of outputs in the program. */
    int nOut() const { return static_cast<int>(outs_.size()); }

    /**
     * @brief Evaluates the first output for every observation (Gaussian prediction).
     *
     * Scalar nodes are hoisted and evaluated once; vector nodes are evaluated
     * once per row. If the first output is scalar, its value is replicated for
     * all @p n entries of @p f.
     *
     * @param psi Individual parameters (read-only).
     * @param npsi Number of available entries in @p psi.
     * @param cov Subject covariates (read-only).
     * @param ncov Number of available entries in @p cov.
     * @param x Observation matrix with @p n rows and @p stride columns; the
     *        last column is the response type.
     * @param n Number of observation rows.
     * @param stride Number of columns per row.
     * @param doses Expanded dose record for the subject, or nullptr.
     * @param f Output array with room for @p n values.
     * @throws std::invalid_argument If the program references more psi,
     *         covariate, x, or extra values than were supplied.
     * @note Thread-safe; uses a per-thread scratch buffer.
     */
    void eval(const double* psi, int npsi, const double* cov, int ncov, const double* x, int n, int stride,
              const std::vector<ode::Dose>* doses, double* f) const {
        if (n <= 0) return;
        check(npsi, ncov, stride, 0, 0);
        Ctx cx{psi, cov, nullptr, stride, 0.0, nullptr, nullptr, doses};
        double* r = work();
        for (int k : scalIdx_) r[k] = node(k, r, cx);
        const int out = outs_[0];
        if (!isVec_[out]) {
            const double v = r[out];
            for (int j = 0; j < n; ++j) f[j] = v;
            return;
        }
        for (int j = 0; j < n; ++j) {
            cx.row = x + static_cast<std::size_t>(j) * stride;
            for (int k : vecIdx_) r[k] = node(k, r, cx);
            f[j] = r[out];
        }
    }

    /**
     * @brief Evaluates all outputs for a single row.
     *
     * Intended for non-Gaussian families and custom likelihoods. Scalar nodes
     * are hoisted; all vector nodes are evaluated for the supplied row.
     *
     * @param psi Individual parameters (read-only).
     * @param npsi Number of available entries in @p psi.
     * @param cov Subject covariates (read-only).
     * @param ncov Number of available entries in @p cov.
     * @param row Predictor row with @p stride columns, or nullptr when the
     *        graph does not use x.
     * @param stride Number of columns in @p row.
     * @param y Observation value exposed through E_Y.
     * @param extra Optional extra parameters exposed through E_EXTRA.
     * @param nextra Number of available entries in @p extra.
     * @param t Optional value that replaces x[0] (hazard/cumulative hazard
     *        for time-to-event models).
     * @param out Output array with room for nOut() values.
     * @throws std::invalid_argument If the graph uses x but @p row is null, or
     *         if the program references more inputs than were supplied.
     * @note Thread-safe; uses a per-thread scratch buffer.
     */
    void evalRow(const double* psi, int npsi, const double* cov, int ncov, const double* row, int stride,
                 double y, const double* extra, int nextra, const double* t, double* out) const {
        if (usesRow_ && row == nullptr) throw std::invalid_argument("expr: the graph uses x but no row was provided");
        check(npsi, ncov, row ? stride : 1 << 30, 0, nextra, extra != nullptr);
        Ctx cx{psi, cov, row, stride, y, extra, t, nullptr};
        double* r = work();
        for (int k : scalIdx_) r[k] = node(k, r, cx);
        for (int k : vecIdx_) r[k] = node(k, r, cx);
        for (std::size_t i = 0; i < outs_.size(); ++i) out[i] = r[outs_[i]];
    }

private:
    struct Ctx {
        const double* psi; const double* cov; const double* row; int stride;
        double y; const double* extra; const double* t; const std::vector<ode::Dose>* doses;
    };
    std::vector<int> op_, a_, b_, c_;
    std::vector<double> cst_;
    std::vector<int> outs_;
    std::vector<char> isVec_, lvl2_;
    std::vector<int> scalIdx_, vecIdx_, subOf_;
    std::vector<std::vector<int>> subs_;
    int maxPsi_ = -1, maxCov_ = -1, maxX_ = -1, maxExtra_ = -1;
    int dtNode_ = -1, damtNode_ = -1, ddurNode_ = -1, dcmtNode_ = -1;
    bool usesRow_ = false;

    double* work() const {
        thread_local std::vector<double> R;
        if (R.size() < op_.size()) R.resize(op_.size());
        return R.data();
    }
    void check(int npsi, int ncov, int stride, int, int nextra, bool haveExtra = true) const {
        if (maxPsi_ >= npsi) throw std::invalid_argument("expr: the model uses more parameters than are available");
        if (maxCov_ >= ncov) throw std::invalid_argument("expr: the model uses more covariates than are available");
        if (maxX_ >= stride) throw std::invalid_argument("expr: the model uses more x columns than are available");
        if (maxExtra_ >= 0 && (!haveExtra || maxExtra_ >= nextra)) throw std::invalid_argument("expr: the model uses more extra values than are available");
    }

    static double nmax(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? std::numeric_limits<double>::quiet_NaN() : (a > b ? a : b); }
    static double nmin(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? std::numeric_limits<double>::quiet_NaN() : (a < b ? a : b); }
    static double lgam(double x) {
#if defined(_WIN32)
        return std::lgamma(x);
#else
        int sg = 0;
        return ::lgamma_r(x, &sg);          // reentrant: std::lgamma writes the global signgam
#endif
    }

    inline double node(int k, double* r, const Ctx& cx) const {
        const int o = op_[k];
        const double A = (o >= E_ADD && o < E_YTYPE) ? r[a_[k]] : 0.0;
        switch (o) {
            case E_CONST: return cst_[k];
            case E_PSI:   return cx.psi[a_[k]];
            case E_COV:   return cx.cov[a_[k]];
            case E_X:     return (cx.t && a_[k] == 0) ? *cx.t : cx.row[a_[k]];
            case E_YTYPE: return cx.row[cx.stride - 1];
            case E_Y:     return cx.y;
            case E_EXTRA: return cx.extra[a_[k]];
            case E_DT: case E_DAMT: case E_DDUR: case E_DCMT: return r[k];   // set by the E_SUMDOSE loop
            case E_SUMDOSE: {
                double acc = 0.0;
                if (cx.doses) {
                    const double tv = r[b_[k]];
                    const std::vector<int>& sub = subs_[subOf_[k]];
                    const int core = a_[k];
                    for (const auto& d : *cx.doses) {
                        if (d.ss != 0) throw std::invalid_argument("doses.superpose does not support steady state (SS): use compile=False or a built-in model");
                        if (d.modeled != 0) throw std::invalid_argument("doses.superpose does not support RATE = -1/-2 (modeled rate or duration)");
                        const double dt = tv - d.time;
                        if (!(dt >= 0.0)) continue;
                        if (dtNode_ >= 0) r[dtNode_] = dt;
                        if (damtNode_ >= 0) r[damtNode_] = d.amount;
                        if (ddurNode_ >= 0) r[ddurNode_] = d.duration;
                        if (dcmtNode_ >= 0) r[dcmtNode_] = double(d.cmt + 1);
                        for (int u : sub) r[u] = node(u, r, cx);
                        acc += r[core];
                    }
                }
                return acc;
            }
            case E_ADD:   return A + r[b_[k]];
            case E_SUB:   return A - r[b_[k]];
            case E_MUL:   return A * r[b_[k]];
            case E_DIV:   return A / r[b_[k]];
            case E_POW:   return std::pow(A, r[b_[k]]);
            case E_NEG:   return -A;
            case E_EXP:   return std::exp(A);
            case E_LOG:   return std::log(A);
            case E_LOG10: return std::log10(A);
            case E_LOG2:  return std::log2(A);
            case E_LOG1P: return std::log1p(A);
            case E_EXPM1: return std::expm1(A);
            case E_EXP2:  return std::exp2(A);
            case E_SQRT:  return std::sqrt(A);
            case E_CBRT:  return std::cbrt(A);
            case E_ABS:   return std::fabs(A);
            case E_SIGN:  return std::isnan(A) ? A : double((A > 0.0) - (A < 0.0));
            case E_SIN:   return std::sin(A);
            case E_COS:   return std::cos(A);
            case E_TAN:   return std::tan(A);
            case E_ASIN:  return std::asin(A);
            case E_ACOS:  return std::acos(A);
            case E_ATAN:  return std::atan(A);
            case E_SINH:  return std::sinh(A);
            case E_COSH:  return std::cosh(A);
            case E_TANH:  return std::tanh(A);
            case E_ASINH: return std::asinh(A);
            case E_ACOSH: return std::acosh(A);
            case E_ATANH: return std::atanh(A);
            case E_FLOOR: return std::floor(A);
            case E_CEIL:  return std::ceil(A);
            case E_SQUARE: return A * A;
            case E_RECIP: return 1.0 / A;
            case E_MAX:   return nmax(A, r[b_[k]]);
            case E_MIN:   return nmin(A, r[b_[k]]);
            case E_ATAN2: return std::atan2(A, r[b_[k]]);
            case E_MOD: {                                   // np.remainder: the result takes the sign of the divisor
                const double b = r[b_[k]];
                double m = std::fmod(A, b);
                if (m != 0.0 && ((m < 0.0) != (b < 0.0))) m += b;
                return m;
            }
            case E_LT:    return A <  r[b_[k]] ? 1.0 : 0.0;
            case E_LE:    return A <= r[b_[k]] ? 1.0 : 0.0;
            case E_GT:    return A >  r[b_[k]] ? 1.0 : 0.0;
            case E_GE:    return A >= r[b_[k]] ? 1.0 : 0.0;
            case E_EQ:    return A == r[b_[k]] ? 1.0 : 0.0;
            case E_NE:    return A != r[b_[k]] ? 1.0 : 0.0;
            case E_AND:   return (A != 0.0 && r[b_[k]] != 0.0) ? 1.0 : 0.0;
            case E_OR:    return (A != 0.0 || r[b_[k]] != 0.0) ? 1.0 : 0.0;
            case E_NOT:   return A != 0.0 ? 0.0 : 1.0;
            case E_WHERE: return A != 0.0 ? r[b_[k]] : r[c_[k]];
            case E_ERF:   return std::erf(A);
            case E_ERFC:  return std::erfc(A);
            case E_EXPIT: return A >= 0.0 ? 1.0 / (1.0 + std::exp(-A)) : std::exp(A) / (1.0 + std::exp(A));
            case E_GAMMALN: return lgam(A);
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
};

} // namespace mcmckl
