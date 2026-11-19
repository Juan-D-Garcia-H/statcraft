// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

#ifndef LBFGSB_LOGGER_HPP
#define LBFGSB_LOGGER_HPP

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <functional>
#include <chrono>
#include "../common.hpp"

namespace min {

/**
 * @file logger.hpp
 * @ingroup lbfgsb_driver
 * @brief Structured logging helper for the L-BFGS-B minimizer.
 *
 * The logger centralizes all console-style output produced by the L-BFGS-B
 * driver. Messages are routed through a single `write()` primitive that
 * prefers a user-supplied sink (a `std::function<void(std::string_view)>`)
 * when present, and otherwise falls back to a `std::ostream`. All
 * output-producing methods honor the configured @ref VerbosityLevel, so a
 * `Silent` logger becomes a complete no-op.
 *
 * Output categories:
 *  - **Header**: problem size, machine epsilon, and (at Debug level) the
 *    initial point and bounds.
 *  - **Table header / iteration rows**: compact or detailed iteration table.
 *  - **Summary**: final status, counters, timing breakdown.
 *  - **Info messages**: free-form notices prefixed with `[L-BFGS-B]`.
 *
 * @tparam FP Floating-point scalar type (only used to format numeric output).
 */
template<typename FP = double>
class Logger {
private:
    VerbosityLevel level_;                    ///< Configured verbosity level.
    std::ostream* os_;                        ///< Fallback output stream (may be null).
    std::function<void(std::string_view)> sink_; ///< Optional user-supplied sink.

    /**
     * @brief Route a message to the configured destination.
     *
     * If a sink is installed, it receives the message; otherwise the message
     * is written to the fallback stream (if any) and the stream is flushed.
     * When neither is available the message is silently discarded.
     */
    void write(std::string_view msg) const {
        if (sink_) {
            sink_(msg);
        } else if (os_) {
            (*os_) << msg;
            os_->flush();
        }
    }

public:
    /**
     * @brief Construct a logger.
     *
     * @param level Verbosity level; below `Summary` all output is suppressed.
     * @param os    Fallback output stream; may be null if a sink is provided.
     * @param sink  Optional per-message callback; takes precedence over @p os.
     */
    explicit Logger(VerbosityLevel level = VerbosityLevel::Silent,
                    std::ostream* os = &std::cout,
                    std::function<void(std::string_view)> sink = nullptr)
        : level_(level), os_(os), sink_(sink) {}

    /** @brief Return the configured verbosity level. */
    VerbosityLevel level() const { return level_; }

    // ── Header output ───────────────────────────────────────────────────────

    /**
     * @brief Print the startup banner and (at Debug level) the initial setup.
     *
     * Emitted only when `level_ >= Summary`. At `Debug` verbosity, previews of
     * the initial point and the lower/upper bound vectors are also printed
     * (truncated to the first 6 components).
     *
     * @param n      Problem dimension.
     * @param m      L-BFGS memory size (number of correction pairs).
     * @param epsmch Machine epsilon.
     * @param l      Lower bound vector.
     * @param u      Upper bound vector.
     * @param x0     Initial point.
     */
    void print_header(int n, int m, FP epsmch, const Vec<FP>& l, const Vec<FP>& u, const Vec<FP>& x0) const {
        if (level_ < VerbosityLevel::Summary) return;

        std::ostringstream ss;
        ss << "\n============================================================\n"
           << "              L-BFGS-B Modern Minimizer                     \n"
           << "============================================================\n"
           << "  Problem dimension (n)        : " << n << "\n"
           << "  L-BFGS corrections (m)       : " << m << "\n"
           << "  Machine precision (eps)      : " << std::scientific << std::setprecision(3) << epsmch << "\n"
           << "------------------------------------------------------------\n\n";

        if (level_ >= VerbosityLevel::Debug) {
            ss << "  [Initial Setup]\n";
            ss << "  x0 = [";
            for (int i = 0; i < std::min<int>(n, 6); ++i) ss << (i ? ", " : "") << std::scientific << std::setprecision(4) << x0(i);
            if (n > 6) ss << ", ...";
            ss << "]\n";
            ss << "  l  = [";
            for (int i = 0; i < std::min<int>(n, 6); ++i) ss << (i ? ", " : "") << std::scientific << std::setprecision(4) << l(i);
            if (n > 6) ss << ", ...";
            ss << "]\n";
            ss << "  u  = [";
            for (int i = 0; i < std::min<int>(n, 6); ++i) ss << (i ? ", " : "") << std::scientific << std::setprecision(4) << u(i);
            if (n > 6) ss << ", ...";
            ss << "]\n\n";
        }
        write(ss.str());
    }

    // ── Iteration header ────────────────────────────────────────────────────

    /**
     * @brief Print the column header for the iteration table.
     *
     * Emitted only when `level_ >= Iteration`. The set of columns depends on
     * the verbosity level:
     *  - `Iteration`:            Iter, nfgv, Sub, f(x), |proj g|.
     *  - `Detailed` and higher:  adds nact, nseg, iback, step, ||dx||.
     */
    void print_table_header() const {
        if (level_ < VerbosityLevel::Iteration) return;

        std::ostringstream ss;
        if (level_ >= VerbosityLevel::Detailed) {
            ss << "  "
               << std::left
               << std::setw(7)  << "Iter"
               << std::setw(7)  << "nfgv"
               << std::setw(7)  << "nact"
               << std::setw(7)  << "nseg"
               << std::setw(6)  << "Sub"
               << std::setw(7)  << "iback"
               << std::right
               << std::setw(13) << "step"
               << std::setw(13) << "||dx||"
               << std::setw(14) << "|proj g|"
               << std::setw(15) << "f(x)"
               << "\n  " << std::string(98, '-') << "\n";
        } else {
            ss << "  "
               << std::left
               << std::setw(7)  << "Iter"
               << std::setw(7)  << "nfgv"
               << std::setw(6)  << "Sub"
               << std::right
               << std::setw(15) << "f(x)"
               << std::setw(14) << "|proj g|"
               << "\n  " << std::string(49, '-') << "\n";
        }
        write(ss.str());
    }

    // ── Iteration row ───────────────────────────────────────────────────────

    /**
     * @brief Print a single iteration row.
     *
     * Emitted only when `level_ >= Iteration`. Column layout matches the
     * header produced by @ref print_table_header. At `Debug` verbosity,
     * previews of the current iterate @p x and gradient @p g are appended
     * (truncated to the first 4 components).
     *
     * The @p sub code is rendered as a 3-letter abbreviation:
     *  - `Constant`  -> `con`
     *  - `Boundary`  -> `bnd`
     *  - `Truncated` -> `TNT`
     *  - other       -> `---`
     *
     * @param iter  Iteration index.
     * @param nfgv  Cumulative function/gradient evaluation count.
     * @param nact  Number of active bound constraints.
     * @param nseg  Number of segments in the Cauchy point computation.
     * @param sub   Subspace minimization status.
     * @param iback Number of backtracking steps used by the line search.
     * @param stp   Line-search step length.
     * @param xstep Norm of the parameter update ||dx||.
     * @param sbgnrm Infinity-norm of the projected gradient.
     * @param f     Current objective value.
     * @param x     Current iterate (used only at Debug verbosity).
     * @param g     Current gradient (used only at Debug verbosity).
     */
    void print_iteration(int iter, int nfgv, int nact, int nseg, SubspaceStatus sub,
                         int iback, FP stp, FP xstep, FP sbgnrm, FP f,
                         const Vec<FP>& x, const Vec<FP>& g) const {
        if (level_ < VerbosityLevel::Iteration) return;

        std::string_view sub_str = "---";
        switch (sub) {
            case SubspaceStatus::Constant:  sub_str = "con"; break;
            case SubspaceStatus::Boundary:  sub_str = "bnd"; break;
            case SubspaceStatus::Truncated: sub_str = "TNT"; break;
            default: break;
        }

        std::ostringstream ss;
        if (level_ >= VerbosityLevel::Detailed) {
            ss << "  "
               << std::left
               << std::setw(7)  << iter
               << std::setw(7)  << nfgv
               << std::setw(7)  << nact
               << std::setw(7)  << nseg
               << std::setw(6)  << sub_str
               << std::setw(7)  << iback
               << std::right << std::scientific << std::setprecision(4)
               << std::setw(13) << stp
               << std::setw(13) << xstep
               << std::setw(14) << sbgnrm
               << std::setw(15) << f
               << "\n";

            if (level_ >= VerbosityLevel::Debug) {
                ss << "     x preview: [";
                for (Eigen::Index i = 0; i < std::min<Eigen::Index>(x.size(), 4); ++i) ss << (i ? ", " : "") << x(i);
                ss << "]\n";
                ss << "     g preview: [";
                for (Eigen::Index i = 0; i < std::min<Eigen::Index>(g.size(), 4); ++i) ss << (i ? ", " : "") << g(i);
                ss << "]\n";
            }
        } else {
            ss << "  "
               << std::left
               << std::setw(7)  << iter
               << std::setw(7)  << nfgv
               << std::setw(6)  << sub_str
               << std::right << std::scientific << std::setprecision(4)
               << std::setw(15) << f
               << std::setw(14) << sbgnrm
               << "\n";
        }
        write(ss.str());
    }

    // ── Final Summary ───────────────────────────────────────────────────────

    /**
     * @brief Print the final summary block.
     *
     * Emitted only when `level_ >= Summary`. Reports the termination message,
     * error code, iteration and evaluation counters, the number of active
     * constraints, the final objective value and projected-gradient norm, and
     * a per-phase timing breakdown (all durations are given in seconds and
     * displayed in milliseconds).
     *
     * @param n            Problem dimension.
     * @param iter         Number of iterations performed.
     * @param nfgv         Number of function/gradient evaluations performed.
     * @param nact         Number of active bound constraints at termination.
     * @param sbgnrm       Final infinity-norm of the projected gradient.
     * @param f            Final objective value.
     * @param err          Final error code.
     * @param msg          Human-readable termination message.
     * @param t_total      Total wall time (seconds).
     * @param t_cauchy     Time spent in the Cauchy-point search (seconds).
     * @param t_subspace   Time spent in subspace minimization (seconds).
     * @param t_linesearch Time spent in line search (seconds).
     */
    void print_summary(int n, int iter, int nfgv, int nact, FP sbgnrm, FP f,
                       ErrorCode err, std::string_view msg,
                       FP t_total, FP t_cauchy, FP t_subspace, FP t_linesearch) const {
        if (level_ < VerbosityLevel::Summary) return;

        std::ostringstream ss;
        ss << "\n============================================================\n"
           << "                    FINAL SUMMARY                           \n"
           << "============================================================\n"
           << "  Status               : " << msg << "\n"
           << "  Error code           : " << static_cast<int>(err) << "\n"
           << "  Iterations           : " << iter << "\n"
           << "  Function evaluations : " << nfgv << "\n"
           << "  Active constraints   : " << nact << " / " << n << "\n"
           << "  Final f(x)           : " << std::scientific << std::setprecision(6) << f << "\n"
           << "  Projected grad norm  : " << std::scientific << std::setprecision(6) << sbgnrm << "\n"
           << "------------------------------------------------------------\n"
           << "  Timing Breakdown:\n"
           << "  - Cauchy search      : " << std::fixed << std::setprecision(4) << (t_cauchy * 1000.0) << " ms\n"
           << "  - Subspace min       : " << std::fixed << std::setprecision(4) << (t_subspace * 1000.0) << " ms\n"
           << "  - Line search        : " << std::fixed << std::setprecision(4) << (t_linesearch * 1000.0) << " ms\n"
           << "  - Total time         : " << std::fixed << std::setprecision(4) << (t_total * 1000.0) << " ms\n"
           << "============================================================\n\n";
        write(ss.str());
    }

    /**
     * @brief Print a free-form informational message.
     *
     * Emitted only when `level_ >= Summary`. The message is prefixed with
     * `[L-BFGS-B]` to distinguish it from tabular output.
     *
     * @param msg Message text.
     */
    void print_info_message(std::string_view msg) const {
        if (level_ < VerbosityLevel::Summary) return;
        std::ostringstream ss;
        ss << "  [L-BFGS-B] " << msg << "\n";
        write(ss.str());
    }
};

} // namespace min

#endif // LBFGSB_LOGGER_HPP
