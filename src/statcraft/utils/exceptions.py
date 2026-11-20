# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
exceptions
==========
Package-specific exception hierarchy for the ``nls``/``FormulaParser``/
``Initializer``/``NLS`` stack.

Design goal: let callers catch precise, semantically meaningful errors
(``except ConvergenceError``, ``except FormulaError``) instead of a bare
``except ValueError`` that would also swallow unrelated errors from other
libraries. At the same time, every exception here also inherits from the
built-in exception type it conceptually replaces (``ValueError``,
``TypeError`` or ``RuntimeError``), so existing code written against the
previous, plain-builtin exceptions keeps working unchanged — this is a
purely additive, non-breaking change.

Hierarchy
---------
::

    NLSError (Exception)
    ├── FormulaError (NLSError, ValueError)
    │   └── FormulaNamespaceError (FormulaError)   -- defined in FormulaParser.py
    ├── DataValidationError (NLSError, ValueError)
    ├── NotFittedError (NLSError, RuntimeError)
    └── FitError (NLSError, RuntimeError)
        └── ConvergenceError (FitError)

Where each one is raised
-------------------------
- ``FormulaError``: the formula string itself is malformed, references an
  unknown function/name, uses a name that collides with a reserved
  identifier, or an ``SS*`` self-starting model is invoked incorrectly.
- ``DataValidationError``: the formula is fine, but the supplied
  ``pandas.DataFrame`` (or a user-supplied ``start``/``p0``) does not
  satisfy what's needed to fit against it — missing columns, empty data,
  NaN/Inf values, mismatched ``start`` keys, etc.
- ``NotFittedError``: a fitted-only property or method (e.g. ``rsquared``,
  ``predict()``) is accessed before calling ``fit()``/``fit_multistart()``.
- ``FitError``: something went wrong while actually running the optimizer
  (invalid ``bounds``/``cov_type``, all multistart attempts raised, the
  underlying ``scipy.optimize.least_squares`` call itself raised).
- ``ConvergenceError`` (a kind of ``FitError``): the optimizer *ran*
  successfully but did not converge. By default this is only surfaced as
  a ``RuntimeWarning`` (to match the previous behavior and because a
  non-converged fit is still often inspectable), but ``fit()`` and
  ``fit_multistart()`` accept ``raise_on_non_convergence=True`` to raise
  this exception instead.
"""


class NLSError(Exception):
    """Base class for every exception raised by this package."""


class FormulaError(NLSError, ValueError):
    """
    Raised when a formula string is malformed or otherwise invalid:
    missing ``~``, empty left/right-hand side, an identifier that isn't a
    valid Python name, a disallowed transformation on the response
    variable, an unknown function/name in the namespace, an unsupported or
    incorrectly-invoked ``SS*`` self-starting model, and similar
    formula-level problems.

    Inherits from ``ValueError`` so existing ``except ValueError`` clauses
    written against the previous behavior keep working unchanged.
    """


class DataValidationError(NLSError, ValueError):
    """
    Raised when the formula itself is valid but the data (or a
    user-supplied starting value / bounds) does not satisfy what fitting
    requires: an empty ``DataFrame``, missing columns, NaN/Inf values,
    ``start``/``p0`` keys that don't match the formula's parameters, and
    similar data-level problems.

    Inherits from ``ValueError`` so existing ``except ValueError`` clauses
    written against the previous behavior keep working unchanged.
    """


class NotFittedError(NLSError, RuntimeError):
    """
    Raised when a method or property that requires a fitted model (e.g.
    ``rsquared``, ``predict()``, ``summary()``) is accessed before calling
    ``fit()`` or ``fit_multistart()``.

    Inherits from ``RuntimeError`` so existing ``except RuntimeError``
    clauses written against the previous behavior keep working unchanged.
    """


class FitError(NLSError, RuntimeError):
    """
    Raised when something goes wrong while configuring or running the
    optimizer itself: invalid ``bounds``, an unsupported ``cov_type``, an
    analytical jacobian requested on a backend that doesn't support it,
    every attempt of ``fit_multistart`` raising, or the underlying
    ``scipy.optimize.least_squares`` call raising unexpectedly.

    Inherits from ``RuntimeError`` so existing ``except RuntimeError``
    clauses written against the previous behavior keep working unchanged.
    """


class ConvergenceError(FitError):
    """
    Raised instead of a ``RuntimeWarning`` when ``fit()`` or
    ``fit_multistart()`` is called with ``raise_on_non_convergence=True``
    and the optimizer terminates without satisfying its convergence
    criteria (``OptimizeResult.success`` is ``False``, or ``status <= 0``).

    A ``ConvergenceError`` still carries a usable, already-computed result
    on the instance that raised it (``coefficients``, ``residuals``, etc.)
    — the fit did run, it just didn't converge — so callers may choose to
    inspect the model instead of only catching and discarding the error.
    """
