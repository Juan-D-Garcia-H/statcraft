# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Binary classification evaluation for fitted models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .adapters import resolve
from .foundation import (
    DEFAULT_COLORS,
    PlotResult,
    panel_axes,
    single_axis,
    style_axis,
)
from .predictions import _cmap, _multiclass_arrays

__all__ = [
    "ClassificationEvaluator",
    "roc_curve",
    "precision_recall_curve",
    "confusion_matrix_plot",
    "calibration_curve",
    "multiclass_calibration_curve",
    "gains_chart",
    "lift_chart",
    "threshold_metrics",
]


def _trapezoid(y, x) -> float:
    fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(fn(y, x))


def _as_binary(values, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    unique = np.unique(arr[finite])
    if unique.size and not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(
            f"{what} must contain only 0/1 values for classification evaluation "
            f"(found {unique[:5].tolist()} ...)."
        )
    return arr


def _extract_binary(model, exog=None, y_true=None) -> tuple[np.ndarray, np.ndarray]:
    data = resolve(model)
    if exog is None:
        y_obs = data.response
        y_prob = data.fitted
    else:
        if y_true is None:
            raise ValueError("pass y_true together with exog for out-of-sample evaluation")
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise TypeError(f"{data.model_name} does not expose predict()")
        try:
            y_prob = predict(exog=exog)
        except TypeError:
            y_prob = predict(exog)
        y_obs = y_true
    if y_obs is None or y_prob is None:
        raise ValueError(f"observed values or fitted probabilities are not available for {data.model_name}")
    y_obs = _as_binary(y_obs, "the observed response")
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_prob.size != y_obs.size:
        raise ValueError("the observed response and predicted probabilities have different lengths")
    keep = np.isfinite(y_obs) & np.isfinite(y_prob)
    return y_obs[keep], y_prob[keep]


class ClassificationEvaluator:
    """Binary classification metrics and plots for a fitted model.

    Parameters
    ----------
    model : fitted result
        Any result whose fitted values are probabilities in ``[0, 1]`` (most
        commonly a logistic ``Binomial`` GLM).
    exog : array-like, optional
        New predictors to evaluate out of sample.  Requires ``y_true``.
    y_true : array-like, optional
        Observed 0/1 response matching ``exog``.
    labels : sequence of str, optional
        Names for the negative and positive classes.
    threshold : float, default 0.5
        Default decision threshold.  ``"youden"`` and ``"f1"`` select the
        optimal thresholds after the curves are built.
    """

    def __init__(self, model, exog=None, y_true=None, labels=None, threshold: float | str = 0.5):
        self.model = model
        self.y_true, self.y_prob = _extract_binary(model, exog=exog, y_true=y_true)
        if self.y_true.size == 0:
            raise ValueError("no finite binary observations are available")
        if np.unique(self.y_true).size < 2:
            raise ValueError("both classes (0 and 1) must be present to evaluate a classifier")
        default_labels = ["Class 0", "Class 1"]
        self.labels = list(labels) if labels is not None else default_labels

        self.fpr, self.tpr, self.roc_thresholds = self._compute_roc(self.y_true, self.y_prob)
        self.roc_auc = _trapezoid(self.tpr, self.fpr)
        j_scores = self.tpr - self.fpr
        self._best_roc_idx = int(np.argmax(j_scores))
        self.optimal_threshold_youden = float(self.roc_thresholds[self._best_roc_idx])
        if not np.isfinite(self.optimal_threshold_youden):
            self.optimal_threshold_youden = 0.5

        self.precision_curve, self.recall_curve, self.pr_thresholds = self._compute_pr(self.y_true, self.y_prob)
        self.pr_auc = _trapezoid(self.precision_curve, self.recall_curve)
        self.prevalence = float(np.mean(self.y_true == 1))

        p_vals = self.precision_curve[:-1]
        r_vals = self.recall_curve[:-1]
        denom = p_vals + r_vals
        f1_vals = np.where(denom > 0, 2.0 * p_vals * r_vals / np.where(denom > 0, denom, 1.0), 0.0)
        self._best_pr_idx = int(np.argmax(f1_vals)) if f1_vals.size else 0
        self.optimal_threshold_f1 = (
            float(self.pr_thresholds[self._best_pr_idx]) if self.pr_thresholds.size else 0.5
        )

        self.default_threshold = threshold
        self._ci_cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------ curves
    @staticmethod
    def _compute_roc(y_true: np.ndarray, y_prob: np.ndarray):
        order = np.argsort(-y_prob, kind="mergesort")
        y_sorted = y_true[order]
        p_sorted = y_prob[order]
        distinct = np.where(np.diff(p_sorted))[0]
        threshold_idx = np.r_[distinct, y_sorted.size - 1]
        tps = np.cumsum(y_sorted)[threshold_idx]
        fps = 1 + threshold_idx - tps
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[np.inf, p_sorted[threshold_idx]]
        total_pos = max(float(np.sum(y_sorted == 1)), 1.0)
        total_neg = max(float(np.sum(y_sorted == 0)), 1.0)
        return fps / total_neg, tps / total_pos, thresholds

    @staticmethod
    def _compute_pr(y_true: np.ndarray, y_prob: np.ndarray):
        order = np.argsort(-y_prob, kind="mergesort")
        y_sorted = y_true[order]
        p_sorted = y_prob[order]
        distinct = np.where(np.diff(p_sorted))[0]
        threshold_idx = np.r_[distinct, y_sorted.size - 1]
        tps = np.cumsum(y_sorted)[threshold_idx]
        fps = 1 + threshold_idx - tps
        total_pos = max(float(np.sum(y_sorted == 1)), 1.0)
        precision = tps / np.maximum(tps + fps, 1e-12)
        recall = tps / total_pos
        return np.r_[1.0, precision], np.r_[0.0, recall], p_sorted[threshold_idx]

    # ------------------------------------------------------------------ metrics
    def resolve_threshold(self, threshold=None) -> float:
        value = self.default_threshold if threshold is None else threshold
        if isinstance(value, str):
            key = value.lower()
            if key == "youden":
                return self.optimal_threshold_youden
            if key == "f1":
                return self.optimal_threshold_f1
            raise ValueError("threshold must be a number, 'youden' or 'f1'")
        return float(value)

    def predict_class(self, threshold=None) -> np.ndarray:
        return (self.y_prob >= self.resolve_threshold(threshold)).astype(int)

    def confusion_matrix(self, threshold=None) -> np.ndarray:
        """Confusion matrix ``[[TN, FP], [FN, TP]]``."""
        predicted = self.predict_class(threshold)
        tn = int(np.sum((self.y_true == 0) & (predicted == 0)))
        fp = int(np.sum((self.y_true == 0) & (predicted == 1)))
        fn = int(np.sum((self.y_true == 1) & (predicted == 0)))
        tp = int(np.sum((self.y_true == 1) & (predicted == 1)))
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    def confusion_matrix_df(self, threshold=None) -> pd.DataFrame:
        matrix = self.confusion_matrix(threshold)
        return pd.DataFrame(
            matrix,
            index=[f"Actual: {self.labels[0]}", f"Actual: {self.labels[1]}"],
            columns=[f"Predicted: {self.labels[0]}", f"Predicted: {self.labels[1]}"],
        )

    def metrics_summary(self, threshold=None, include_ci: bool = False, alpha: float = 0.05, n_bootstraps: int = 1000, seed: int = 42) -> dict:
        """Accuracy, precision, recall, specificity, F1 and AUCs at a threshold."""
        used = self.resolve_threshold(threshold)
        tn, fp, fn, tp = self.confusion_matrix(used).ravel()
        total = max(tn + fp + fn + tp, 1)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp + fp) and (tp + fn) and (tn + fp) and (tn + fn) else 0.0
        summary = {
            "threshold": used,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "prevalence_baseline": self.prevalence,
            "accuracy": (tp + tn) / total,
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "negative_predictive_value": npv,
            "balanced_accuracy": 0.5 * (recall + specificity),
            "f1_score": f1,
            "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
            "brier_score": float(np.mean((self.y_prob - self.y_true) ** 2)),
        }
        if include_ci:
            ci = self.auc_confidence_interval(alpha=alpha, n_bootstraps=n_bootstraps, seed=seed)
            summary["roc_auc_ci"] = (ci["roc_auc"]["ci_lower"], ci["roc_auc"]["ci_upper"])
            summary["pr_auc_ci"] = (ci["pr_auc"]["ci_lower"], ci["pr_auc"]["ci_upper"])
        return summary

    def auc_confidence_interval(self, alpha: float = 0.05, n_bootstraps: int = 1000, seed: int = 42) -> dict:
        """Stratified bootstrap confidence intervals for ROC-AUC and PR-AUC."""
        key = (float(alpha), int(n_bootstraps), int(seed))
        if key in self._ci_cache:
            return self._ci_cache[key]

        rng = np.random.default_rng(seed)
        pos = np.where(self.y_true == 1)[0]
        neg = np.where(self.y_true == 0)[0]
        roc_values = np.empty(n_bootstraps, dtype=float)
        pr_values = np.empty(n_bootstraps, dtype=float)
        for b in range(n_bootstraps):
            idx = np.concatenate(
                [rng.choice(pos, size=pos.size, replace=True), rng.choice(neg, size=neg.size, replace=True)]
            )
            yb, pb = self.y_true[idx], self.y_prob[idx]
            fpr, tpr, _ = self._compute_roc(yb, pb)
            roc_values[b] = _trapezoid(tpr, fpr)
            prec, rec, _ = self._compute_pr(yb, pb)
            pr_values[b] = _trapezoid(prec, rec)

        def _summary(values, point):
            return {
                "auc": point,
                "std_err": float(np.std(values, ddof=1)),
                "ci_lower": float(np.percentile(values, 100.0 * alpha / 2.0)),
                "ci_upper": float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0))),
            }

        result = {
            "confidence_level": 1.0 - alpha,
            "n_bootstraps": n_bootstraps,
            "roc_auc": _summary(roc_values, self.roc_auc),
            "pr_auc": _summary(pr_values, self.pr_auc),
        }
        self._ci_cache[key] = result
        return result

    def gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative population share and cumulative captured-positive share."""
        order = np.argsort(-self.y_prob, kind="mergesort")
        y_sorted = self.y_true[order]
        population = np.r_[0.0, np.arange(1, y_sorted.size + 1) / y_sorted.size]
        total = max(float(np.sum(y_sorted)), 1.0)
        captured = np.r_[0.0, np.cumsum(y_sorted) / total]
        return population, captured

    def report(self, threshold=None, alpha: float = 0.05, n_bootstraps: int = 1000, seed: int = 42) -> str:
        """Return (and print) a formatted text report."""
        metrics = self.metrics_summary(threshold, include_ci=True, alpha=alpha, n_bootstraps=n_bootstraps, seed=seed)
        pct = int((1.0 - alpha) * 100)
        roc_lo, roc_hi = metrics["roc_auc_ci"]
        pr_lo, pr_hi = metrics["pr_auc_ci"]
        lines = [
            f"=== Classification evaluation (threshold = {metrics['threshold']:.4f}) ===",
            f"  ROC-AUC: {metrics['roc_auc']:.4f}  ({pct}% CI [{roc_lo:.4f}, {roc_hi:.4f}])",
            f"  PR-AUC:  {metrics['pr_auc']:.4f}  ({pct}% CI [{pr_lo:.4f}, {pr_hi:.4f}])",
            "",
            self.confusion_matrix_df(metrics["threshold"]).to_string(),
            "",
            f"  Accuracy:            {metrics['accuracy']:.4f}",
            f"  Precision (PPV):     {metrics['precision']:.4f}",
            f"  Recall (sensitivity):{metrics['recall_sensitivity']:.4f}",
            f"  Specificity:         {metrics['specificity']:.4f}",
            f"  F1 score:            {metrics['f1_score']:.4f}",
            f"  MCC:                 {metrics['mcc']:.4f}",
            f"  Brier score:         {metrics['brier_score']:.4f}",
        ]
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------ plots
    def plot_confusion_matrix(self, threshold=None, ax=None, cmap: str = "Blues", figsize=(5.0, 4.5), title: str | None = None):
        used = self.resolve_threshold(threshold)
        matrix = self.confusion_matrix(used)
        fig, ax, _ = single_axis(ax, figsize)
        ax.imshow(matrix, interpolation="nearest", cmap=cmap)
        tick = [0, 1]
        ax.set_xticks(tick)
        ax.set_yticks(tick)
        ax.set_xticklabels(self.labels)
        ax.set_yticklabels(self.labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        midpoint = matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] > midpoint else "black"
                ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color=color, fontsize=13)
        style_axis(ax, grid=False)
        label = title or f"Confusion matrix (threshold = {used:.2f})"
        ax.set_title(label)
        return PlotResult(fig, label)

    def plot_roc_curve(self, ax=None, *, show_optimal: bool = True, show_ci: bool = True, alpha: float = 0.05, n_bootstraps: int = 1000, seed: int = 42, figsize=(5.2, 4.6), title: str | None = None):
        fig, ax, _ = single_axis(ax, figsize)
        if show_ci:
            ci = self.auc_confidence_interval(alpha=alpha, n_bootstraps=n_bootstraps, seed=seed)["roc_auc"]
            label = f"ROC (AUC = {self.roc_auc:.3f}; {int((1 - alpha) * 100)}% CI {ci['ci_lower']:.3f}–{ci['ci_upper']:.3f})"
        else:
            label = f"ROC (AUC = {self.roc_auc:.4f})"
        ax.plot(self.fpr, self.tpr, color="#d97706", linewidth=2.2, label=label)
        ax.plot([0, 1], [0, 1], color="#1e3a8a", linestyle="--", linewidth=1.4, label="Chance (AUC = 0.50)")
        if show_optimal:
            ax.scatter(
                self.fpr[self._best_roc_idx],
                self.tpr[self._best_roc_idx],
                color=DEFAULT_COLORS["accent"],
                zorder=5,
                label=f"Youden optimum ({self.optimal_threshold_youden:.2f})",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False positive rate (1 − specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.legend(loc="lower right", fontsize=8.2)
        style_axis(ax)
        label = title or "ROC curve"
        ax.set_title(label)
        return PlotResult(fig, label)

    def plot_pr_curve(self, ax=None, *, show_optimal: bool = True, show_ci: bool = True, alpha: float = 0.05, n_bootstraps: int = 1000, seed: int = 42, figsize=(5.2, 4.6), title: str | None = None):
        fig, ax, _ = single_axis(ax, figsize)
        if show_ci:
            ci = self.auc_confidence_interval(alpha=alpha, n_bootstraps=n_bootstraps, seed=seed)["pr_auc"]
            label = f"PR (AUC = {self.pr_auc:.3f}; {int((1 - alpha) * 100)}% CI {ci['ci_lower']:.3f}–{ci['ci_upper']:.3f})"
        else:
            label = f"PR (AUC = {self.pr_auc:.4f})"
        ax.plot(self.recall_curve, self.precision_curve, color="#0f766e", linewidth=2.2, label=label)
        ax.axhline(self.prevalence, color=DEFAULT_COLORS["neutral"], linestyle="--", linewidth=1.4, label=f"Prevalence ({self.prevalence:.2f})")
        if show_optimal and self.pr_thresholds.size:
            idx = self._best_pr_idx + 1
            ax.scatter(self.recall_curve[idx], self.precision_curve[idx], color=DEFAULT_COLORS["accent"], zorder=5, label=f"Max F1 ({self.optimal_threshold_f1:.2f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylabel("Precision (positive predictive value)")
        ax.legend(loc="lower left", fontsize=8.2)
        style_axis(ax)
        label = title or "Precision–Recall curve"
        ax.set_title(label)
        return PlotResult(fig, label)

    def plot_calibration(
        self,
        ax=None,
        *,
        n_bins: int = 10,
        alpha: float = 0.05,
        figsize=(5.4, 4.6),
        title: str | None = None,
    ):
        """Reliability diagram with Wilson 95% intervals per probability bin.

        The bin-size annotations are offset in *points*, so the y-axis needs
        explicit headroom: with auto-scaling (or a fixed ``ylim=1.02``) the
        counts of bins whose observed rate is near 1 get clipped by the top
        spine of the axes.
        """
        from scipy.stats import norm

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(self.y_prob, edges) - 1, 0, n_bins - 1)
        xs, ys, counts, lower, upper = [], [], [], [], []
        z = float(norm.ppf(1.0 - alpha / 2.0))
        for b in range(n_bins):
            mask = idx == b
            if not mask.any():
                continue
            successes = float(np.sum(self.y_true[mask] == 1))
            total = float(mask.sum())
            p_hat = successes / total
            denom = 1.0 + z * z / total
            centre = (p_hat + z * z / (2.0 * total)) / denom
            half = z * np.sqrt(p_hat * (1.0 - p_hat) / total + z * z / (4.0 * total * total)) / denom
            xs.append(float(np.mean(self.y_prob[mask])))
            ys.append(p_hat)
            counts.append(int(total))
            lower.append(max(0.0, centre - half))
            upper.append(min(1.0, centre + half))

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        brier = float(np.mean((self.y_prob - self.y_true) ** 2))
        fig, ax, _ = single_axis(ax, figsize)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.3, label="Perfect calibration")
        # Clip at zero: a bin with no successes can produce -0.0 (or a tiny
        # negative rounding error) from ``p_hat - lower``, which matplotlib
        # rejects even though it is visually zero.
        ax.errorbar(
            xs, ys,
            yerr=[np.maximum(ys - lower, 0.0), np.maximum(upper - ys, 0.0)],
            fmt="none", ecolor=DEFAULT_COLORS["neutral"], elinewidth=1.0, capsize=2.5,
            alpha=0.8,
        )
        ax.plot(xs, ys, "s-", color="#2563eb", linewidth=2.0, label=f"Model (Brier = {brier:.4f})")
        for x, y, n in zip(xs, ys, counts):
            ax.annotate(
                str(n), (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.5, color=DEFAULT_COLORS["neutral"],
                annotation_clip=False,
            )
        top = 1.06
        if upper.size:
            top = max(top, float(np.max(upper)) + 0.08, float(np.max(ys)) + 0.10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, top)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive proportion")
        ax.legend(loc="lower right", fontsize=8.5)
        style_axis(ax)
        label = title or "Calibration curve"
        ax.set_title(label)
        return PlotResult(fig, label)

    def plot_gains(self, ax=None, *, figsize=(5.6, 4.6), title: str | None = None):
        population, captured = self.gains()
        fig, ax, _ = single_axis(ax, figsize)
        ax.plot(population, captured, color="#7e22ce", linewidth=2.2, label="Model")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.3, label="Random")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Cumulative share of observations")
        ax.set_ylabel("Cumulative share of positives captured")
        ax.legend(loc="lower right", fontsize=8.5)
        style_axis(ax)
        label = title or "Cumulative gains"
        ax.set_title(label)
        return PlotResult(fig, label)

    def plot_lift(self, ax=None, *, figsize=(5.6, 4.6), title: str | None = None):
        population, captured = self.gains()
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(population > 0, captured / population, np.nan)
        fig, ax, _ = single_axis(ax, figsize)
        ax.plot(population, lift, color="#0f766e", linewidth=2.2, label="Lift")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.3, label="Random")
        top = float(np.nanmax(lift)) if np.isfinite(lift).any() else 2.0
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, max(top * 1.05, 1.2))
        ax.set_xlabel("Cumulative share of observations")
        ax.set_ylabel("Lift over chance")
        ax.legend(loc="upper right", fontsize=8.5)
        style_axis(ax)
        label = title or "Lift chart"
        ax.set_title(label)
        return PlotResult(fig, label)

    def plot_threshold_metrics(self, ax=None, *, n_points: int = 60, figsize=(6.2, 4.6), title: str | None = None):
        """Accuracy, precision, recall and F1 as a function of the threshold."""
        thresholds = np.unique(np.quantile(self.y_prob, np.linspace(0.0, 1.0, n_points)))
        accuracy, precision, recall, f1 = [], [], [], []
        for t in thresholds:
            m = self.metrics_summary(float(t))
            accuracy.append(m["accuracy"])
            precision.append(m["precision"])
            recall.append(m["recall_sensitivity"])
            f1.append(m["f1_score"])
        fig, ax, _ = single_axis(ax, figsize)
        ax.plot(thresholds, accuracy, label="Accuracy", linewidth=1.8)
        ax.plot(thresholds, precision, label="Precision", linewidth=1.8)
        ax.plot(thresholds, recall, label="Recall", linewidth=1.8)
        ax.plot(thresholds, f1, label="F1", linewidth=2.2)
        ax.axvline(self.resolve_threshold(), color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.1, label="Default threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="best", fontsize=8.2)
        style_axis(ax)
        label = title or "Metrics by threshold"
        ax.set_title(label)
        return PlotResult(fig, label)

    def plot_all(self, ax=None, *, threshold=None, show_optimal: bool = True, show_ci: bool = True, alpha: float = 0.05, n_bootstraps: int = 1000, seed: int = 42, figsize=(12.0, 9.0), title: str | None = None):
        """A 2×2 panel: confusion matrix, ROC, PR and calibration."""
        fig, axes = panel_axes(ax, 2, 2, figsize=figsize)
        a = axes.ravel()
        self.plot_confusion_matrix(threshold=threshold, ax=a[0])
        self.plot_roc_curve(ax=a[1], show_optimal=show_optimal, show_ci=show_ci, alpha=alpha, n_bootstraps=n_bootstraps, seed=seed)
        self.plot_pr_curve(ax=a[2], show_optimal=show_optimal, show_ci=show_ci, alpha=alpha, n_bootstraps=n_bootstraps, seed=seed)
        self.plot_calibration(ax=a[3])
        label = title or "Classification evaluation"
        fig.suptitle(label, fontsize=14)
        return PlotResult(fig, label)


# ---------------------------------------------------------------------------
# Functional shortcuts
# ---------------------------------------------------------------------------


def roc_curve(result: Any, ax=None, **kwargs):
    """ROC curve for a fitted probabilistic classifier."""
    return ClassificationEvaluator(result).plot_roc_curve(ax=ax, **kwargs)


def precision_recall_curve(result: Any, ax=None, **kwargs):
    """Precision–Recall curve for a fitted probabilistic classifier."""
    return ClassificationEvaluator(result).plot_pr_curve(ax=ax, **kwargs)


def confusion_matrix_plot(result: Any, ax=None, **kwargs):
    """Confusion matrix for a fitted probabilistic classifier."""
    return ClassificationEvaluator(result).plot_confusion_matrix(ax=ax, **kwargs)


def calibration_curve(result: Any, ax=None, *, exog=None, y_true=None, **kwargs):
    """Reliability diagram for a fitted probabilistic classifier."""
    return ClassificationEvaluator(result, exog=exog, y_true=y_true).plot_calibration(ax=ax, **kwargs)


def multiclass_calibration_curve(
    result: Any,
    ax=None,
    *,
    n_bins: int = 10,
    figsize=(6.2, 4.6),
    title: str | None = None,
):
    """One-vs-rest reliability curves for a fitted Multinomial GLM.

    For every category ``k`` the observed indicator ``1(y_i = k)`` is binned by
    its predicted probability ``p_ik``.  A well-calibrated model puts every
    curve on the 45-degree line.  Brier scores in the legend make the relative
    quality of the per-category probabilities explicit.
    """
    probs, one_hot, names, data = _multiclass_arrays(result)
    if type(getattr(data.result, "family", None)).__name__ != "Multinomial":
        raise ValueError(f"multiclass calibration needs a Multinomial fit (got {data.model_name})")

    n_bins = max(2, int(n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    colors = _cmap(probs.shape[1])
    fig, ax, _ = single_axis(ax, figsize)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.3, label="Perfect calibration")

    for k, name in enumerate(names):
        p_k, y_k = probs[:, k], one_hot[:, k]
        idx = np.clip(np.digitize(p_k, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if not mask.any():
                continue
            xs.append(float(np.mean(p_k[mask])))
            ys.append(float(np.mean(y_k[mask])))
        brier = float(np.mean((y_k - p_k) ** 2))
        ax.plot(
            xs, ys, "o-", color=colors[k], linewidth=1.8, markersize=4.5,
            label=f"{name} (Brier {brier:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed class proportion")
    ax.legend(loc="lower right", fontsize=8.0)
    style_axis(ax)
    label = title or f"Multiclass calibration (one-vs-rest) — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)


def gains_chart(result: Any, ax=None, **kwargs):
    """Cumulative gains chart for a fitted probabilistic classifier."""
    return ClassificationEvaluator(result).plot_gains(ax=ax, **kwargs)


def lift_chart(result: Any, ax=None, **kwargs):
    """Lift chart for a fitted probabilistic classifier."""
    return ClassificationEvaluator(result).plot_lift(ax=ax, **kwargs)


def threshold_metrics(result: Any, ax=None, **kwargs):
    """Accuracy/precision/recall/F1 against the decision threshold."""
    return ClassificationEvaluator(result).plot_threshold_metrics(ax=ax, **kwargs)
