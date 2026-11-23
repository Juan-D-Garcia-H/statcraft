# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Minimal smoke suite: imports, packaged data and one fit per engine family."""

import json
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import statcraft
import statcraft.glm as scg
from statcraft.datasets import (
    data_loader,
    dataset_info,
    dataset_registry_path,
    list_datasets,
)
from statcraft.nls import NLS
from statcraft.plots import plot, suggest_plots


def test_public_imports():
    import statcraft.datasets  # noqa: F401
    import statcraft.mcglm  # noqa: F401
    import statcraft.nlme  # noqa: F401
    import statcraft.nls  # noqa: F401
    import statcraft.saem  # noqa: F401

    assert statcraft.__version__
    assert isinstance(statcraft.glm, ModuleType)
    assert not callable(statcraft.glm)
    assert isinstance(statcraft.datasets, ModuleType)
    assert callable(statcraft.list_datasets)
    assert callable(statcraft.dataset_info)
    assert callable(statcraft.data_loader)


def test_packaged_data_loader():
    orth = data_loader("Orthodont.csv").load()
    assert not orth.empty
    assert {"Subject", "age", "Sex", "distance"}.issubset(orth.columns)


def test_data_loader_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    soya = data_loader("soya").load()
    assert not soya.empty


def test_dataset_registry_tracks_changes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "alpha.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    registry = dataset_registry_path(data_dir)

    assert list_datasets(data_dir) == ["alpha.csv"]
    assert registry.exists()
    assert json.loads(registry.read_text(encoding="utf-8"))["dataset_count"] == 1

    (data_dir / "beta.csv").write_text("x,y\n3,4\n", encoding="utf-8")
    assert list_datasets(data_dir) == ["alpha.csv", "beta.csv"]
    assert json.loads(registry.read_text(encoding="utf-8"))["dataset_count"] == 2
    assert list(dataset_info(data_dir)["name"]) == ["alpha.csv", "beta.csv"]

    (data_dir / "beta.csv").unlink()
    assert list_datasets(data_dir) == ["alpha.csv"]
    assert json.loads(registry.read_text(encoding="utf-8"))["dataset_count"] == 1


def test_glm_poisson_fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = rng.poisson(np.exp(0.5 + 0.7 * x))
    df = pd.DataFrame({"x": x, "y": y})

    fit = scg.glm("y ~ x", data=df, family=scg.Poisson()).fit()
    table = fit.coef_table()

    assert len(table) == 2
    assert np.isfinite(np.asarray(table.iloc[:, 0], dtype=float)).all()


def test_tweedie_quantile_residuals_randomise_zeros():
    rng = np.random.default_rng(0)
    n = 160
    x = rng.normal(size=n)
    mu = np.exp(0.3 + 0.5 * x)
    y = rng.binomial(1, 0.25, size=n) * rng.gamma(shape=2.0, scale=mu / 2.0)
    df = pd.DataFrame({"x": x, "y": y})

    fit = scg.glm("y ~ x", data=df, family=scg.Tweedie(var_power=1.5)).fit()
    assert fit.has_quantile_residuals
    r1 = np.asarray(fit.resid_quantile(seed=1), dtype=float)
    r2 = np.asarray(fit.resid_quantile(seed=2), dtype=float)
    zeros = y == 0

    assert r1.size == n and np.isfinite(r1).all() and np.isfinite(r2).all()
    # Tweedie's compound-Poisson point mass is randomised at zero; positive
    # values are continuous and therefore deterministic given the seed.
    assert not np.allclose(r1[zeros], r2[zeros])
    assert np.allclose(r1[~zeros], r2[~zeros])


@pytest.mark.parametrize("family, label", [(scg.Beta, "φ"), (scg.Simplex, "σ²")])
@pytest.mark.parametrize("modelled", [False, True])
def test_beta_simplex_diagnostics_quartet(family, label, modelled):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(1234 + (family is scg.Simplex) * 10 + modelled)
    n = 160
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    mu = expit(-0.1 + 0.6 * x)
    if modelled:
        secondary = (
            np.exp(3.2 + 0.7 * z) if family is scg.Beta else 0.02 + 0.02 * expit(z)
        )
        formula = "y ~ x | z"
    else:
        secondary = np.full(n, 60.0 if family is scg.Beta else 0.04)
        formula = "y ~ x"
    if family is scg.Beta:
        y = rng.beta(mu * secondary, (1.0 - mu) * secondary)
    else:
        # A beta with matched mean/variance keeps the smoke fixture simple;
        # Simplex only needs unit-interval responses with varying dispersion.
        shape = np.maximum(mu * (1.0 - mu) / secondary - 1.0, 5.0)
        y = rng.beta(mu * shape, (1.0 - mu) * shape)
    df = pd.DataFrame({"y": y, "x": x, "z": z})

    fit = scg.glm(formula, data=df, family=family()).fit()
    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes]
    notes = [text.get_text() for text in panel.axes[1].texts]

    assert titles[0].startswith("Observed vs. fitted")
    assert titles[1] == "Q-Q plot — Quantile residuals"
    assert titles[2].startswith("Quantile residuals vs. η")
    assert ("secondary_vs_eta" in suggest_plots(fit)) is modelled
    if modelled:
        assert titles[3].startswith(f"{label} (")
        assert titles[3].endswith("vs. η — GLM (" + family.__name__ + ")")
        assert not notes
    else:
        assert titles[3] == "Cook's distance vs. leverage — GLM (" + family.__name__ + ")"
        assert len(notes) == 1 and notes[0].startswith(f"{label} = ")
    panel.close()


def test_multinomial_diagnostics_quartet():
    pytest.importorskip("matplotlib")
    wines = data_loader("wines.csv").load()
    order = ["Barolo", "Grignolino", "Barbera"]
    wines["wine"] = pd.Categorical(wines["wine"], categories=order, ordered=False)
    fit = scg.glm(
        "wine ~ alcohol + colour - 1", data=wines, family=scg.Multinomial()
    ).fit()

    suggestions = set(suggest_plots(fit))
    assert {
        "category_probs",
        "multiclass_binned_residuals",
        "multiclass_calibration",
    } <= suggestions

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes]
    assert titles[0].startswith("Category discrimination")
    assert titles[1].startswith("Multiclass binned residuals")
    assert titles[2].startswith("Multiclass calibration (one-vs-rest)")
    assert titles[3].startswith("Influence")

    probs = np.asarray(fit.predict_proba(), dtype=float)
    codes = np.rint(np.asarray(fit.endog, dtype=float)).astype(int)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), codes] = 1.0
    # The N×K cell residuals are exactly centred over the whole sample.
    assert abs(float(np.mean(one_hot - probs))) < 1e-10

    binned = next(
        line for line in panel.axes[1].lines
        if line.get_label() == "Bin mean ±2 SE"
    )
    assert np.isfinite(binned.get_ydata()).all()
    calibration_labels = [line.get_label() for line in panel.axes[2].lines]
    assert "Perfect calibration" in calibration_labels
    assert sum("Brier" in label for label in calibration_labels) == probs.shape[1]
    panel.close()


def test_continuous_hurdle_axes_ignore_extreme_response():
    pytest.importorskip("matplotlib")
    med = data_loader("MedExp.csv").load()
    families = [
        scg.HurdleGamma(),
        scg.HurdleLogNormal("identity", zero_link="logit"),
    ]
    for family in families:
        fit = scg.glm(
            "med ~ health + linc + age", data=med, family=family
        ).fit()
        observed_max = float(np.nanmax(np.asarray(fit.endog, dtype=float)))

        panel = plot(fit)
        ax = panel.axes[0]
        assert ax.get_title().startswith("Observed vs. fitted")
        assert ax.get_xscale() == "symlog" and ax.get_yscale() == "symlog"
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert xlim[0] == 0.0 and ylim[0] == 0.0
        # Before the fix the identity line stretched this axis to ~41141,
        # i.e. beyond the largest observed response (39182).
        assert xlim[1] < 0.2 * observed_max
        identity = next(line for line in ax.lines if line.get_label() == "Identity")
        assert float(np.max(identity.get_xdata())) <= xlim[1]
        panel.close()

        standalone = plot(fit, kind="hurdle")
        ax = standalone.axes[0]
        assert ax.get_xscale() == "symlog" and ax.get_yscale() == "symlog"
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert xlim[0] == 0.0 and ylim[0] == 0.0
        assert xlim[1] < 0.2 * observed_max
        identity = next(line for line in ax.lines if line.get_label() == "Identity")
        assert float(np.max(identity.get_xdata())) <= xlim[1]
        standalone.close()


def test_survival_diagnostics_uses_all_rows():
    pytest.importorskip("matplotlib")
    lung = data_loader("lung.csv").load()
    data = lung[["time", "age", "sex", "status"]].copy()
    data["status"] = data["status"].map({1: 0, 2: 1}).astype(int)
    fit = scg.glm("time ~ age + sex", data=data, family=scg.Weibull("identity")).fit(
        cens=data["status"].to_numpy()
    )

    n = len(data)
    assert np.isfinite(np.asarray(fit.resid_deviance_martingale, dtype=float)).sum() == n
    assert "cox_snell" in suggest_plots(fit)
    assert "deviance_martingale_vs_eta" in suggest_plots(fit)

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes]
    assert titles[0].startswith("Observed vs. fitted")
    assert titles[1].startswith("Cox-Snell goodness of fit")
    assert titles[2].startswith("Deviance-martingale residuals vs. η")
    assert titles[3].startswith("Influence")
    # The Nelson-Aalen step starts at the origin and the Greenwood band is drawn.
    step = next(line for line in panel.axes[1].lines if line.get_label() == "Nelson-Aalen H(r)")
    assert step.get_xdata()[0] == 0.0 and step.get_ydata()[0] == 0.0
    assert any(c.get_label() == "95% Greenwood band" for c in panel.axes[1].collections)
    assert len(panel.axes[2].collections) == 2  # events and censored rows
    assert any("censored" in text.get_text() for text in panel.axes[2].texts)

    # The fitted axis must be scaled from fitted values, not stretched to the
    # extreme observed times by the identity reference line.
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    fitted = fitted[np.isfinite(fitted) & (fitted > 0)]
    x0, x1 = panel.axes[0].get_xlim()
    assert np.log10(x1 / x0) < np.log10(fitted.max() / fitted.min()) + 0.5
    identity = next(line for line in panel.axes[0].lines if line.get_label() == "Identity")
    assert identity.get_xdata()[0] >= fitted.min()
    assert identity.get_xdata()[-1] <= fitted.max()
    # Cox-Snell keeps a shared numeric scale without squeezing the axes into a
    # square box inside the rectangular subplot.
    assert panel.axes[1].get_aspect() == "auto"
    panel.close()


def test_aft_families_share_survival_quartet():
    pytest.importorskip("matplotlib")
    lung = data_loader("lung.csv").load()
    data = lung[["time", "age", "sex", "status"]].copy()
    data["status"] = data["status"].map({1: 0, 2: 1}).astype(int)
    families = [
        scg.Weibull("identity"),
        scg.LogNormal(),
        scg.LogLogistic("identity"),
    ]
    expected = [
        "Observed vs. fitted",
        "Cox-Snell goodness of fit",
        "Deviance-martingale residuals vs. η",
        "Influence",
    ]
    for family in families:
        fit = scg.glm("time ~ age + sex", data=data, family=family).fit(
            cens=data["status"].to_numpy()
        )
        panel = plot(fit)
        structure = [ax.get_title().split(" — ")[0] for ax in panel.axes]
        assert structure == expected
        panel.close()


def test_generalized_gamma_has_cox_snell_panel():
    pytest.importorskip("matplotlib")
    lung = data_loader("lung.csv").load()
    data = lung.dropna(subset=["time", "age", "sex", "ph_ecog", "status"]).copy()
    data["status"] = data["status"] - 1
    fit = scg.glm(
        "time ~ age + sex + ph_ecog",
        data=data,
        family=scg.GeneralizedGamma(
            mx_mu_cols=[1, 2, 3],
            mx_sigma_cols=[],
            mx_Q_cols=[],
            link_mu="identity",
            link_sigma="log",
            link_Q="identity",
        ),
    ).fit(cens="status")

    rcs = np.asarray(fit.resid_cox_snell, dtype=float)
    assert rcs.size == len(data) and np.isfinite(rcs).all()
    assert "cox_snell" in suggest_plots(fit)

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes]
    assert titles[1].startswith("Cox-Snell goodness of fit")
    assert titles[2].startswith("Deviance-martingale residuals vs. η")
    assert titles[3].startswith("GeneralizedGamma submodels")
    panel.close()


def test_nls_fit():
    muscle = data_loader("Muscle.csv").load()

    def michaelis_menten(conc, Vm, K):
        return Vm * conc / (K + conc)

    fit = NLS(
        michaelis_menten,
        muscle,
        response="length",
        predictors="conc",
        start={"Vm": 200.0, "K": 0.1},
    ).fit()

    assert len(fit.params) == 2
    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))


def test_truncated_diagnostics_mosaic():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(2026)
    n = 500
    x = rng.normal(size=n)
    lo, hi = -1.0, 4.0
    raw = 1.0 + 0.7 * x + rng.normal(scale=0.9, size=n)
    keep = (raw > lo) & (raw < hi)
    data = pd.DataFrame({"y": raw[keep], "x": x[keep]})
    fit = scg.glm("y ~ x", data=data, family=scg.TruncatedGaussian()).fit(
        left_limit=lo, right_limit=hi
    )

    assert "truncated_diagnostics" in suggest_plots(fit)

    expected_titles = (
        "Q-Q with 95% envelope",
        "Residuals and spread vs. η",
        "Predictive density with truncation",
        "Truncated calibration",
    )
    default = plot(fit)
    default_titles = [ax.get_title() for ax in default.axes[:4]]
    for title, prefix in zip(default_titles, expected_titles):
        assert title.startswith(prefix)
    default.close()

    legacy = plot(fit, kind="truncated")
    assert len(legacy.axes) == 1
    assert legacy.axes[0].get_title().startswith("Truncated fit")
    legacy.close()

    panel = plot(fit, kind="truncated_diagnostics")
    titles = [ax.get_title() for ax in panel.axes[:4]]
    for title, prefix in zip(titles, expected_titles):
        assert title.startswith(prefix)

    density_ax = panel.axes[2]
    assert len(density_ax.patches) <= 20
    curve = next(line for line in density_ax.lines if "predictive" in line.get_label())
    gx, gy = curve.get_data()
    assert 0.85 < float(np.trapezoid(gy, gx)) < 1.15
    limits = sorted(
        line.get_xdata()[0]
        for line in density_ax.lines
        if line.get_label().startswith(("Left limit", "Right limit"))
    )
    assert np.allclose(limits, [lo, hi])
    panel.close()

    plain = scg.glm("y ~ x", data=data, family=scg.Gaussian()).fit()
    with pytest.raises(ValueError, match="continuous truncated GLM"):
        plot(plain, kind="truncated_diagnostics")


def test_mcglm_diagnostics_colours_and_split_covariance():
    pytest.importorskip("matplotlib")
    import statcraft.mcglm as scm

    soya = data_loader("soya.csv").load().astype(
        {"block": "category", "water": "category", "pot": "category"}
    )
    soya["viable_peas"] = soya["viablepeas"] / soya["totalpeas"]
    n = len(soya)
    fit = scm.mcglm(
        [
            "grain ~ block + water * pot",
            "seeds ~ block + water * pot",
            "viable_peas ~ block + water * pot",
        ],
        data=soya,
        z=[scm.mc_id(n)],
        link=["identity", "log", "logit"],
        variance=["constant", "tweedie", "binomialP"],
        ntrial=[None, None, soya["totalpeas"].values],
    ).fit()

    suggestions = suggest_plots(fit)
    assert "standardized_residuals_vs_fitted" in suggestions
    assert "mcglm_residuals_vs_fitted" not in suggestions  # alias, not canonical

    panel = fit.plot()
    residual_ax = panel.axes[2]
    assert residual_ax.get_title().startswith("Standardised residuals vs. μ")
    assert residual_ax.get_xlabel() == "Fitted-value quantile (by response)"
    assert len(residual_ax.collections) == 3
    colours = [tuple(np.round(c.get_facecolor()[0][:3], 3)) for c in residual_ax.collections]
    from matplotlib.colors import to_rgb

    expected = [tuple(np.round(to_rgb(c), 3)) for c in ("#1f4e79", "#b45309", "#0f766e")]
    assert colours == expected

    covariance_ax = panel.axes[3]
    assert covariance_ax.get_title().startswith("Covariance parameters")
    corr_ax, dispersion_ax = covariance_ax.child_axes
    assert corr_ax.get_title() == "Correlations (ρ)"
    assert np.allclose(corr_ax.get_xlim(), (-1.05, 1.05))
    assert [t.get_text() for t in corr_ax.get_yticklabels()] == ["rho_12", "rho_13", "rho_23"]
    assert dispersion_ax.get_title() == "Dispersion / power (τ)"
    assert dispersion_ax.get_xlim()[1] > 1.0
    assert all("tau" in t.get_text() for t in dispersion_ax.get_yticklabels())

    # The stacked panels need real vertical separation: the top x tick labels
    # must not collide with the bottom panel title.
    top_pos, bottom_pos = corr_ax.get_position(), dispersion_ax.get_position()
    gap = top_pos.y0 - (bottom_pos.y0 + bottom_pos.height)
    assert gap / covariance_ax.get_position().height >= 0.12
    assert covariance_ax.title.get_position()[1] >= 1.08
    panel.close()


def test_lme_mosaic_covers_both_levels():
    pytest.importorskip("matplotlib")
    import statcraft.nlme as nl

    orth = data_loader("Orthodont.csv").load()
    fit = nl.lme("distance ~ age * Sex", orth, random="~ age | Subject").fit()
    suggestions = set(suggest_plots(fit))
    assert {
        "population_vs_conditional",
        "normalized_residuals_vs_fitted",
        "qq_plot_with_envelope",
        "caterpillar",
    } <= suggestions

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes]
    assert titles[0].startswith("Population vs. conditional")
    assert titles[1].startswith("Residual trend vs. Conditional fitted value")
    assert titles[2].startswith("Q-Q with 95% envelope")
    assert titles[3].startswith("Random effects: Subject") and "SD =" in titles[3]

    trend = panel.axes[1]
    assert any(line.get_label() == "LOESS trend" for line in trend.lines)
    assert any(c.get_label() == "95% LOESS band" for c in trend.collections)
    assert any(c.get_label() == "95% simulation envelope" for c in panel.axes[2].collections)
    assert any(c.get_label() == "95% conditional CI" for c in panel.axes[3].containers)
    panel.close()


def test_gls_mosaic_uses_grouped_acf_and_dynamic_spread():
    pytest.importorskip("matplotlib")
    import statcraft.nlme as nl

    ovary = data_loader("Ovary.csv").load()
    ovary["sin"] = np.sin(2 * np.pi * ovary["Time"])
    ovary["cos"] = np.cos(2 * np.pi * ovary["Time"])
    fit = nl.gls(
        "follicles ~ sin + cos",
        ovary,
        correlation=nl.corAR1(form="~ 1 | Mare"),
        method="REML",
    ).fit()

    suggestions = set(suggest_plots(fit))
    assert {
        "gls_diagnostics",
        "observed_vs_fitted",
        "qq_plot_with_envelope",
        "normalized_residuals_and_spread",
        "grouped_acf",
    } <= suggestions

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes[:4]]
    assert titles == [
        "Observed vs. fitted — GLS",
        "Normalised residual Q-Q — GLS",
        "Normalised residuals & spread vs. fitted — GLS",
        "Normalised residual ACF — GLS",
    ]
    assert panel.fig._suptitle.get_text() == "GLS diagnostics"
    observed_ax = panel.axes[0]
    assert any(line.get_label() == "LOESS trend" for line in observed_ax.lines)
    observed = observed_ax.collections[0].get_offsets()
    for axis, values in (
        (observed_ax.get_xlim(), observed[:, 0]),
        (observed_ax.get_ylim(), observed[:, 1]),
    ):
        assert axis[0] > 0.0
        assert np.isclose(axis[0], values.min() * 0.95)
        assert np.isclose(axis[1], values.max() * 1.03)
    assert any(c.get_label() == "95% simulation envelope" for c in panel.axes[1].collections)
    assert any(c.get_label() == "Normalised residuals" for c in panel.axes[1].collections)

    trend = panel.axes[2]
    right = panel.axes[-1]
    assert any(c.get_label() == "95% spread band" for c in right.collections)
    residuals = np.asarray(fit.residuals(type="normalized"), dtype=float)
    plotted = trend.collections[0].get_offsets()
    assert np.allclose(np.sort(plotted[:, 1]), np.sort(residuals))
    lower, upper = trend.get_ylim()
    assert lower <= residuals.min() and upper >= residuals.max()
    root_lower, root_upper = right.get_ylim()
    assert root_lower <= 0.0 and root_upper >= np.sqrt(np.abs(residuals)).max()

    acf = panel.axes[3]
    lags = np.asarray(acf.containers[0].markerline.get_xdata(), dtype=float)
    largest_group = ovary.groupby("Mare").size().max()
    assert 0 < int(lags.max()) <= largest_group - 1
    ticks = np.asarray(acf.get_xticks(), dtype=float)
    assert np.allclose(ticks, np.round(ticks))  # lag ticks are integers
    assert any(p.get_label().startswith("95% white-noise band") for p in acf.patches)
    panel.close()

    alias = plot(fit, kind="diagnose_gls")
    assert alias.axes[0].get_title().startswith("Observed vs. fitted")
    alias.close()


def test_gnls_mosaic_uses_structural_variable():
    pytest.importorskip("matplotlib")
    import statcraft.nlme as nl

    soybean = data_loader("Soybean.csv").load()
    fit = nl.gnls(
        "weight ~ SSlogis(Time, Asym, xmid, scal)",
        soybean,
        params="Asym + xmid + scal ~ 1",
    ).fit()

    suggestions = set(suggest_plots(fit))
    assert {
        "gnls_diagnostics",
        "observed_vs_fitted",
        "qq_plot_with_envelope",
        "normalized_residuals_and_spread",
        "grouped_acf",
    } <= suggestions

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes[:4]]
    assert titles == [
        "Observed vs. fitted — GNLS",
        "Q-Q with 95% envelope — GNLS",
        "Normalised residuals & spread vs. Time — GNLS",
        "Normalised residual ACF — GNLS",
    ]
    assert panel.axes[2].get_xlabel() == "Time"
    assert any(line.get_label() == "LOESS trend" for line in panel.axes[0].lines)
    assert panel.axes[2].get_legend()._loc == 2  # upper left
    assert any(c.get_label() == "95% simulation envelope" for c in panel.axes[1].collections)
    assert any(c.get_label() == "95% spread band" for c in panel.axes[-1].collections)

    acf = panel.axes[3]
    lags = np.asarray(acf.containers[0].markerline.get_xdata(), dtype=float)
    largest_plot = soybean.groupby("Plot").size().max()
    assert 0 < int(lags.max()) <= largest_plot - 1
    assert any(p.get_label().endswith("(within group)") for p in acf.patches)
    panel.close()

    alias = plot(fit, kind="diagnose_gnls")
    assert alias.axes[2].get_xlabel() == "Time"
    alias.close()


def test_gnls_normalized_residuals_include_sigma_and_weights():
    pytest.importorskip("matplotlib")
    from scipy.stats import norm

    import statcraft.nlme as nl

    soybean = data_loader("Soybean.csv").load()
    for weights in (nl.varPower(), nl.varExp()):
        fit = nl.gnls(
            "weight ~ SSlogis(Time, Asym, xmid, scal)",
            soybean,
            params="Asym + xmid + scal ~ 1",
            weights=weights,
        ).fit()
        mu = np.asarray(fit.fitted(), dtype=float)
        raw = np.asarray(fit.residuals(type="response"), dtype=float)
        delta = float(np.asarray(fit.theta)[-1])
        factor = np.abs(mu) ** delta if weights.kind == "Power" else np.exp(delta * mu)
        expected = raw / (fit.sigma_r * factor)
        got = np.asarray(fit.residuals(type="normalized"), dtype=float)

        assert np.allclose(got, expected)
        assert abs(float(np.std(got)) - 1.0) < 0.15
        probs = (np.arange(1, got.size + 1) - 0.375) / (got.size + 0.25)
        slope = float(np.polyfit(norm.ppf(probs), np.sort(got), 1)[0])
        assert 0.8 < slope < 1.2


def test_glmer_mosaic_uses_simulated_quantile_residuals():
    pytest.importorskip("matplotlib")
    import statcraft.nlme as nl
    from statcraft.glm import Binomial, Gaussian, NegativeBinomial, Poisson

    bdf = data_loader("bdf.csv").load()
    bdf = bdf.dropna(subset=["repeatgr", "aritPRET", "Minority", "schoolNR"]).copy()
    bdf["repeatgr"] = (bdf["repeatgr"] > 0).astype(int)
    fit = nl.glmer(
        "repeatgr ~ aritPRET + Minority + (1 | schoolNR)",
        bdf,
        family=Binomial("logit"),
    ).fit()

    suggestions = set(suggest_plots(fit))
    assert {
        "glmer_diagnostics",
        "population_vs_conditional",
        "randomized_quantile_qq",
        "binned_quantile_residuals_by_rank",
    } <= suggestions
    assert "qq_residuals" not in suggestions  # raw normal Q-Q is not suggested for discrete GLMER

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes[:4]]
    assert titles == [
        "Population vs. conditional — GLMER",
        "Simulated quantile Q-Q — GLMER",
        "Binned quantile residuals vs. rank(ŷ) — GLMER",
        "Random effects caterpillar — GLMER",
    ]
    assert panel.fig._suptitle.get_text() == "GLMER diagnostics"
    # Legacy callers that pass the redundant suffix get the short title too.
    legacy = plot(fit, kind="glmer_diagnostics", title="GLMER diagnostics — GLMER")
    assert legacy.fig._suptitle.get_text() == "GLMER diagnostics"
    legacy.close()
    assert any(c.get_label() == "95% KS envelope" for c in panel.axes[1].collections)
    assert any(line.get_label() == "Uniform reference" for line in panel.axes[1].lines)
    assert np.allclose(panel.axes[2].get_xlim(), (0.0, 1.0))
    assert len(panel.axes[2].containers) >= 1
    assert any(c.get_label() == "95% conditional CI" for c in panel.axes[3].containers)
    panel.close()

    quantile = np.asarray(fit.resid_quantile(seed=11, n_sim=120), dtype=float)
    assert quantile.size == fit.nobs
    assert np.isfinite(quantile).all()
    assert np.abs(quantile).max() < 6.0

    se_tables = fit.blup_conditional_se()
    assert set(se_tables) == set(fit.ranef)
    for name, se in se_tables.items():
        ranef = fit.ranef[name]
        assert se.index.equals(ranef.index)
        assert [str(c) for c in se.columns] == [str(c) for c in ranef.columns]
        values = se.to_numpy(dtype=float)
        assert np.isfinite(values).all() and np.all(values > 0.0)

    # Grouped binomial k/n uses the same quantile construction.
    cbpp = data_loader("cbpp.csv").load()
    grouped = nl.glmer(
        "cbind(incidence, size - incidence) ~ period + (1 | herd)",
        cbpp,
        family=Binomial(),
    ).fit()
    grouped_panel = plot(grouped, kind="glmer_diagnostics")
    assert grouped_panel.axes[1].get_title().startswith("Simulated quantile Q-Q")
    assert np.allclose(grouped_panel.axes[2].get_xlim(), (0.0, 1.0))
    grouped_panel.close()

    rng = np.random.default_rng(7)
    n = 180
    counts = pd.DataFrame(
        {"x": np.linspace(-1.0, 1.0, n), "g": np.repeat(np.arange(18), 10)}
    )
    eta = -0.3 + 0.8 * counts["x"].to_numpy() + rng.normal(0.0, 0.4, 18)[counts["g"].to_numpy()]
    count_cases = (
        (Poisson(), rng.poisson(np.exp(eta))),
        (NegativeBinomial(theta=3.0), rng.negative_binomial(3.0, 3.0 / (3.0 + np.exp(eta)))),
    )
    for family, response in count_cases:
        count_fit = nl.glmer("y ~ x + (1 | g)", counts.assign(y=response), family=family).fit()
        count_panel = plot(count_fit)
        assert count_panel.axes[1].get_title().startswith("Simulated quantile Q-Q")
        assert count_panel.axes[2].get_title().startswith("Binned quantile residuals")
        count_panel.close()

    # Continuous families keep the prediction/residual/BLUP layout.
    machines = data_loader("Machines.csv").load()
    continuous = nl.glmer("score ~ 1 + (1 | Machine)", machines, family=Gaussian()).fit()
    continuous_panel = plot(continuous, kind="glmer_diagnostics")
    assert [ax.get_title() for ax in continuous_panel.axes[:4]] == [
        "Population vs. conditional — GLMER",
        "Normal Q-Q — GLMER",
        "Binned standardised residuals — GLMER",
        "Random effects caterpillar — GLMER",
    ]
    continuous_panel.close()


def test_nlme_mosaic_uses_time_and_dominant_ranef():
    pytest.importorskip("matplotlib")
    import statcraft.nlme as nl
    from statcraft.plots.effects import dominant_random_effect

    orange = data_loader("Orange.csv").load()
    fit = nl.nlme(
        "circumference ~ SSlogis(age, Asym, xmid, scal)",
        orange,
        params="Asym + xmid + scal ~ 1",
        random="Asym ~ 1 | Tree",
    ).fit()
    suggestions = set(suggest_plots(fit))
    assert {"qq_plot_with_envelope", "normalized_residuals_vs_time"} <= suggestions

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes]
    assert titles[0].startswith("Population vs. conditional")
    assert titles[1].startswith("Q-Q with 95% envelope")
    assert titles[2].startswith("Residual trend vs. age")
    assert titles[3].startswith("Random effects: Tree [Asym]") and "SD =" in titles[3]
    assert panel.axes[2].get_xlabel() == "age"
    assert any(c.get_label() == "95% conditional CI" for c in panel.axes[3].containers)
    panel.close()

    class _Dummy:
        ranef = {
            "group": pd.DataFrame(
                {"stable": [1.0, 2.0, 3.0], "variable": [-10.0, 0.0, 10.0]},
                index=["a", "b", "c"],
            )
        }

    assert dominant_random_effect(_Dummy()) == "variable"


def _saem_linear_fit(censored: bool):
    import statcraft.saem as saem

    times = np.array([0.0, 1.0, 2.0, 3.0])
    rng = np.random.default_rng(7)
    rows = []
    for group in range(8):
        shift = rng.normal(0.0, 0.15)
        y = 0.5 + 1.2 * times + shift + rng.normal(0.0, 0.08, times.size)
        cens = np.where(times == 0.0, -1, 0) if censored else np.zeros(times.size, dtype=int)
        rows.append(pd.DataFrame({
            "id": group + 1,
            "t": times,
            "y": np.where(cens == -1, -2.0, y),
            "cens": cens,
        }))

    def linear(x, p):
        a, b = p
        return a + b * x

    data = pd.concat(rows, ignore_index=True)
    options = {"cens": "cens"} if censored else {}
    model = saem.SAEM(
        linear, data, id="id", time="t", response="y", params=["a", "b"],
        predictors=["t"], error="constant", start_error=0.2,
        start={"a": 0.5, "b": 1.0}, **options,
    )
    return model.fit(n_iter=5, n_burn=2, se_method="none", ll_method="none", seed=1), data


def test_saem_mosaic_uses_censoring_aware_npde():
    pytest.importorskip("matplotlib")
    fit, data = _saem_linear_fit(censored=True)

    suggestions = set(suggest_plots(fit))
    assert {
        "diagnostics", "saem_diagnostics", "grouped_acf",
        "qq_plot_with_envelope", "population_vs_conditional", "npde",
    } <= suggestions

    panel = plot(fit)
    titles = [ax.get_title() for ax in panel.axes[:4]]
    assert titles == [
        "Population vs. conditional (censored limits) — SAEM",
        "NPDE vs. time — SAEM",
        "Q-Q with 95% envelope — SAEM",
        "NPDE ACF — SAEM",
    ]
    assert panel.fig._suptitle.get_text() == "SAEM diagnostics"
    assert len(set(titles)) == 4
    assert not any("Panel unavailable" in text.get_text() for ax in panel.axes for text in ax.texts)

    scatter = panel.axes[1].collections[0].get_offsets()
    assert len(scatter) == len(data)  # censored rows are not dropped
    assert np.isfinite(scatter).all()
    assert any(line.get_label() == "±1.96" for line in panel.axes[1].lines)

    qq_labels = [collection.get_label() for collection in panel.axes[2].collections]
    assert "NPDE" in qq_labels and "95% simulation envelope" in qq_labels

    acf = panel.axes[3]
    lags = np.asarray(acf.containers[0].markerline.get_xdata(), dtype=float)
    assert 0 < int(lags.max()) <= int(data.groupby("id").size().max()) - 1
    assert any(patch.get_label().endswith("(within group)") for patch in acf.patches)

    legacy = plot(fit, kind="saem_diagnostics", title="SAEM diagnostics — SAEM")
    assert legacy.fig._suptitle.get_text() == "SAEM diagnostics"
    legacy.close()
    panel.close()


def test_saem_mosaic_uses_pearson_and_within_subject_acf():
    pytest.importorskip("matplotlib")
    fit, data = _saem_linear_fit(censored=False)

    panel = plot(fit, kind="diagnose_saem")
    assert [ax.get_title() for ax in panel.axes[:4]] == [
        "Population vs. conditional — SAEM",
        "Pearson residuals vs. IPRED — SAEM",
        "Q-Q with 95% envelope — SAEM",
        "Residual ACF — SAEM",
    ]
    assert panel.axes[1].get_xlabel() == "Conditional prediction (IPRED)"
    assert any(line.get_label() == "±2" for line in panel.axes[1].lines)
    qq_labels = [collection.get_label() for collection in panel.axes[2].collections]
    assert "Pearson residuals" in qq_labels
    acf = panel.axes[3]
    lags = np.asarray(acf.containers[0].markerline.get_xdata(), dtype=float)
    assert 0 < int(lags.max()) <= int(data.groupby("id").size().max()) - 1
    panel.close()
