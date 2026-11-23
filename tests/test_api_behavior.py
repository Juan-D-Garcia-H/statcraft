# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Behavioral checks for univariate callable predictors and compact fit reprs."""

from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import statcraft as sf
import statcraft.glm as scg
import statcraft.nlme as scnl
import statcraft.saem as scsaem
from statcraft.mcglm.mcglm import MCGLMResults
from statcraft.nlme import GNLS, NLME
from statcraft.nls import NLS
from statcraft.saem.fit import SaemFit
from statcraft.utils.modelview import ModelView


def _linear_frame():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    rows = []
    for group in range(3):
        rows.append(pd.DataFrame({
            "id": group + 1,
            "group": f"g{group}",
            "t": times,
            "y": 0.5 + 1.2 * times + np.array([0.0, 0.1, -0.1, 0.0]),
        }))
    return pd.concat(rows, ignore_index=True)


def _linear(x, p):
    a, b = p
    return a + b * x


def _assert_compact_repr(fit):
    text = repr(fit)
    assert "\n" not in text
    assert "Std. Error" not in text
    assert "Dep. Variable" not in text
    assert str(fit) == text
    assert not hasattr(fit, "_repr_html_")
    assert "summary" not in text.lower()


def test_single_predictor_is_available_directly():
    view = ModelView(["t"], [np.array([1.0, 2.0, 3.0])], label="x", columnar=True)
    assert np.asarray(view).shape == (3,)
    assert np.allclose(2.0 * view, [2.0, 4.0, 6.0])
    time, = view
    assert np.allclose(time, [1.0, 2.0, 3.0])

    df = _linear_frame()
    nls_fit = NLS(_linear, df, response="y", predictors="t",
                  start={"a": 0.0, "b": 1.0}).fit()
    assert np.allclose(np.asarray(nls_fit.params, dtype=float), [0.5, 1.2], atol=0.1)

    nlme_fit = NLME(
        _linear, df, params={"a": "1", "b": "1"}, predictors="t", response="y",
        random="a ~ 1 | group", start={"a": 0.5, "b": 1.2},
    ).fit()
    assert np.all(np.isfinite(np.asarray(nlme_fit.fixef, dtype=float)))

    saem_fit = sf.SAEM(
        _linear, df, id="id", time="t", response="y", params=["a", "b"],
        predictors=["t"], omega="diagonal", start={"a": 0.5, "b": 1.0},
    ).fit(n_iter=3, n_burn=1, se_method="none", ll_method="none", seed=1)
    assert np.all(np.isfinite(np.asarray(saem_fit.fixef(), dtype=float)))
    assert saem_fit.model.is_compiled


def test_single_parameter_view_behaves_as_its_value():
    view = ModelView(["a"], [np.array([1.5])], label="p")
    value = view
    assert np.allclose(np.asarray(value), [1.5])
    assert np.allclose(2.0 * value, [3.0])
    a, = view
    assert np.isclose(a, 1.5)

    times = np.array([0.0, 1.0, 2.0, 3.0])
    rows = []
    for group in range(3):
        rows.append(pd.DataFrame({
            "t": times,
            "y": 1.2 * times + np.array([0.0, 0.1, -0.1, 0.0]),
            "group": f"g{group}",
        }))
    df = pd.concat(rows, ignore_index=True)

    def scaled(x, params):
        time = x
        a = params
        return a * time

    nls_fit = NLS(scaled, df, response="y", predictors="t", start={"a": 1.0}).fit()
    assert np.isclose(float(nls_fit.params[0]), 1.2, atol=0.1)

    nlme_fit = NLME(
        scaled, df, predictors="t", response="y",
        random="a ~ 1 | group", start={"a": 1.0},
    ).fit()
    assert "scaled(t, a)" in nlme_fit.formula
    assert list(nlme_fit.fixef.index) == ["a"]
    assert np.all(np.isfinite(np.asarray(nlme_fit.fixef, dtype=float)))

    gnls_fit = GNLS(scaled, df, predictors="t", response="y", start={"a": 1.0}).fit()
    assert list(gnls_fit.fixef.index) == ["a"]

    def shifted(x, params):
        time = x
        a = params
        return 1.0 + a * time

    shifted_df = df.assign(y=1.0 + df["y"])
    saem_fit = sf.SAEM(
        shifted, shifted_df, id="group", time="t", response="y",
        params=["a"], predictors=["t"], omega="diagonal", start={"a": 1.0},
    ).fit(n_iter=3, n_burn=1, se_method="none", ll_method="none", seed=1)
    assert saem_fit.model.is_compiled
    assert np.all(np.isfinite(np.asarray(saem_fit.fixef(), dtype=float)))


def test_single_parameter_named_form_is_not_a_vector_model():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    df = pd.DataFrame({"t": times, "y": 1.2 * times + [0.0, 0.1, -0.1, 0.0]})

    def named(time, a):
        return a * time

    fit = GNLS(named, df, predictors="t", response="y", start={"a": 1.0}).fit()
    assert "named(time, a)" in fit.formula
    assert list(fit.fixef.index) == ["a"]


def test_nlme_and_gnls_infer_parameter_order_from_start():
    df = _linear_frame()

    def ordered_model(x, p):
        slope, intercept = p
        return intercept + slope * x

    start = {"slope": 1.2, "intercept": 0.5}
    nlme_fit = NLME(
        ordered_model, df, predictors="t", response="y",
        random="intercept ~ 1 | group", start=start,
    ).fit()
    assert list(nlme_fit.fixef.index) == ["slope", "intercept"]
    assert np.all(np.isfinite(np.asarray(nlme_fit.fixef, dtype=float)))

    gnls_fit = GNLS(
        ordered_model, df, predictors="t", response="y", start=start,
    ).fit()
    assert list(gnls_fit.fixef.index) == ["slope", "intercept"]
    assert np.all(np.isfinite(np.asarray(gnls_fit.fixef, dtype=float)))

    with pytest.raises(TypeError, match="params is required"):
        NLME(
            ordered_model, df, predictors="t", response="y",
            random="intercept ~ 1 | group", start=[1.2, 0.5],
        ).fit()


def test_vector_models_accept_arbitrary_argument_names():
    df = _linear_frame()

    def growth(values, coefs):
        a, b = coefs
        return a * (1.0 - np.exp(-b * values))

    nlme_fit = NLME(
        growth, df, params={"a": "t", "b": "1"}, predictors="t", response="y",
        random="a ~ 1 | group", start={"a": 4.0, "b": 1.0},
    ).fit()
    assert "growth(t, a, b)" in nlme_fit.formula
    assert np.all(np.isfinite(np.asarray(nlme_fit.fixef, dtype=float)))

    def saem_linear(values, coefs):
        a, b = coefs
        return a + b * values

    saem_fit = sf.SAEM(
        saem_linear, df, id="id", time="t", response="y",
        params=["a", "b"], predictors=["t"], omega="diagonal",
        start={"a": 0.5, "b": 1.0},
    ).fit(n_iter=3, n_burn=1, se_method="none", ll_method="none", seed=1)
    assert saem_fit.model.is_compiled
    assert np.all(np.isfinite(np.asarray(saem_fit.fixef(), dtype=float)))


def test_saem_npde_uses_the_documented_censoring_direction(monkeypatch):
    fit = SaemFit.__new__(SaemFit)
    fit.res = SimpleNamespace(
        data=SimpleNamespace(
            y=np.array([1.0, 2.0, 3.0]),
            cens=np.array([-1, 0, 1]),
        ),
    )
    fit.info = SimpleNamespace(row_index=np.arange(3))
    simulations = np.array([
        [0.5, 1.5],  # left-censored limit: F(limit) = 0.5
        [1.0, 3.0],  # observed value: F(y) = 0.5
        [2.0, 4.0],  # right-censored limit: F(limit) = 0.5
    ])
    monkeypatch.setattr(fit, "_simulated_y", lambda **kwargs: simulations)

    result = fit.npde(nsim=2, seed=7).to_numpy()

    assert result[0] < 0.0
    assert abs(result[1]) < 1e-12
    assert result[2] > 0.0


def test_printing_a_fit_does_not_render_the_summary():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    glm_fit = scg.glm("y ~ x", data=pd.DataFrame({"x": x, "y": rng.poisson(np.exp(0.4 + 0.6 * x))}),
                      family=scg.Poisson()).fit()
    _assert_compact_repr(glm_fit)

    df = _linear_frame()
    nls_fit = NLS(_linear, df, response="y", predictors="t",
                  start={"a": 0.0, "b": 1.0}).fit()
    nlme_fit = NLME(
        _linear, df, params={"a": "1", "b": "1"}, predictors="t", response="y",
        random="a ~ 1 | group", start={"a": 0.5, "b": 1.2},
    ).fit()
    saem_fit = sf.SAEM(
        _linear, df, id="id", time="t", response="y", params=["a", "b"],
        predictors=["t"], omega="diagonal", start={"a": 0.5, "b": 1.0},
    ).fit(n_iter=3, n_burn=1, se_method="none", ll_method="none", seed=1)

    for fit in (nls_fit, nlme_fit, saem_fit):
        _assert_compact_repr(fit)

    assert "_repr_html_" not in MCGLMResults.__dict__
    assert "__str__" not in MCGLMResults.__dict__


def test_family_classes_are_reexported_by_nlme_and_saem():
    assert scnl.Binomial is scg.Binomial
    assert scsaem.Binomial is scg.Binomial
    assert scsaem.Family is scg.Family
    for name in ("Gaussian", "Poisson", "NegativeBinomial", "Multinomial", "Ordinal",
                 "ZeroInflatedPoisson", "Gamma", "TruncatedGaussian"):
        assert getattr(scnl, name) is getattr(scg, name)
        assert getattr(scsaem, name) is getattr(scg, name)
        assert name in scsaem.__all__


def test_top_level_exposes_every_public_engine_name():
    for module_name in ("glm", "nlme", "mcglm", "nls", "plots", "datasets", "summary", "saem", "utils"):
        module = getattr(sf, module_name)
        assert isinstance(module, ModuleType)
        for name in module.__all__:
            if name == module_name:
                continue
            value = getattr(module, name)
            if isinstance(value, ModuleType):
                continue
            assert hasattr(sf, name), f"statcraft.{name} is missing"
            assert getattr(sf, name) is value
            assert name in sf.__all__

    assert sf.Binomial is scg.Binomial
    assert sf.poisson is scg.poisson
    assert sf.varIdent is scnl.varIdent
    assert sf.NLS is NLS


@pytest.mark.parametrize("family, expected, extra", [
    (scsaem.Binomial("logit"), "bernoulli_logit", 0),
    (scsaem.Binomial, "bernoulli_logit", 0),
    (scsaem.Binomial("probit"), "bernoulli", 0),
    (scsaem.Poisson("log"), "poisson", 0),
    (scsaem.NegativeBinomial(theta=2.0), "negbin", 1),
    (scsaem.ZeroInflatedPoisson(), "zip", 0),
    (scsaem.Multinomial(), "categorical", 0),
    (scsaem.Ordinal(), "ordinal", 0),
    (scsaem.Weibull(), "tte", 0),
])
def test_saem_accepts_family_classes_only(family, expected, extra):
    model = sf.SAEM(
        _linear, _linear_frame(), id="id", time="t", response="y",
        params=["a", "b"], predictors=["t"], family=family,
        cuts=lambda p: [0.0, 1.0], cumhazard=lambda a, b, t: t,
    )
    assert model.model.family == expected
    assert model.model.n_extra == extra


def test_saem_accepts_custom_family_class_with_loglik():
    model = scsaem.NLModel(
        fn=None, params=["a", "b"], family=scsaem.CustomFamily,
        loglik=lambda y, a, b, t: 0.0,
    )
    assert model.family == "custom"


@pytest.mark.parametrize("family", [
    "gaussian", "bernoulli_logit", "binomial", "poisson", "negbin", "zip",
    "ordinal", "categorical", "tte", "custom", "negative_binomial",
])
def test_saem_rejects_every_family_string(family):
    with pytest.raises(ValueError, match="no longer accepted"):
        sf.SAEM(
            _linear, _linear_frame(), id="id", time="t", response="y",
            params=["a", "b"], predictors=["t"], family=family,
        )


@pytest.mark.parametrize("family, message", [
    (scsaem.Gaussian("log"), "mean scale"),
    (scsaem.Ordinal("probit"), "logit"),
    (scsaem.Gamma(), "not supported by SAEM"),
    (scsaem.Weibull(), "require cumhazard"),
    (scsaem.CustomFamily, "requires loglik"),
])
def test_saem_rejects_unavailable_family_objects(family, message):
    with pytest.raises(ValueError, match=message):
        sf.SAEM(
            _linear, _linear_frame(), id="id", time="t", response="y",
            params=["a", "b"], predictors=["t"], family=family,
        )


def test_saem_fit_uses_a_glm_family_object():
    rng = np.random.default_rng(11)
    times = np.array([0.0, 1.0, 2.0, 3.0])
    rows = []
    for group in range(8):
        eta = -0.5 + 0.4 * times + rng.normal(0.0, 0.2)
        prob = 1.0 / (1.0 + np.exp(-eta))
        rows.append(pd.DataFrame({
            "id": group + 1,
            "t": times,
            "y": rng.binomial(1, prob),
        }))

    def logistic(x, p):
        a, b = p
        return a + b * x

    fit = sf.SAEM(
        logistic, pd.concat(rows, ignore_index=True), id="id", time="t", response="y",
        params=["a", "b"], predictors=["t"], random=["a"],
        family=scsaem.Binomial("logit"), start={"a": -0.5, "b": 0.4},
    ).fit(n_iter=3, n_burn=1, se_method="none", ll_method="none", seed=1)

    assert fit.model.family == "bernoulli_logit"
    assert list(fit.fixef().index) == ["a", "b"]
