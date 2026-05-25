# -*- coding: utf-8 -*-
"""
dashboard.py – Streamlit Dashboard
Prognozowanie i Symulacje: Zużycie energii elektrycznej w Polsce
Uruchomienie: streamlit run dashboard.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    import pmdarima as pm
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Energia Elektryczna – PiS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

BLUE   = "#1a5c96"; RED    = "#c0392b"; GREEN  = "#27ae60"
ORANGE = "#e67e22"; GRAY   = "#7f8c8d"
PALETTE = list(plt.cm.tab20.colors[:16])

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "font.family": "DejaVu Sans",
})

# ── HELPER: show PNG ──────────────────────────────────────────
def show_png(fname, caption=""):
    path = os.path.join(SCRIPT_DIR, fname)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Brak pliku: **{fname}** – uruchom najpierw odpowiedni skrypt.")

# ── DATA LOADING ──────────────────────────────────────────────
@st.cache_data
def load_polska():
    df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_polska.xlsx"))
    df = df.sort_values("rok").reset_index(drop=True)
    df["pkb_per_capita"] = df["pkb_mln_zl"] * 1e6 / df["ludnosc"]
    df["ln_pkb_pc"]      = np.log(df["pkb_per_capita"])
    df["ln_zuzycie"]     = np.log(df["zuzycie_energii_GWh"])
    df["ln_cena"]        = np.log(df["cena_energii_zl_kWh"])
    return df

@st.cache_data
def load_woj():
    df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_wojewodztwa.xlsx"))
    df = df.sort_values(["wojewodztwo", "rok"]).reset_index(drop=True)
    zero_mask = df["dochod_os"] <= 0
    if zero_mask.any():
        df.loc[zero_mask, "dochod_os"] = np.nan
        df["dochod_os"] = df.groupby("wojewodztwo")["dochod_os"].transform(
            lambda s: s.interpolate(method="linear", limit=3, limit_direction="both"))
    df["ln_dochod_os"]      = np.log(df["dochod_os"].where(df["dochod_os"] > 0))
    df["ln_dochod_os_lag1"] = df.groupby("wojewodztwo")["ln_dochod_os"].shift(1)
    df["ln_zuzycie"]        = np.log(df["zuzycie_energii_GWh"].where(df["zuzycie_energii_GWh"] > 0))
    df["ln_cena"]           = np.log(df["cena_energii_zl_kWh"].where(df["cena_energii_zl_kWh"] > 0))
    return df

# ── MODEL FITTING ─────────────────────────────────────────────
@st.cache_data
def fit_polska():
    df    = load_polska()
    MC    = ["ln_pkb_pc", "ln_cena", "hdd"]
    df_tr = df[df["rok"] <= 2022]
    model_tr   = sm.OLS(df_tr["ln_zuzycie"], sm.add_constant(df_tr[MC])).fit()
    model_full = sm.OLS(df["ln_zuzycie"],    sm.add_constant(df[MC])).fit()
    X_vif = sm.add_constant(df[MC])
    vif_df = pd.DataFrame({
        "Zmienna": MC,
        "VIF":     [variance_inflation_factor(X_vif.values, i+1) for i in range(len(MC))]
    })
    resid  = model_full.resid
    dw_v   = durbin_watson(resid)
    sw_p   = shapiro(resid)[1]
    bg_p   = acorr_breusch_godfrey(model_full, nlags=2)[1]
    bp_p   = het_breuschpagan(resid, sm.add_constant(df[MC]))[1]
    diag   = {"DW": round(dw_v,4), "SW_p": round(sw_p,4), "BG_p": round(bg_p,4), "BP_p": round(bp_p,4)}
    return model_tr, model_full, df, df_tr, MC, vif_df, diag

@st.cache_data
def fit_woj():
    df    = load_woj()
    XC    = ["ln_dochod_os_lag1","ln_cena","urbanizacja_pct","liczba_os","pow_os","hdd"]
    dm    = df.dropna(subset=XC + ["ln_zuzycie"])
    dm    = dm[np.isfinite(dm[XC + ["ln_zuzycie"]]).all(axis=1)].copy()
    df_tr = dm[dm["rok"] <= 2022]
    model_tr   = sm.OLS(df_tr["ln_zuzycie"], sm.add_constant(df_tr[XC])).fit()
    model_full = sm.OLS(dm["ln_zuzycie"],    sm.add_constant(dm[XC])).fit()
    X_vif = sm.add_constant(df_tr[XC])
    vif_df = pd.DataFrame({
        "Zmienna": XC,
        "VIF":     [variance_inflation_factor(X_vif.values, i+1) for i in range(len(XC))]
    })
    resid = model_tr.resid
    diag  = {"DW": round(durbin_watson(resid),4), "SW_p": round(shapiro(resid)[1],4)}
    return model_tr, model_full, df, dm, XC, vif_df, diag

@st.cache_data
def fit_prov_models():
    df = load_woj()
    XC = ["ln_dochod_os_lag1","ln_cena","urbanizacja_pct","liczba_os","pow_os","hdd"]
    dm = df.dropna(subset=XC + ["ln_zuzycie"])
    dm = dm[np.isfinite(dm[XC + ["ln_zuzycie"]]).all(axis=1)].copy()
    PROV = sorted(df["wojewodztwo"].unique())
    rows = []; models = {}
    for prov in PROV:
        dp = dm[dm["wojewodztwo"] == prov]
        if len(dp) < 8: continue
        mdl = sm.OLS(dp["ln_zuzycie"], sm.add_constant(dp[XC])).fit()
        models[prov] = mdl
        rows.append({
            "Województwo": prov,
            "R²":         round(mdl.rsquared, 4),
            "R²_adj":     round(mdl.rsquared_adj, 4),
            "AIC":        round(mdl.aic, 2),
            "β_dochod":   round(mdl.params.get("ln_dochod_os_lag1", np.nan), 3),
            "β_cena":     round(mdl.params.get("ln_cena", np.nan), 3),
            "β_urban":    round(mdl.params.get("urbanizacja_pct", np.nan), 4),
            "β_hdd":      round(mdl.params.get("hdd", np.nan), 6),
            "DW":         round(durbin_watson(mdl.resid), 3),
            "SW_p":       round(shapiro(mdl.resid)[1], 3),
        })
    return pd.DataFrame(rows), PROV, models

# ── Z5 POLSKA (pełne obliczenia, cache) ───────────────────────
@st.cache_data
def compute_z5_polska():
    df       = load_polska()
    TRAIN    = 2022; TEST = [2023, 2024]; FC = 2025
    MC       = ["ln_pkb_pc", "ln_cena", "hdd"]
    df_tr    = df[df["rok"] <= TRAIN]
    df_te    = df[df["rok"].isin(TEST)]
    YEARS    = df["rok"].values

    model_tr   = sm.OLS(df_tr["ln_zuzycie"], sm.add_constant(df_tr[MC])).fit()
    model_full = sm.OLS(df["ln_zuzycie"],    sm.add_constant(df[MC])).fit()

    def _X1(t):
        t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
        return np.column_stack([np.ones(len(t)), t])
    def _X2(t):
        t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
        return np.column_stack([np.ones(len(t)), t, t**2])

    def best_forecast(y_all, n_exante=1):
        mask_tr = YEARS <= TRAIN; mask_te = YEARS > TRAIN
        y_tr = y_all[mask_tr]; y_te = y_all[mask_te]
        n_tr = len(y_tr); n_te_ = len(y_te)
        t_tr = np.arange(1, n_tr+1, dtype=float)
        t_te = np.arange(n_tr+1, n_tr+n_te_+1, dtype=float)
        t_fc = np.arange(n_tr+n_te_+1, n_tr+n_te_+n_exante+1, dtype=float)
        cands = {}
        try:
            m = sm.OLS(y_tr, _X1(t_tr)).fit()
            pt = np.asarray(m.predict(_X1(t_te))).ravel()
            cands["OLS_lin"] = (pt, np.asarray(m.predict(_X1(t_fc))).ravel())
        except: pass
        try:
            m = sm.OLS(y_tr, _X2(t_tr)).fit()
            pt = np.asarray(m.predict(_X2(t_te))).ravel()
            cands["OLS_kw"] = (pt, np.asarray(m.predict(_X2(t_fc))).ravel())
        except: pass
        try:
            m = AutoReg(y_tr, lags=1, old_names=False).fit()
            pt = np.asarray(m.predict(start=n_tr, end=n_tr+n_te_-1)).ravel()
            pf = np.asarray(m.predict(start=n_tr+n_te_, end=n_tr+n_te_+n_exante-1)).ravel()
            cands["AR(1)"] = (pt, pf)
        except: pass
        try:
            m = AutoReg(y_tr, lags=2, old_names=False).fit()
            pt = np.asarray(m.predict(start=n_tr, end=n_tr+n_te_-1)).ravel()
            pf = np.asarray(m.predict(start=n_tr+n_te_, end=n_tr+n_te_+n_exante-1)).ravel()
            cands["AR(2)"] = (pt, pf)
        except: pass
        if PMDARIMA_OK:
            try:
                m = pm.auto_arima(y_tr, seasonal=False, suppress_warnings=True, stepwise=True)
                pred_all = m.predict(n_periods=n_te_+n_exante)
                cands["ARIMA"] = (pred_all[:n_te_], pred_all[n_te_:])
            except: pass
        try:
            m = ExponentialSmoothing(y_tr, trend="add", seasonal=None).fit(optimized=True)
            fc_all = np.asarray(m.forecast(n_te_+n_exante)).ravel()
            cands["Holt"] = (fc_all[:n_te_], fc_all[n_te_:])
        except: pass
        if not cands:
            return None, None, None, np.nan
        best_name = min(cands, key=lambda k:
            100*np.sqrt(np.mean((cands[k][0]-y_te)**2))/np.mean(y_te) if np.mean(y_te)!=0 else np.inf)
        rmspe = 100*np.sqrt(np.mean((cands[best_name][0]-y_te)**2))/np.mean(y_te)
        return best_name, cands[best_name][0], cands[best_name][1], abs(rmspe)

    VAR_CONFIG = [
        {"col": "pkb_per_capita",      "transform": "log",  "mc": "ln_pkb_pc"},
        {"col": "cena_energii_zl_kWh", "transform": "log",  "mc": "ln_cena"},
        {"col": "hdd",                 "transform": None,   "mc": "hdd"},
    ]
    x_test = {}; x_fc = {}; x_info = {}
    for cfg in VAR_CONFIG:
        y_x = df[cfg["col"]].values.astype(float)
        bname, pt, pf_arr, rmspe_v = best_forecast(y_x)
        if bname is None:
            x_test[cfg["mc"]] = np.full(2, np.nan); x_fc[cfg["mc"]] = np.nan; continue
        if cfg["transform"] == "log":
            x_test[cfg["mc"]] = np.log(np.clip(pt, 1e-9, None))
            x_fc[cfg["mc"]]   = float(np.log(max(pf_arr[0], 1e-9)))
        else:
            x_test[cfg["mc"]] = pt
            x_fc[cfg["mc"]]   = float(pf_arr[0])
        x_info[cfg["col"]] = {"method": bname, "rmspe": rmspe_v}

    n_te = 2
    X_test = pd.DataFrame({"const": np.ones(n_te), **{c: x_test[c] for c in MC}})
    y_cond = np.exp(np.asarray(model_tr.predict(X_test)).ravel())
    X_act  = pd.DataFrame({"const": np.ones(n_te), **{c: df_te[c].values for c in MC}})
    y_act  = np.exp(np.asarray(model_tr.predict(X_act)).ravel())
    y_naive  = np.full(n_te, float(df_tr["zuzycie_energii_GWh"].iloc[-1]))
    y_actual = df_te["zuzycie_energii_GWh"].values.astype(float)

    X_fc25 = pd.DataFrame({"const": [1.0], **{c: [x_fc[c]] for c in MC}})
    y_fc25 = np.exp(float(model_full.predict(X_fc25).values[0]))
    pframe = model_full.get_prediction(X_fc25).summary_frame(alpha=0.05)
    y_lo   = np.exp(pframe["mean_ci_lower"].values[0])
    y_hi   = np.exp(pframe["mean_ci_upper"].values[0])

    def em(act, pred):
        act = np.array(act, dtype=float); pred = np.array(pred, dtype=float)
        if not (np.all(np.isfinite(act)) and np.all(np.isfinite(pred))):
            return {k: np.nan for k in ["ME","MPE%","MAE","MAPE%","RMSE","RMSPE%","Theil_U"]}
        me   = np.mean(pred-act); mpe = 100*np.mean((pred-act)/act)
        mae  = np.mean(np.abs(pred-act)); mape = 100*np.mean(np.abs((pred-act)/act))
        rmse = np.sqrt(np.mean((pred-act)**2))
        rmspe = 100*rmse/np.mean(act)
        theil = np.sqrt(np.mean((pred[1:]-act[1:])**2)/np.mean(act[1:]**2)) if len(act)>1 else np.nan
        return {"ME":round(me,1),"MPE%":round(mpe,2),"MAE":round(mae,1),
                "MAPE%":round(mape,2),"RMSE":round(rmse,1),"RMSPE%":round(rmspe,2),
                "Theil_U":round(theil,4) if not np.isnan(theil) else np.nan}

    return {
        "df": df, "df_tr": df_tr, "df_te": df_te,
        "y_cond": y_cond, "y_act": y_act, "y_naive": y_naive, "y_actual": y_actual,
        "y_fc25": y_fc25, "y_lo": y_lo, "y_hi": y_hi,
        "m_cond": em(y_actual, y_cond),
        "m_act":  em(y_actual, y_act),
        "m_naive":em(y_actual, y_naive),
        "x_info": x_info,
        "TEST": TEST, "FC": FC,
    }

# ── HELPER: format model coefficients ─────────────────────────
def model_coef_df(model):
    rows = []
    for nm, co, se, tv, pv in zip(
        model.params.index, model.params, model.bse, model.tvalues, model.pvalues
    ):
        sig = "***" if pv<0.01 else ("**" if pv<0.05 else ("*" if pv<0.1 else ""))
        rows.append({"Zmienna": nm, "Współczynnik": round(co,5),
                     "Std.błąd": round(se,5), "t-stat": round(tv,3),
                     "p-wartość": round(pv,4), "Istotność": sig})
    return pd.DataFrame(rows)

# ── TITLE ─────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:0'>⚡ Zużycie energii elektrycznej w Polsce</h1>
<p style='color:gray;margin-top:4px'>Prognozowanie i Symulacje &nbsp;|&nbsp; Dashboard analityczny &nbsp;|&nbsp; Dane: 2004–2024</p>
""", unsafe_allow_html=True)
st.divider()

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dane",
    "🇵🇱 Model Polska",
    "🗺️ Model Województwa",
    "📈 Prognozy Z4",
    "🎯 Prognoza Warunkowa Z5",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 – DANE
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("Dane źródłowe i statystyki opisowe")
    scope = st.radio("Zakres danych", ["Polska (szereg czasowy)", "Województwa (dane panelowe)"],
                     horizontal=True)

    if scope.startswith("Polska"):
        df_p = load_polska()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Okres", f"{int(df_p['rok'].min())}–{int(df_p['rok'].max())}")
        c2.metric("Obserwacji", len(df_p))
        c3.metric("Min. zużycie", f"{df_p['zuzycie_energii_GWh'].min():,.0f} GWh")
        c4.metric("Max. zużycie", f"{df_p['zuzycie_energii_GWh'].max():,.0f} GWh")

        st.subheader("Dane roczne")
        st.dataframe(df_p.drop(columns=["ln_pkb_pc","ln_zuzycie","ln_cena"], errors="ignore")
                     .round(3), use_container_width=True, height=300)

        st.subheader("Statystyki opisowe")
        sc = ["zuzycie_energii_GWh","cena_energii_zl_kWh","pkb_per_capita","hdd","cdd","ludnosc"]
        sc = [c for c in sc if c in df_p.columns]
        st.dataframe(df_p[sc].describe().round(3), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Szereg czasowy – zużycie energii")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(df_p["rok"], df_p["zuzycie_energii_GWh"],
                    "o-", color=BLUE, lw=2.2, ms=7)
            ax.fill_between(df_p["rok"], df_p["zuzycie_energii_GWh"],
                            alpha=0.08, color=BLUE)
            ax.set_xlabel("Rok"); ax.set_ylabel("GWh")
            ax.set_title("Zużycie energii – Polska 2004–2024", fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
            ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with c2:
            st.subheader("Macierz korelacji")
            corr_cols = [c for c in ["zuzycie_energii_GWh","pkb_per_capita",
                                      "cena_energii_zl_kWh","hdd","cdd","ludnosc"]
                         if c in df_p.columns]
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(df_p[corr_cols].corr(), annot=True, fmt=".2f",
                        cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                        linewidths=0.4, annot_kws={"size": 8})
            ax.set_title("Korelacje Pearsona – Polska", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with st.expander("Wykresy z analizy (wygenerowane pliki)"):
            c1, c2 = st.columns(2)
            with c1: show_png("ep01_szeregi_czasowe.png", "Szereg czasowy")
            with c2: show_png("ep02_korelacja.png", "Macierz korelacji")
            show_png("ep03_scatter.png", "Wykresy rozrzutu")

    else:  # Województwa
        df_w = load_woj()
        PROV = sorted(df_w["wojewodztwo"].unique())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Województwa", 16)
        c2.metric("Lata", f"{int(df_w['rok'].min())}–{int(df_w['rok'].max())}")
        c3.metric("Obserwacji (panel)", len(df_w))
        total_2024 = df_w[df_w["rok"]==2024]["zuzycie_energii_GWh"].sum()
        c4.metric("Łączne zużycie 2024", f"{total_2024:,.0f} GWh")

        prov_sel = st.selectbox("Wybierz województwo:", PROV)
        dp = df_w[df_w["wojewodztwo"] == prov_sel].sort_values("rok")

        c1, c2 = st.columns([3, 2])
        with c1:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(dp["rok"], dp["zuzycie_energii_GWh"],
                         "o-", color=BLUE, lw=2, ms=6)
            axes[0].set_title(f"Zużycie – {prov_sel}", fontweight="bold")
            axes[0].set_xlabel("Rok"); axes[0].set_ylabel("GWh")
            axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
            axes[0].xaxis.set_major_locator(mticker.MultipleLocator(5))
            axes[1].plot(dp["rok"], dp["dochod_os"],
                         "o-", color=GREEN, lw=2, ms=6)
            axes[1].set_title(f"Dochód na osobę – {prov_sel}", fontweight="bold")
            axes[1].set_xlabel("Rok"); axes[1].set_ylabel("zł")
            axes[1].xaxis.set_major_locator(mticker.MultipleLocator(5))
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with c2:
            st.dataframe(dp[["rok","zuzycie_energii_GWh","dochod_os",
                              "cena_energii_zl_kWh","hdd"]].round(2)
                         .reset_index(drop=True), use_container_width=True, height=300)

        st.subheader("Porównanie województw – zużycie 2024")
        df_2024 = (df_w[df_w["rok"]==2024]
                   .sort_values("zuzycie_energii_GWh", ascending=True))
        fig, ax = plt.subplots(figsize=(12, 6))
        colors_bar = [RED if r["wojewodztwo"] == prov_sel else BLUE
                      for _, r in df_2024.iterrows()]
        bars = ax.barh(df_2024["wojewodztwo"], df_2024["zuzycie_energii_GWh"],
                       color=colors_bar, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Zużycie energii [GWh]")
        ax.set_title("Zużycie energii per województwo (2024)", fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        for bar, val in zip(bars, df_2024["zuzycie_energii_GWh"]):
            ax.text(val+20, bar.get_y()+bar.get_height()/2,
                    f"{val:,.0f}", va="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with st.expander("Wykresy z analizy (wygenerowane pliki)"):
            c1, c2 = st.columns(2)
            with c1: show_png("ew01_szeregi_czasowe.png", "Szeregi per województwo")
            with c2: show_png("ew02_porownanie_woj.png", "Porównanie województw")
            c1, c2 = st.columns(2)
            with c1: show_png("ew03_korelacja.png", "Macierz korelacji")
            with c2: show_png("ew04_scatter.png", "Wykresy rozrzutu")

# ═══════════════════════════════════════════════════════════════
# TAB 2 – MODEL POLSKA
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("Model OLS – Polska (Iteracja 2)")
    st.markdown(
        "`ln(ZUZYCIE) = β₀ + β₁·ln(PKB_pc) + β₂·ln(CENA) + β₃·HDD`  \n"
        "*CDD usunięte: p=0.57, usuniecie poprawia AIC; urbanizacja usunięta: VIF>35*"
    )

    model_tr_p, model_full_p, df_p, df_tr_p, MC_p, vif_p, diag_p = fit_polska()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²",     f"{model_full_p.rsquared:.4f}")
    c2.metric("R²_adj", f"{model_full_p.rsquared_adj:.4f}")
    c3.metric("AIC",    f"{model_full_p.aic:.2f}")
    c4.metric("BIC",    f"{model_full_p.bic:.2f}")
    c5.metric("F p-val",f"{model_full_p.f_pvalue:.4f}")

    sub1, sub2 = st.tabs(["Parametry i diagnostyka", "Wykresy"])

    with sub1:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Parametry modelu (2004–2024)")
            coef_df = model_coef_df(model_full_p)

            def color_sig(val):
                if val == "***": return "background-color:#e8f5e9;color:#1a5c96;font-weight:bold"
                if val == "**":  return "background-color:#f0f8e8"
                if val == "*":   return "background-color:#fff3e0"
                return "background-color:#ffebee;color:#c0392b"

            st.dataframe(
                coef_df.style.map(color_sig, subset=["Istotność"]),
                use_container_width=True, height=210, hide_index=True
            )
            st.caption("Proba uczaca (2004–2022):")
            coef_tr = model_coef_df(model_tr_p)
            st.dataframe(coef_tr.style.map(color_sig, subset=["Istotność"]),
                         use_container_width=True, height=210, hide_index=True)

        with c2:
            st.subheader("VIF")
            st.dataframe(vif_p.round(3), use_container_width=True,
                         height=160, hide_index=True)
            st.subheader("Testy diagnostyczne")
            diag_rows = [
                {"Test": "Durbin-Watson", "Wartość": diag_p["DW"],
                 "Wynik": "✅ OK" if 1.5 < diag_p["DW"] < 2.5 else "⚠️ Autokorelacja"},
                {"Test": "Shapiro-Wilk (p)", "Wartość": diag_p["SW_p"],
                 "Wynik": "✅ OK" if diag_p["SW_p"] > 0.05 else "⚠️ Nienormalność"},
                {"Test": "Breusch-Godfrey (p)", "Wartość": diag_p["BG_p"],
                 "Wynik": "✅ OK" if diag_p["BG_p"] > 0.05 else "⚠️ Autokorelacja"},
                {"Test": "Breusch-Pagan (p)", "Wartość": diag_p["BP_p"],
                 "Wynik": "✅ OK" if diag_p["BP_p"] > 0.05 else "⚠️ Heterosked."},
            ]
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True,
                         height=190, hide_index=True)

    with sub2:
        fitted_p = np.exp(np.asarray(model_full_p.fittedvalues).ravel())
        actual_p = df_p["zuzycie_energii_GWh"].values
        resid_p  = np.asarray(model_full_p.resid).ravel()

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax = axes[0]
        ax.plot(df_p["rok"], actual_p, "o-", color="black", lw=2, ms=6, label="Rzeczywiste")
        ax.plot(df_p["rok"], fitted_p, "--", color=BLUE,   lw=2, label="Dopasowane OLS")
        ax.set_title("Rzeczywiste vs Dopasowane", fontweight="bold")
        ax.set_ylabel("GWh"); ax.set_xlabel("Rok")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.scatter(fitted_p, resid_p, color=BLUE, alpha=0.6, s=60, edgecolors="none")
        ax.axhline(0, color=RED, ls="--", lw=1.5)
        ax.set_xlabel("Wartości dopasowane (ln)"); ax.set_ylabel("Reszty")
        ax.set_title("Reszty vs Dopasowane", fontweight="bold")

        ax = axes[2]
        ax.hist(resid_p, bins=12, color=BLUE, alpha=0.75, edgecolor="white")
        from scipy.stats import norm as sp_norm
        x_n = np.linspace(resid_p.min(), resid_p.max(), 100)
        ax.plot(x_n, sp_norm.pdf(x_n, resid_p.mean(), resid_p.std())
                * len(resid_p)*(resid_p.max()-resid_p.min())/12,
                "r-", lw=2, label="Rozkład normalny")
        ax.set_title("Histogram reszt", fontweight="bold")
        ax.legend(fontsize=9)

        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with st.expander("Pełna diagnostyka z analizy"):
            c1, c2 = st.columns(2)
            with c1: show_png("ep04_diagnostyka.png", "Diagnostyka – Model 1")
            with c2: show_png("ep08_diagnostyka_iter2.png", "Diagnostyka – Iteracja 2")
        show_png("ep05_prognoza_warunkowa.png", "Prognoza scenariuszowa 2025–2030")

# ═══════════════════════════════════════════════════════════════
# TAB 3 – MODEL WOJEWÓDZTWA
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Pooled OLS – Województwa")
    st.markdown(
        "`ln(ZUZYCIE) = β₀ + β₁·ln(DOCHOD_lag1) + β₂·ln(CENA) + β₃·URBANIZACJA "
        "+ β₄·LICZBA_OS + β₅·POW_OS + β₆·HDD`  \n"
        "*Panel: 16 woj. × 20 lat (2005–2024) = 320 obs. | CDD usunięte: p=0.61*"
    )

    model_tr_w, model_full_w, df_w, dm_w, XC_w, vif_w, diag_w = fit_woj()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²",       f"{model_tr_w.rsquared:.4f}")
    c2.metric("R²_adj",   f"{model_tr_w.rsquared_adj:.4f}")
    c3.metric("AIC",      f"{model_tr_w.aic:.2f}")
    c4.metric("BIC",      f"{model_tr_w.bic:.2f}")
    c5.metric("n (train)", f"{int(model_tr_w.nobs)}")

    sub1, sub2, sub3 = st.tabs(["Pooled OLS", "Per województwo", "Diagnostyka"])

    with sub1:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Parametry Pooled OLS (2005–2022)")
            coef_w = model_coef_df(model_tr_w)

            def color_sig(val):
                if val == "***": return "background-color:#e8f5e9;font-weight:bold"
                if val == "**":  return "background-color:#f0f8e8"
                if val == "*":   return "background-color:#fff3e0"
                return "background-color:#ffebee;color:#c0392b"

            st.dataframe(coef_w.style.map(color_sig, subset=["Istotność"]),
                         use_container_width=True, height=280, hide_index=True)
        with c2:
            st.subheader("VIF (Pooled OLS)")
            st.dataframe(vif_w.round(2), use_container_width=True,
                         height=240, hide_index=True)
            diag_rows_w = [
                {"Test": "Durbin-Watson", "Wartość": diag_w["DW"],
                 "Wynik": "✅ OK" if 1.5 < diag_w["DW"] < 2.5 else "⚠️"},
                {"Test": "Shapiro-Wilk (p)", "Wartość": diag_w["SW_p"],
                 "Wynik": "✅ OK" if diag_w["SW_p"] > 0.05 else "⚠️ Nienormalność"},
            ]
            st.dataframe(pd.DataFrame(diag_rows_w), use_container_width=True,
                         height=130, hide_index=True)

    with sub2:
        df_prov_res, PROV, prov_models_dict = fit_prov_models()
        st.subheader("Zestawienie modeli per województwo")
        st.dataframe(df_prov_res, use_container_width=True, height=340, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            df_s = df_prov_res.sort_values("β_dochod")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(df_s["Województwo"], df_s["β_dochod"],
                    color=[GREEN if v>0 else RED for v in df_s["β_dochod"]],
                    alpha=0.85, edgecolor="white")
            ax.axvline(0, color="black", lw=1)
            ax.set_title("Elastyczność dochodowa β₁\n(ln_dochod_os_lag1)",
                         fontweight="bold")
            ax.set_xlabel("β₁")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with c2:
            df_s = df_prov_res.sort_values("β_cena")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(df_s["Województwo"], df_s["β_cena"],
                    color=[GREEN if v<0 else RED for v in df_s["β_cena"]],
                    alpha=0.85, edgecolor="white")
            ax.axvline(0, color="black", lw=1)
            ax.set_title("Elastyczność cenowa β₂\n(ln_cena)", fontweight="bold")
            ax.set_xlabel("β₂")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.subheader("Szczegóły modelu – wybrane województwo")
        prov_detail = st.selectbox("Województwo:", list(prov_models_dict.keys()))
        mdl_d = prov_models_dict[prov_detail]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²",     f"{mdl_d.rsquared:.4f}")
        c2.metric("R²_adj", f"{mdl_d.rsquared_adj:.4f}")
        c3.metric("AIC",    f"{mdl_d.aic:.2f}")
        c4.metric("n",      f"{int(mdl_d.nobs)}")
        st.dataframe(model_coef_df(mdl_d).style.map(color_sig, subset=["Istotność"]),
                     use_container_width=True, height=280, hide_index=True)

    with sub3:
        c1, c2 = st.columns(2)
        with c1: show_png("ew06_diagnostyka.png", "Diagnostyka Pooled OLS")
        with c2: show_png("ew05_elastycznosci_woj.png", "Elastyczności per województwo")
        show_png("ew07_prognoza.png", "Prognoza scenariuszowa 2025–2030")

# ═══════════════════════════════════════════════════════════════
# TAB 4 – PROGNOZY Z4
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("Zadanie 4 – Prognozy zmiennych objaśniających")
    st.markdown(
        "**Metody:** OLS_lin · OLS_kw · AR(1) · AR(2) · ARIMA · Holt · Pawłowski  \n"
        "**Proba uczaca:** 2004–2022 &nbsp;|&nbsp; **Test:** 2023–2024 &nbsp;|&nbsp; "
        "**Kryterium wyboru:** RMSPE% (≤10% = OK)"
    )

    scope4 = st.radio("Zakres", ["Polska", "Województwa"], horizontal=True, key="r4")

    if scope4 == "Polska":
        c1, c2 = st.columns(2)
        with c1: show_png("ep_z4_01_pkb.png", "PKB per capita")
        with c2: show_png("ep_z4_02_cena.png", "Cena energii [zł/kWh]")
        c1, c2 = st.columns(2)
        with c1: show_png("ep_z4_03_urban.png", "Urbanizacja [%]")
        with c2: show_png("ep_z4_04_hdd.png",   "HDD (stopniodni grzewcze)")
        show_png("ep_z4_05_podsumowanie.png", "Podsumowanie – wszystkie zmienne")
    else:
        vtabs = st.tabs(["Dochód","Cena","Urbanizacja","Liczba os.","Pow. os.","HDD","Podsumowanie"])
        pngs4 = [
            ("ew_z4_01_dochod.png",   "Dochód na osobę"),
            ("ew_z4_02_cena.png",     "Cena energii"),
            ("ew_z4_03_urban.png",    "Urbanizacja"),
            ("ew_z4_04_liczba_os.png","Liczba osób w gosp."),
            ("ew_z4_05_pow_os.png",   "Pow. mieszk. na os."),
            ("ew_z4_06_hdd.png",      "HDD"),
            ("ew_z4_07_heatmapy.png", "Heatmapy RMSPE%"),
        ]
        for vt, (png, cap) in zip(vtabs, pngs4):
            with vt:
                show_png(png, cap)
                if "heatmap" in png or "rmspe" in png.lower():
                    show_png("ew_z4_08_rmspe_bar.png", "RMSPE% – wykres słupkowy")

# ═══════════════════════════════════════════════════════════════
# TAB 5 – PROGNOZA WARUNKOWA Z5
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.header("Zadanie 5 – Prognoza warunkowa")
    st.markdown(
        "Prognoza Y oparta na prognozach X z Z4. "
        "Miary jakości (test 2023–2024) + prognoza ex-ante 2025."
    )

    scope5 = st.radio("Zakres", ["Polska", "Województwa"], horizontal=True, key="r5")

    if scope5 == "Polska":
        with st.spinner("Obliczanie prognozy warunkowej (pierwsze uruchomienie ~5s)…"):
            z5 = compute_z5_polska()

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prognoza 2025",   f"{z5['y_fc25']:,.0f} GWh")
        c2.metric("95% CI dolna",    f"{z5['y_lo']:,.0f} GWh")
        c3.metric("95% CI górna",    f"{z5['y_hi']:,.0f} GWh")
        last_y = float(z5["df"]["zuzycie_energii_GWh"].iloc[-1])
        c4.metric("Zmiana vs 2024",  f"{(z5['y_fc25']/last_y - 1)*100:+.2f}%")

        # Quality table
        st.subheader("Miary jakości (test 2023–2024)")
        miary = [
            ("Prog. warunkowa (X z Z4)",       z5["m_cond"]),
            ("Model z rzecz. X – dolna granica", z5["m_act"]),
            ("Naiwna (ostatnia obs.)",            z5["m_naive"]),
        ]
        miary_rows = []
        for nm, m in miary:
            th = f"{m['Theil_U']:.4f}" if isinstance(m.get("Theil_U"), float) and not np.isnan(m.get("Theil_U",np.nan)) else "N/A"
            ok = "✅ OK" if not np.isnan(m.get("RMSPE%", np.nan)) and abs(m.get("RMSPE%",99)) <= 10 else "❌ >10%"
            miary_rows.append({
                "Metoda": nm, "ME [GWh]": m.get("ME"), "MPE%": m.get("MPE%"),
                "MAE [GWh]": m.get("MAE"), "MAPE%": m.get("MAPE%"),
                "RMSE [GWh]": m.get("RMSE"), "RMSPE%": m.get("RMSPE%"),
                "Theil U": th, "Ocena": ok,
            })
        st.dataframe(pd.DataFrame(miary_rows), use_container_width=True,
                     hide_index=True, height=160)

        # Forecast chart (inline)
        df_plt = z5["df"]; df_tr_plt = z5["df_tr"]
        TEST = z5["TEST"]; FC = z5["FC"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        ax = axes[0]
        ax.plot(df_tr_plt["rok"], df_tr_plt["zuzycie_energii_GWh"],
                "ko-", lw=2, ms=6, label="Historia (train)", zorder=5)
        ax.plot(TEST, z5["y_actual"], "ko-", lw=2, ms=6, zorder=5)
        ax.plot(TEST, z5["y_cond"],  "r^-",  lw=2.5, ms=10, zorder=7,
                label="Prog. warunkowa (X z Z4)")
        ax.plot(TEST, z5["y_act"],   "gs--", lw=1.8, ms=8,  zorder=6,
                label="Model z rzecz. X")
        ax.plot(FC, z5["y_fc25"], "r*", ms=18, zorder=8,
                label=f"FC {FC}: {z5['y_fc25']:,.0f} GWh")
        ax.fill_between([FC-0.3, FC+0.3], [z5["y_lo"]]*2, [z5["y_hi"]]*2,
                        color="red", alpha=0.25, zorder=5)
        ax.axvspan(TEST[0]-0.5, TEST[-1]+0.5, alpha=0.07, color="red")
        ax.axvline(FC-0.5, color=GRAY, ls=":", lw=1.2)
        ax.set_xlabel("Rok"); ax.set_ylabel("GWh")
        ax.set_title("Prognoza warunkowa – Polska", fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.legend(fontsize=8, ncol=2)

        ax2 = axes[1]
        names_b = ["Prog. warunkowa\n(X z Z4)", "Model z\nrzeczywnymi X", "Naiwna"]
        rmspe_b = [abs(m.get("RMSPE%", 0)) for _, m in miary]
        mape_b  = [m.get("MAPE%", 0)        for _, m in miary]
        cols_b  = [GREEN if v <= 10 else RED for v in rmspe_b]
        xp = np.arange(3); w = 0.35
        ax2.bar(xp-w/2, rmspe_b, w, color=cols_b, alpha=0.85, edgecolor="white", label="RMSPE%")
        ax2.bar(xp+w/2, mape_b,  w, color=BLUE, alpha=0.5,    edgecolor="white", label="MAPE%")
        ax2.axhline(10, color=RED, ls="--", lw=2, label="Próg 10%")
        ax2.set_xticks(xp); ax2.set_xticklabels(names_b, fontsize=9)
        ax2.set_title("Miary jakości (test 2023–2024)", fontweight="bold")
        ax2.set_ylabel("Błąd [%]")
        for bar, val in zip(ax2.patches[:3], rmspe_b):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # X variable methods
        with st.expander("Szczegóły prognoz zmiennych X (Z4)"):
            xi_rows = [{"Zmienna": col, "Najlepsza metoda": inf["method"],
                        "RMSPE%": round(inf["rmspe"], 2),
                        "Ocena": "✅" if inf["rmspe"] <= 10 else "⚠️"}
                       for col, inf in z5["x_info"].items()]
            st.dataframe(pd.DataFrame(xi_rows), use_container_width=True,
                         hide_index=True, height=160)

    else:  # Województwa
        c1, c2 = st.columns(2)
        with c1:
            show_png("ew_z5_01_prognoza_woj.png",
                     "Prognoza warunkowa per województwo (test 2023–2024)")
        with c2:
            show_png("ew_z5_02_miary_woj.png",
                     "RMSPE% per województwo")

        show_png("ew_z5_03_agregat.png",
                 "Agregat krajowy + miary per województwo")

        with st.expander("Tabela pełnych miar jakości"):
            show_png("ew_z5_04_tabela_miar.png", "Miary jakości – wszystkie województwa")

        st.info(
            "**Interpretacja wyników:**  \n"
            "- 5/16 województw ma RMSPE% ≤ 10% (śląskie 2.6%, łódzkie 5.6%, "
            "dolnośląskie 8.9%, kujawsko-pomorskie 8.0%, podkarpackie 9.7%)  \n"
            "- Większe błędy wynikają z ograniczeń Pooled OLS (brak efektów stałych, R²=0.39)  \n"
            "- Model FE (Fixed Effects) dawałby lepsze dopasowanie per województwo  \n"
            "- Agregat krajowy (suma 16 woj.) prognoza 2025: **33 837 GWh**"
        )
