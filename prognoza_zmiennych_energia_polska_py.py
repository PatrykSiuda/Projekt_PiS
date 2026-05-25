# -*- coding: utf-8 -*-
# ============================================================
# ZADANIE 4 – PROGNOZY ZMIENNYCH OBJASNIAJACYCH
# ZUZYCIE ENERGII ELEKTRYCZNEJ – POLSKA (OGOLNOPOLSKA)
# ============================================================
# Proba uczaca  : 2004–2022  (n = 19)
# Okres testowy : 2023–2024  (h = 2)
# Prognoza ex-ante : 2025    (h = 1)
# Metody: OLS trend lin, OLS trend kw, AR(1), AR(2),
#         ARIMA (auto), Holt, Pawlowski
# Miary: ME, MPE%, MAE, MAPE%, RMSE, RMSPE%, Theil U
# ============================================================

import os
import sys

_terminal = len(sys.argv) > 0 and sys.argv[0].endswith(".py")
if _terminal:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib

if _terminal:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    import pmdarima as pm
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False
    print("UWAGA: pmdarima niedostepne – ARIMA pominiete.")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "font.family": "DejaVu Sans",
})
BLUE   = "#1a5c96"; RED    = "#c0392b"; GREEN  = "#27ae60"
ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY   = "#7f8c8d"

METHOD_COLORS = {
    "OLS_lin":   "#1a5c96",
    "OLS_kw":    "#27ae60",
    "AR(1)":     "#e67e22",
    "AR(2)":     "#c0392b",
    "ARIMA":     "#8e44ad",
    "Holt":      "#16a085",
    "Pawlowski": "#f39c12",
}

# ── 1. WCZYTANIE DANYCH ──────────────────────────────────────
df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_polska.xlsx"))
df = df.sort_values("rok").reset_index(drop=True)
df["pkb_per_capita"] = df["pkb_mln_zl"] * 1e6 / df["ludnosc"]

TRAIN_END    = 2022
TEST_START   = 2023
FC_YEAR      = 2025
YEARS_ALL    = df["rok"].values

print("=" * 65)
print("ZADANIE 4 – PROGNOZY ZMIENNYCH OBJASNIAJACYCH")
print("Energia elektryczna Polska | Proba: 2004-2022 | Test: 2023-2024")
print("=" * 65)
print(f"Obserwacje lacznie: {len(df)}  |  Proba uczaca: n = {(df['rok']<=TRAIN_END).sum()}  |  Test: n = {(df['rok']>TRAIN_END).sum()}")

# Konfiguracja zmiennych
VAR_CONFIG = [
    {"col": "pkb_per_capita",       "label": "PKB per capita [zl]",
     "transform": "log", "model_col": "ln_pkb_pc",          "file": "ep_z4_01_pkb.png"},
    {"col": "cena_energii_zl_kWh",  "label": "Cena energii [zl/kWh]",
     "transform": "log", "model_col": "ln_cena",             "file": "ep_z4_02_cena.png"},
    {"col": "urbanizacja_pct",      "label": "Urbanizacja [%]",
     "transform": None,  "model_col": "urbanizacja_pct",     "file": "ep_z4_03_urban.png"},
    {"col": "hdd",                  "label": "HDD",
     "transform": None,  "model_col": "hdd",                 "file": "ep_z4_04_hdd.png"},
    # CDD usuniete z modelu optymalnego (Iteracja 2): p=0.57, RMSPE%>20% w prognozie
]

# ── 2. MIARY JAKOSCI (wg PiS_lab_7_1) ────────────────────────
def eval_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred   = np.array(pred,   dtype=float)
    me    = np.mean(pred - actual)
    mpe   = 100.0 * np.mean((pred - actual) / actual)
    mae   = np.mean(np.abs(pred - actual))
    mape  = 100.0 * np.mean(np.abs((pred - actual) / actual))
    rmse  = np.sqrt(np.mean((pred - actual) ** 2))
    rmspe = 100.0 * rmse / np.mean(actual) if np.mean(actual) != 0 else np.nan
    theil = (np.sqrt(np.mean((pred[1:] - actual[1:]) ** 2) / np.mean(actual[1:] ** 2))
             if len(actual) > 1 else np.nan)
    return {
        "ME":      round(me,    4),
        "MPE%":    round(mpe,   4),
        "MAE":     round(mae,   4),
        "MAPE%":   round(mape,  4),
        "RMSE":    round(rmse,  4),
        "RMSPE%":  round(rmspe, 4),
        "Theil_U": round(theil, 4) if not np.isnan(theil) else np.nan,
    }

def _X1(t):
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    return np.column_stack([np.ones(len(t)), t])

def _X2(t):
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    return np.column_stack([np.ones(len(t)), t, t**2])

# ── 3. FUNKCJA PROGNOZOWANIA ──────────────────────────────────
def forecast_variable(y_all, years_all, train_end, n_exante):
    mask_tr = years_all <= train_end
    mask_te = years_all >  train_end
    y_tr = y_all[mask_tr].astype(float)
    y_te = y_all[mask_te].astype(float)
    n_tr, n_te = len(y_tr), len(y_te)
    t_tr = np.arange(1, n_tr + 1, dtype=float)
    t_te = np.arange(n_tr + 1, n_tr + n_te + 1, dtype=float)
    t_fc = np.arange(n_tr + n_te + 1, n_tr + n_te + n_exante + 1, dtype=float)

    res = {}

    # 1. OLS trend liniowy
    try:
        m = sm.OLS(y_tr, _X1(t_tr)).fit()
        res["OLS_lin"] = {
            "fitted":     np.asarray(m.fittedvalues).ravel(),
            "pred_test":  np.asarray(m.predict(_X1(t_te))).ravel(),
            "pred_fc":    np.asarray(m.predict(_X1(t_fc))).ravel(),
        }
    except Exception as e:
        print(f"  [OLS_lin] BLAD: {e}")

    # 2. OLS trend kwadratowy
    try:
        m = sm.OLS(y_tr, _X2(t_tr)).fit()
        res["OLS_kw"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": np.asarray(m.predict(_X2(t_te))).ravel(),
            "pred_fc":   np.asarray(m.predict(_X2(t_fc))).ravel(),
        }
    except Exception as e:
        print(f"  [OLS_kw] BLAD: {e}")

    # 3. AR(1)
    try:
        m = AutoReg(y_tr, lags=1, old_names=False).fit()
        res["AR(1)"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr + n_te - 1)).ravel(),
            "pred_fc":   np.asarray(m.predict(start=n_tr + n_te,
                                              end=n_tr + n_te + n_exante - 1)).ravel(),
        }
    except Exception as e:
        print(f"  [AR(1)] BLAD: {e}")

    # 4. AR(2)
    try:
        m = AutoReg(y_tr, lags=2, old_names=False).fit()
        res["AR(2)"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr + n_te - 1)).ravel(),
            "pred_fc":   np.asarray(m.predict(start=n_tr + n_te,
                                              end=n_tr + n_te + n_exante - 1)).ravel(),
        }
    except Exception as e:
        print(f"  [AR(2)] BLAD: {e}")

    # 5. ARIMA (auto)
    if PMDARIMA_OK:
        try:
            m = pm.auto_arima(y_tr, seasonal=False,
                               suppress_warnings=True, stepwise=True)
            pred_all = m.predict(n_periods=n_te + n_exante)
            res["ARIMA"] = {
                "fitted":    m.predict_in_sample(),
                "pred_test": pred_all[:n_te],
                "pred_fc":   pred_all[n_te:],
            }
        except Exception as e:
            print(f"  [ARIMA] BLAD: {e}")

    # 6. Holt (podwojne wygladzanie wykladnicze)
    try:
        m = ExponentialSmoothing(y_tr, trend="add", seasonal=None).fit(
            optimized=True)
        fc_all = np.asarray(m.forecast(n_te + n_exante)).ravel()
        res["Holt"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": fc_all[:n_te],
            "pred_fc":   fc_all[n_te:],
        }
    except Exception as e:
        print(f"  [Holt] BLAD: {e}")

    # 7. Metoda Pawlowskiego (trend lin + AR(1) reszt)
    try:
        m_trend = sm.OLS(y_tr, _X1(t_tr)).fit()
        e_tr    = np.asarray(m_trend.resid).ravel()
        e_df    = pd.DataFrame({"e_t": e_tr[1:], "e_lag": e_tr[:-1]})
        m_res   = sm.OLS(e_df["e_t"], sm.add_constant(e_df["e_lag"])).fit()
        alpha_e, rho_e = m_res.params["const"], m_res.params["e_lag"]

        e_last = e_tr[-1]
        e_list = []
        for _ in range(n_te + n_exante):
            e_next = alpha_e + rho_e * (e_list[-1] if e_list else e_last)
            e_list.append(e_next)
        e_arr     = np.array(e_list)
        trend_te  = np.asarray(m_trend.predict(_X1(t_te))).ravel()
        trend_fc  = np.asarray(m_trend.predict(_X1(t_fc))).ravel()
        res["Pawlowski"] = {
            "fitted":    np.asarray(m_trend.fittedvalues).ravel(),
            "pred_test": trend_te + e_arr[:n_te],
            "pred_fc":   trend_fc + e_arr[n_te:],
        }
    except Exception as e:
        print(f"  [Pawlowski] BLAD: {e}")

    # Miary jakosci
    metrics = {}
    for mname, mdata in res.items():
        try:
            metrics[mname] = eval_metrics(y_te, mdata["pred_test"])
        except Exception as ex:
            print(f"  [metrics {mname}] BLAD: {ex}")

    metrics_df = pd.DataFrame(metrics).T
    best_method = metrics_df["RMSPE%"].abs().idxmin()
    best_rmspe  = float(metrics_df.loc[best_method, "RMSPE%"])

    return metrics_df, best_method, best_rmspe, res

# ── 4. GLOWNA PETLA ───────────────────────────────────────────
YEARS_TR = YEARS_ALL[YEARS_ALL <= TRAIN_END]
YEARS_TE = YEARS_ALL[YEARS_ALL >  TRAIN_END]
YEARS_FC = np.array([FC_YEAR])

best_forecasts_raw = {}   # model_col -> wartosc raw (przed transformacja)
best_forecasts_mdl = {}   # model_col -> wartosc wejsciowa do modelu
summary_rows = []

for cfg in VAR_CONFIG:
    col = cfg["col"]
    print("\n" + "=" * 65)
    print(f"ZMIENNA: {cfg['label']}  (kolumna: {col})")
    print("=" * 65)

    y_all = df[col].values.astype(float)

    metrics_df, best_method, best_rmspe, results = forecast_variable(
        y_all, YEARS_ALL, TRAIN_END, len(YEARS_FC)
    )

    # Tabela miar
    print(f"\n  Miary jakosci prognozy – próba testowa {TEST_START}–{YEARS_TE[-1]}:")
    for mname, row in metrics_df.iterrows():
        ok = " <-- NAJLEPSZA" if mname == best_method else ""
        print(f"  {mname:<12}  ME={row['ME']:+8.4f}  MPE%={row['MPE%']:+7.2f}"
              f"  MAE={row['MAE']:8.4f}  MAPE%={row['MAPE%']:6.2f}"
              f"  RMSE={row['RMSE']:8.4f}  RMSPE%={row['RMSPE%']:6.2f}  Theil={row['Theil_U']:.4f}{ok}")

    ok_str = "OK (<=10%)" if abs(best_rmspe) <= 10 else "PRZEKROCZONE >10%"
    print(f"\n  >> Najlepsza metoda: {best_method}  |  RMSPE% = {best_rmspe:.2f}%  [{ok_str}]")

    # Prognozy ex-ante
    fc_raw = results[best_method]["pred_fc"][0]
    fc_mdl = np.log(fc_raw) if cfg["transform"] == "log" else fc_raw
    best_forecasts_raw[col]            = fc_raw
    best_forecasts_mdl[cfg["model_col"]] = fc_mdl

    print(f"  Prognoza {FC_YEAR} (raw)  : {fc_raw:.4f}")
    if cfg["transform"] == "log":
        print(f"  Prognoza {FC_YEAR} (ln)   : {fc_mdl:.4f}")

    summary_rows.append({
        "Zmienna":          col,
        "Najlepsza_metoda": best_method,
        "RMSPE%":           round(best_rmspe, 2),
        "OK_<=10%":         abs(best_rmspe) <= 10,
        f"Prog_{FC_YEAR}":  round(fc_raw, 4),
    })

    # ── Wykres ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Prognoza zmiennej: {cfg['label']}  [Proba: 2004–{TRAIN_END} | Test: {TEST_START}–{YEARS_TE[-1]}]",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(YEARS_ALL, y_all, "ko-", lw=2.5, ms=6, label="Rzeczywiste", zorder=5)
    ax.axvspan(TEST_START - 0.5, YEARS_TE[-1] + 0.5,
               alpha=0.08, color="red", label="Okres testowy")
    ax.axvline(FC_YEAR - 0.5, color="gray", lw=1.2, ls=":")

    for mname, mdata in results.items():
        c  = METHOD_COLORS.get(mname, "gray")
        lw = 2.5 if mname == best_method else 1.0
        ls = "-"  if mname == best_method else "--"
        zord = 6 if mname == best_method else 3
        ax.plot(YEARS_TE, mdata["pred_test"], marker="^",
                color=c, lw=lw, ls=ls, ms=5, label=mname, zorder=zord)
        ax.plot(YEARS_FC, mdata["pred_fc"],   marker="D",
                color=c, lw=lw, ls=ls, ms=7, zorder=zord)

    # Najlepsza wyrózniona
    bdata = results[best_method]
    ax.plot(YEARS_TE, bdata["pred_test"], "r-", lw=3.5, zorder=7, label=f"BEST: {best_method}")
    ax.plot(YEARS_FC, bdata["pred_fc"],   "r*", ms=14,  zorder=7)

    ax.set_xlabel("Rok"); ax.set_ylabel(cfg["label"])
    ax.set_title("Wartosci rzeczywiste vs prognozy (wszystkie metody)")
    ax.legend(fontsize=7, ncol=2)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

    # RMSPE bar chart
    ax2 = axes[1]
    rmspe_vals = metrics_df["RMSPE%"].abs()
    bar_colors = [GREEN if v <= 10 else RED for v in rmspe_vals]
    bars = ax2.bar(metrics_df.index, rmspe_vals,
                   color=bar_colors, alpha=0.85, edgecolor="white")
    ax2.axhline(10, color="red", ls="--", lw=2, label="Prog 10%")
    ax2.set_title("RMSPE% w okresie testowym")
    ax2.set_ylabel("RMSPE%"); ax2.set_xlabel("Metoda")
    for bar, val in zip(bars, rmspe_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.legend()
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, cfg["file"]), bbox_inches="tight")
    plt.show(); plt.close()
    print(f"  Zapisano: {cfg['file']}")
    sys.stdout.flush()

# ── 5. TABELA I WYKRES ZBIORCZY ───────────────────────────────
print("\n" + "=" * 65)
print("PODSUMOWANIE ZBIORCZE – ZMIENNE OBJASNIAJACE")
print("=" * 65)
df_sum = pd.DataFrame(summary_rows)
print(df_sum.to_string(index=False))

print(f"\nWartosci prognozowane na rok {FC_YEAR} (postac modelowa):")
for col, val in best_forecasts_mdl.items():
    print(f"  {col:<22} = {val:.6f}")

# Zbiorczy wykres RMSPE
fig, ax = plt.subplots(figsize=(11, 5))
x_labels = [r["Zmienna"] for r in summary_rows]
rmspe_s  = [abs(r["RMSPE%"]) for r in summary_rows]
meth_s   = [r["Najlepsza_metoda"] for r in summary_rows]
cols_s   = [GREEN if v <= 10 else RED for v in rmspe_s]
bars = ax.bar(x_labels, rmspe_s, color=cols_s, alpha=0.85, edgecolor="white")
ax.axhline(10, color="red", ls="--", lw=2, label="Próg 10%")
for bar, val, meth in zip(bars, rmspe_s, meth_s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{meth}\n{val:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_title("RMSPE% najlepszej metody prognozy – zmienne objaśniające energii (Polska)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("RMSPE%")
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ep_z4_05_podsumowanie.png"), bbox_inches="tight")
plt.show(); plt.close()
print("\nZapisano: ep_z4_05_podsumowanie.png")
print("Pliki wygenerowane: ep_z4_01 .. ep_z4_05_*.png")
