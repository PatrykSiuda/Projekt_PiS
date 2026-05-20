# -*- coding: utf-8 -*-
# ============================================================
# ANALIZA I PROGNOZA MIESZKAN ODDANYCH DO UZYTKOWANIA
# DANE PANELOWE – 16 WOJEWÓDZTW, LATA 2004–2024
# ============================================================
# Wymagane pakiety:
# pip install pandas numpy matplotlib seaborn statsmodels scipy openpyxl pmdarima

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
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats.stattools import jarque_bera

try:
    import pmdarima as pm
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False
    print("UWAGA: pmdarima niedostepne – sekcja ARIMA pominieta.")

# ── SCIEZKA ──────────────────────────────────────────────────
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
DATA_FILE = os.path.join(SCRIPT_DIR, "mieszkania_wojewodztwa.xlsx")

# ── KOLORY ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "font.family": "DejaVu Sans",
})
BLUE   = "#1a5c96"; RED    = "#c0392b"; GREEN  = "#27ae60"
ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY   = "#7f8c8d"

PALETTE = plt.cm.tab20.colors[:16]

# ── 1. WCZYTANIE DANYCH ──────────────────────────────────────
df = pd.read_excel(DATA_FILE)

# Poprawka nazwy kolumny nakladów (problem z kodowaniem znaku L)
for col in df.columns:
    if col.startswith("NAK"):
        df = df.rename(columns={col: "NAKL"})
        break

print("=" * 60)
print("WCZYTANE DANE – PODGLAD")
print("=" * 60)
print(f"Wymiary: {df.shape[0]} wierszy x {df.shape[1]} kolumn")
print(f"Kolumny: {list(df.columns)}")
print(f"Województwa: {sorted(df['wojewodztwo'].unique())}")
print(f"Lata: {df['rok'].min()}–{df['rok'].max()}")

# ── 1b. PORZADKOWANIE I ZMIENNE POCHODNE ─────────────────────
df = df.sort_values(["wojewodztwo", "rok"]).reset_index(drop=True)
df["MO_lag1"] = df.groupby("wojewodztwo")["MO"].shift(1)
df["trend"]   = df.groupby("wojewodztwo").cumcount()

PROVINCES = sorted(df["wojewodztwo"].unique())
N_PROV    = len(PROVINCES)

# ── 2. ANALIZA OPISOWA ───────────────────────────────────────
print("\n" + "=" * 60)
print("STATYSTYKI OPISOWE – CALY PANEL")
print("=" * 60)
desc_cols = ["MO", "NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM"]
desc = df[desc_cols].describe().T
desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(2)
print(desc.round(3).to_string())

print("\n" + "=" * 60)
print("SREDNIA LICZBA MIESZKAN ODDANYCH PER WOJEWÓDZTWO [szt./rok]")
print("=" * 60)
avg = (df.groupby("wojewodztwo")["MO"]
         .mean().sort_values(ascending=False))
for woj, val in avg.items():
    print(f"  {woj:<25} {val:,.0f} szt./rok")

# ── 3. WYKRESY SZEREGÓW CZASOWYCH (4x4) ──────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
fig.suptitle("Mieszkania oddane do uzytkowania w województwach (2004–2024)",
             fontsize=14, fontweight="bold", y=1.01)

for i, (ax, prov) in enumerate(zip(axes.flat, PROVINCES)):
    dp = df[df["wojewodztwo"] == prov]
    ax.plot(dp["rok"], dp["MO"],
            marker="o", color=PALETTE[i], linewidth=2, markersize=4)
    ax.set_title(prov, fontsize=9, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    if i >= 12:
        ax.set_xlabel("Rok", fontsize=8)
    if i % 4 == 0:
        ax.set_ylabel("szt.", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw01_szeregi_czasowe.png"),
            bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw01_szeregi_czasowe.png")

# ── 3b. PORÓWNAWCZY WYKRES LINIOWY ───────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
for i, prov in enumerate(PROVINCES):
    dp = df[df["wojewodztwo"] == prov]
    ax.plot(dp["rok"], dp["MO"],
            color=PALETTE[i], linewidth=1.8, label=prov)
ax.set_title("Mieszkania oddane do uzytkowania – wszystkie województwa",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Rok")
ax.set_ylabel("Liczba mieszkan oddanych [szt.]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(fontsize=7, ncol=4, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw02_porownanie_woj.png"),
            bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw02_porownanie_woj.png")

# ── 4. MACIERZ KORELACJI ─────────────────────────────────────
corr_cols   = ["MO", "NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 10})
ax.set_title("Macierz korelacji Pearsona – caly panel", fontsize=13, fontweight="bold")
labels = ["Mieszkania oddane", "Naklady", "Wynagrodzenie",
          "Udzial 25-34", "Urbanizacja", "Saldo migracji"]
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw03_korelacja.png"), bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw03_korelacja.png")

# ── 5. WYKRESY ROZRZUTU ──────────────────────────────────────
scatter_vars = [
    ("NAKL",     "Naklady na budownictwo [mln zl]", GREEN),
    ("WYNAGR",   "Wynagrodzenie [zl]",               ORANGE),
    ("WSK25-34", "Udzial wieku 25-34",               PURPLE),
    ("WSK_URB",  "Urbanizacja",                      GRAY),
    ("SM",       "Saldo migracji [promil]",          RED),
    ("trend",    "Trend",                            "#2980b9"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Zaleznosci liczby mieszkan oddanych od zmiennych – panel",
             fontsize=13, fontweight="bold")
for ax, (col, xlabel, color) in zip(axes.flat, scatter_vars):
    ax.scatter(df[col], df["MO"], color=color, alpha=0.3, s=25, edgecolors="none")
    tmp = df[[col, "MO"]].dropna()
    z = np.polyfit(tmp[col], tmp["MO"], 1)
    x_ln = np.linspace(tmp[col].min(), tmp[col].max(), 100)
    ax.plot(x_ln, np.poly1d(z)(x_ln), "--", color="black", linewidth=1.5)
    r, pv = stats.pearsonr(tmp[col], tmp["MO"])
    ax.set_xlabel(xlabel); ax.set_ylabel("Mieszkania oddane [szt.]")
    ax.set_title(f"r = {r:.3f}  (p = {pv:.3f})", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw04_scatter.png"), bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw04_scatter.png")

# ── 6. POOLED OLS – MODEL 1 ──────────────────────────────────
#
# Model liniowy na danych panelowych (pooled OLS):
#   MO_it = b0 + b1*NAKL + b2*WYNAGR + b3*WSK25-34
#          + b4*WSK_URB + b5*SM + e
#
# i = województwo, t = rok
# Pooled OLS traktuje wszystkie 336 obserwacji jednakowo.

X_cols  = ["NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM"]
y_pool  = df["MO"]
X_pool  = sm.add_constant(df[X_cols])

model_pool = sm.OLS(y_pool, X_pool).fit()

print("\n" + "=" * 60)
print("MODEL 1 – POOLED OLS (wyniki estymacji)")
print("=" * 60)
print("  SPECYFIKACJA:")
print("    MO_it = b0 + b1*NAKL + b2*WYNAGR + b3*WSK25-34")
print("          + b4*WSK_URB + b5*SM + e")
print("  Dane panelowe: 16 wojewodztw x 21 lat = 336 obserwacji")
print("  Pooled OLS: brak efektow stalych wojewodztw.")
print(model_pool.summary())
sys.stdout.flush()

# ── 6b. FE OLS – MODEL 1b ────────────────────────────────────
#
# Fixed Effects OLS: dodajemy zmienne zero-jedynkowe dla kazdego
# województwa, aby kontrolowac nieobserwowalna heterogenicznosc.

dummies      = pd.get_dummies(df["wojewodztwo"], drop_first=True, prefix="woj").astype(float)
X_fe_dummies = pd.concat([X_pool, dummies], axis=1)
model_fe     = sm.OLS(y_pool, X_fe_dummies).fit()

print("\n" + "=" * 60)
print("MODEL 1b – OLS Z EFEKTAMI STALYMI WOJEWÓDZTW (FE)")
print("=" * 60)
print("  Dodano zmienne zero-jedynkowe dla 15 województw")
print("  (dolnoslaskie = baza).")
print("  Efekty stale kontroluja stale roznice miedzy województwami")
print("  (np. struktura rynku, polozenie geograficzne).")
print(model_fe.summary())
sys.stdout.flush()

# ── 7. WERYFIKACJA NUMERYCZNA – POOLED OLS ───────────────────
print("\n" + "=" * 60)
print("WERYFIKACJA NUMERYCZNA – POOLED OLS")
print("=" * 60)

R2      = model_pool.rsquared
R2_adj  = model_pool.rsquared_adj
AIC_p   = model_pool.aic
BIC_p   = model_pool.bic
F_stat  = model_pool.fvalue
F_pval  = model_pool.f_pvalue
n_pool  = int(model_pool.nobs)
k_pool  = len(model_pool.params) - 1
res_pool = model_pool.resid

print(f"  R2              = {R2:.4f}")
print(f"  R2 adj.         = {R2_adj:.4f}")
print(f"  AIC             = {AIC_p:.3f}")
print(f"  BIC             = {BIC_p:.3f}")
print(f"  F-statystyka    = {F_stat:.3f}  (p = {F_pval:.6f})")
print(f"  n = {n_pool},  k = {k_pool}")

print("\n  Istotnosc parametrów:")
for name, coef, pv in zip(model_pool.params.index,
                           model_pool.params, model_pool.pvalues):
    if not name.startswith("woj_"):
        sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else
              ("*" if pv < 0.1 else ""))
        print(f"    {name:<22} b = {coef:+.4f}   p = {pv:.4f}  {sig}")

vif_data = pd.DataFrame({
    "Zmienna": X_cols,
    "VIF": [variance_inflation_factor(X_pool.values, i+1)
            for i in range(len(X_cols))]
})
print("\n  VIF (Pooled OLS):")
print(vif_data.to_string(index=False))

# ── 8. WERYFIKACJA STOCHASTYCZNA – POOLED OLS ────────────────
print("\n" + "=" * 60)
print("WERYFIKACJA STOCHASTYCZNA – POOLED OLS")
print("=" * 60)

stat_sw, p_sw = shapiro(res_pool)
stat_jb, p_jb, _, _ = jarque_bera(res_pool)
dw_p = durbin_watson(res_pool)
bg_stat, bg_p, _, _ = acorr_breusch_godfrey(model_pool, nlags=2)
bp_lm, bp_p, _, _   = het_breuschpagan(res_pool, X_pool)

print(f"\n  Shapiro-Wilk:    p = {p_sw:.4f}  "
      f"{'OK' if p_sw > 0.05 else 'odrzucamy H0 – reszty nienormalne'}")
print(f"  Durbin-Watson:   DW = {dw_p:.4f}  "
      f"({'OK' if 1.5 < dw_p < 2.5 else 'mozliwa autokorelacja'})")
print(f"  Breusch-Godfrey: p = {bg_p:.4f}  "
      f"{'OK' if bg_p > 0.05 else 'autokorelacja wykryta'}")
print(f"  Breusch-Pagan:   p = {bp_p:.4f}  "
      f"{'OK' if bp_p > 0.05 else 'heteroskedastycznosc wykryta'}")
print("  UWAGA: W danych panelowych autokorelacja i heteroskedastycznosc")
print("  sa normalne. W razie wykrycia zalecane bledy standardowe HAC.")

# ── 9. ITERACJA 2 – KOREKTA WIELOKOLINIOWOSCI + HAC ──────────
print("\n" + "=" * 60)
print("ITERACJA 2 – MODEL 2 (korekta wielokoliniowosci + HAC)")
print("=" * 60)
print("  MOTYWACJA:")
print("    NAKL i WYNAGR sa silnie skorelowane (rosna razem w czasie).")
print("    Jesli VIF > 10 -> usuwamy WYNAGR jako zmienna bardziej posrednia.")
print("    Bledy standardowe HAC (Newey-West) koryguja autokorelacje reszt.")
print("  SPECYFIKACJA:")
print("    MO = b0 + b1*NAKL + b2*WSK25-34 + b3*WSK_URB + b4*SM + e")
print("    Bledy standardowe: HAC Newey-West (maxlags=2)")

X2_cols    = ["NAKL", "WSK25-34", "WSK_URB", "SM"]
X2         = sm.add_constant(df[X2_cols])
model2_ols = sm.OLS(df["MO"], X2).fit()
model2     = model2_ols.get_robustcov_results(cov_type="HAC", maxlags=2)

print(model2.summary())
sys.stdout.flush()

R2_2     = model2_ols.rsquared
R2_adj_2 = model2_ols.rsquared_adj
AIC_2    = model2_ols.aic
BIC_2    = model2_ols.bic
res2     = model2_ols.resid

print(f"\n  Porownanie dopasowania Model 1 vs Model 2:")
print(f"  {'Miara':<12} {'Model 1':>10}  {'Model 2':>10}")
print("  " + "-" * 36)
print(f"  {'R2':<12} {R2:>10.4f}  {R2_2:>10.4f}")
print(f"  {'R2 adj.':<12} {R2_adj:>10.4f}  {R2_adj_2:>10.4f}")
print(f"  {'AIC':<12} {AIC_p:>10.3f}  {AIC_2:>10.3f}")
print(f"  {'BIC':<12} {BIC_p:>10.3f}  {BIC_2:>10.3f}")

sw2_stat, sw2_p = shapiro(res2)
dw2 = durbin_watson(res2)
bg2_stat, bg2_p, _, _ = acorr_breusch_godfrey(model2_ols, nlags=2)
bp2_lm, bp2_p, _, _   = het_breuschpagan(res2, X2)

print(f"\n  Weryfikacja stochastyczna – Model 2:")
print(f"  Shapiro-Wilk  p = {sw2_p:.4f}  "
      f"{'OK' if sw2_p > 0.05 else 'BLAD – nienormalne reszty'}")
print(f"  Durbin-Watson   = {dw2:.4f}  "
      f"{'OK' if 1.5 < dw2 < 2.5 else 'UWAGA – autokorelacja -> stosujemy HAC'}")
print(f"  Breusch-Godfrey p = {bg2_p:.4f}  "
      f"{'OK' if bg2_p > 0.05 else 'BLAD – autokorelacja -> bledy HAC aktywne'}")
print(f"  Breusch-Pagan   p = {bp2_p:.4f}  "
      f"{'OK' if bp2_p > 0.05 else 'BLAD – heteroskedastycznosc'}")
print("  INFO: Bledy standardowe w Modelu 2 korygowane metoda HAC")
print("        (Newey-West, maxlags=2) – odporne na autokorelacje i heteroskedastycznosc.")
sys.stdout.flush()

# ── 10. MODELE OLS PER WOJEWÓDZTWO ───────────────────────────
print("\n" + "=" * 60)
print("MODELE OLS PER WOJEWÓDZTWO")
print("=" * 60)
print("  Specyfikacja per województwo (Model 1):")
print("    MO_t = b0 + b1*NAKL_t + b2*WYNAGR_t + b3*WSK25-34_t")
print("         + b4*WSK_URB_t + b5*SM_t + e")
print("  n = 21 obserwacji na województwo")

prov_models  = {}
prov_results = []

for prov in PROVINCES:
    dp  = df[df["wojewodztwo"] == prov].copy()
    y_p = dp["MO"]
    X_p = sm.add_constant(dp[X_cols])
    mdl = sm.OLS(y_p, X_p).fit()
    prov_models[prov] = mdl

    sw_p_v = shapiro(mdl.resid)[1]
    dw_v   = durbin_watson(mdl.resid)
    bg_p_v = acorr_breusch_godfrey(mdl, nlags=2)[1]
    bp_p_v = het_breuschpagan(mdl.resid, X_p)[1]

    prov_results.append({
        "województwo": prov,
        "R2":          round(mdl.rsquared, 4),
        "R2_adj":      round(mdl.rsquared_adj, 4),
        "AIC":         round(mdl.aic, 2),
        "b_NAKL":      round(mdl.params["NAKL"], 4),
        "b_WYNAGR":    round(mdl.params["WYNAGR"], 4),
        "b_WSK25":     round(mdl.params["WSK25-34"], 4),
        "DW":          round(dw_v, 3),
        "BG_p":        round(bg_p_v, 3),
        "SW_p":        round(sw_p_v, 3),
    })

    print(f"\n--- {prov.upper()} ---")
    print(mdl.summary())
    print(f"  DW = {dw_v:.3f}  |  BG p = {bg_p_v:.4f}  |  SW p = {sw_p_v:.4f}  |  BP p = {bp_p_v:.4f}")
    sys.stdout.flush()

df_prov = pd.DataFrame(prov_results)

print("\n" + "=" * 60)
print("ZESTAWIENIE ZBIORCZE – MODELE PER WOJEWÓDZTWO")
print("=" * 60)
print(df_prov.to_string(index=False))

# ── 11. ZBIORCZY WYKRES WSPOLCZYNNIKOW PER WOJEWÓDZTWO ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].barh(df_prov["województwo"], df_prov["b_NAKL"],
             color=[GREEN if v > 0 else RED for v in df_prov["b_NAKL"]],
             alpha=0.85, edgecolor="white")
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_title("Wplyw nakladow (b_NAKL) per województwo\n"
                  "[szt. na 1 mln zl nakladow]",
                  fontweight="bold")
axes[0].set_xlabel("b_NAKL")

axes[1].barh(df_prov["województwo"], df_prov["b_WYNAGR"],
             color=[GREEN if v > 0 else RED for v in df_prov["b_WYNAGR"]],
             alpha=0.85, edgecolor="white")
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Wplyw wynagrodzenia (b_WYNAGR) per województwo\n"
                  "[szt. na 1 zl wynagrodzenia]",
                  fontweight="bold")
axes[1].set_xlabel("b_WYNAGR")

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw05_wspolczynniki_woj.png"),
            bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw05_wspolczynniki_woj.png")

# ── 12. DIAGNOSTYKA – POOLED OLS ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Diagnostyka – Pooled OLS (caly panel)", fontsize=14,
             fontweight="bold")

fitted_pool = model_pool.fittedvalues

ax = axes[0, 0]
ax.scatter(fitted_pool, res_pool, color=BLUE, alpha=0.4, s=20)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Wartosci dopasowane [szt.]"); ax.set_ylabel("Reszty")
ax.set_title("Reszty vs Dopasowane")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax = axes[0, 1]
sm.qqplot(res_pool, line="s", ax=ax, alpha=0.5, color=BLUE)
ax.set_title("QQ-plot")

ax = axes[0, 2]
ax.hist(res_pool, bins=20, color=BLUE, alpha=0.8, edgecolor="white")
xmn, xmx = ax.get_xlim()
xn = np.linspace(xmn, xmx, 100)
ax.plot(xn, stats.norm.pdf(xn, res_pool.mean(), res_pool.std())
        * len(res_pool) * (xmx - xmn) / 20, "r-", linewidth=2)
ax.set_title("Histogram reszt")

ax = axes[1, 0]
for i, prov in enumerate(PROVINCES):
    dp_yr = df[df["wojewodztwo"] == prov]["rok"].values
    dp_re = res_pool[df["wojewodztwo"] == prov].values
    ax.plot(dp_yr, dp_re, color=PALETTE[i], alpha=0.6, linewidth=1)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_title("Reszty w czasie (per województwo)")
ax.set_xlabel("Rok")

ax = axes[1, 1]
ax.scatter(y_pool, fitted_pool, color=BLUE, alpha=0.4, s=20)
mn = min(y_pool.min(), fitted_pool.min())
mx = max(y_pool.max(), fitted_pool.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Idealne dopasowanie")
ax.set_xlabel("Rzeczywiste [szt.]"); ax.set_ylabel("Dopasowane [szt.]")
ax.set_title("Rzeczywiste vs Dopasowane")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()

ax = axes[1, 2]
cusum = np.cumsum(res_pool.values)
ax.plot(range(len(cusum)), cusum, color=PURPLE, linewidth=1.5)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
std_r = res_pool.std()
ax.axhline(+2 * std_r * np.sqrt(n_pool), color="red", linestyle=":", label="+-2sigma*sqrt(n)")
ax.axhline(-2 * std_r * np.sqrt(n_pool), color="red", linestyle=":")
ax.set_title("CUSUM reszt (stabilnosc)")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw06_diagnostyka.png"), bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw06_diagnostyka.png")

# ── 13. ARIMA PER WOJEWÓDZTWO ─────────────────────────────────
if PMDARIMA_OK:
    print("\n" + "=" * 60)
    print("ARIMA PER WOJEWÓDZTWO")
    print("=" * 60)

    arima_models = {}
    for prov in PROVINCES:
        dp = df[df["wojewodztwo"] == prov].copy().sort_values("rok")
        y_ts = dp["MO"].values
        try:
            mdl_ar = pm.auto_arima(y_ts, seasonal=False,
                                    suppress_warnings=True, stepwise=True)
            arima_models[prov] = mdl_ar
            order = mdl_ar.order
            aic   = mdl_ar.aic()
            print(f"  {prov:<25} ARIMA{order}  AIC={aic:.2f}")
        except Exception as e:
            print(f"  {prov:<25} BLAD: {e}")

# ── 14. PROGNOZA GLOBALNA + DEKOMPOZYCJA REGIONALNA ──────────
print("\n" + "=" * 60)
print("PROGNOZA GLOBALNA I DEKOMPOZYCJA REGIONALNA 2025–2030")
print("=" * 60)

# Agregacja do poziomu krajowego
df_nat = (df.groupby("rok").agg(
    MO      = ("MO",       "sum"),
    NAKL    = ("NAKL",     "sum"),
    WYNAGR  = ("WYNAGR",   "mean"),
    WSK2534 = ("WSK25-34", "mean"),
    WSK_URB = ("WSK_URB",  "mean"),
    SM      = ("SM",       "mean"),
).reset_index())
df_nat.rename(columns={"WSK2534": "WSK25-34"}, inplace=True)

# Model OLS na agregacie krajowym
y_nat = df_nat["MO"]
X_nat_cols = ["NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM"]
X_nat = sm.add_constant(df_nat[X_nat_cols])
model_nat = sm.OLS(y_nat, X_nat).fit()

print(f"\n  Model OLS na agregacie krajowym (n={len(df_nat)}):")
print(f"  R2 = {model_nat.rsquared:.4f},  R2adj = {model_nat.rsquared_adj:.4f}")
print(f"  AIC = {model_nat.aic:.2f},  BIC = {model_nat.bic:.2f}")

# Scenariusze 2025–2030
forecast_years = list(range(2025, 2031))
last = df_nat[df_nat["rok"] == df_nat["rok"].max()].iloc[0]

scenarios = {
    "Pesymistyczny": dict(
        nakl=0.010, wynagr=0.010, wsk25=-0.001, wsk_urb=0.001, sm=-0.1,
        color=RED,   ls="--"),
    "Bazowy":        dict(
        nakl=0.030, wynagr=0.030, wsk25=0.000,  wsk_urb=0.002, sm=0.0,
        color=BLUE,  ls="-"),
    "Optymistyczny": dict(
        nakl=0.055, wynagr=0.050, wsk25=0.001,  wsk_urb=0.003, sm=0.1,
        color=GREEN, ls="-."),
}

fc_nat = {}
for sc_name, sc in scenarios.items():
    vals, lo_vals, hi_vals = [], [], []
    for i, yr in enumerate(forecast_years, 1):
        x_pred = pd.DataFrame({
            "const":    [1.0],
            "NAKL":     [last["NAKL"]    * (1 + sc["nakl"])**i],
            "WYNAGR":   [last["WYNAGR"]  * (1 + sc["wynagr"])**i],
            "WSK25-34": [last["WSK25-34"] + sc["wsk25"] * i],
            "WSK_URB":  [last["WSK_URB"]  + sc["wsk_urb"] * i],
            "SM":       [last["SM"]       + sc["sm"] * i],
        })
        pr = model_nat.get_prediction(x_pred).summary_frame(alpha=0.05)
        vals.append(pr["mean"].values[0])
        lo_vals.append(pr["mean_ci_lower"].values[0])
        hi_vals.append(pr["mean_ci_upper"].values[0])

    fc_nat[sc_name] = dict(years=forecast_years, mean=vals,
                            lo=lo_vals, hi=hi_vals,
                            color=sc["color"], ls=sc["ls"])

    print(f"\n  Scenariusz: {sc_name}")
    for yr, mo, lo, hi in zip(forecast_years, vals, lo_vals, hi_vals):
        print(f"    {yr}: {mo:,.0f} szt.  [95% CI: {lo:,.0f}–{hi:,.0f}]")

# Udzialy regionalne (ostatni rok)
yr_last   = df["rok"].max()
last_prov = df[df["rok"] == yr_last][["wojewodztwo", "MO"]].copy()
total_last = last_prov["MO"].sum()
last_prov["udzial"] = last_prov["MO"] / total_last

print("\n" + "=" * 60)
print(f"DEKOMPOZYCJA PROGNOZY NA UDZIALY REGIONALNE (scen. bazowy)")
print("=" * 60)
print(f"  {'Województwo':<25} {'Udzial ' + str(yr_last):>12}  " +
      "  ".join(str(y) for y in forecast_years))
for _, row in last_prov.sort_values("udzial", ascending=False).iterrows():
    fc_prov = [f"{row['udzial'] * mo:,.0f}" for mo in fc_nat["Bazowy"]["mean"]]
    print(f"  {row['wojewodztwo']:<25} {row['udzial']*100:>10.1f}%  " +
          "  ".join(fc_prov))

# ── 15. WYKRES PROGNOZY ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df_nat["rok"], df_nat["MO"],
        "o-", color=GRAY, linewidth=2.5, markersize=7,
        label="Dane historyczne 2004–2024", zorder=5)
ax.axvline(yr_last + 0.5, color="black", linewidth=1.5,
           linestyle=":", alpha=0.6)

for sc_name, sc_data in fc_nat.items():
    ax.plot(sc_data["years"], sc_data["mean"],
            color=sc_data["color"], linewidth=2.5,
            linestyle=sc_data["ls"], marker="D", markersize=6,
            label=f"Scenariusz {sc_name}")
    ax.fill_between(sc_data["years"], sc_data["lo"], sc_data["hi"],
                    color=sc_data["color"], alpha=0.12)

ax.set_title(
    "Prognoza liczby mieszkan oddanych do uzytkowania – agregat z 16 województw 2025–2030\n"
    "Model OLS liniowy z czynnikami podazowymi i demograficznymi",
    fontsize=13, fontweight="bold")
ax.set_xlabel("Rok", fontsize=11)
ax.set_ylabel("Liczba mieszkan oddanych [szt.]", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xlim(2003, 2031)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw07_prognoza.png"), bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw07_prognoza.png")

# ── 16. WYKRES UDZIALÓW REGIONALNYCH ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
df_share = last_prov.sort_values("udzial", ascending=True)
bars = ax.barh(df_share["wojewodztwo"], df_share["udzial"] * 100,
               color=BLUE, alpha=0.85, edgecolor="white")
ax.set_xlabel("Udzial w krajowej liczbie mieszkan oddanych [%]")
ax.set_title(f"Udzialy województw w mieszkaniach oddanych ({yr_last})",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, df_share["udzial"] * 100):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "mw08_udzialy_regionalne.png"),
            bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: mw08_udzialy_regionalne.png")

# ── 17. PODSUMOWANIE ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PODSUMOWANIE")
print("=" * 60)
print(f"  Dane          : panel 16 województw x 21 lat = 336 obserwacji")
print(f"  Model pooled  : MO ~ NAKL + WYNAGR + WSK25-34 + WSK_URB + SM")
print(f"  R2 (pooled)   : {R2:.4f}")
print(f"  R2 (FE)       : {model_fe.rsquared:.4f}")
params_p = model_pool.params
print(f"  b_NAKL   (pooled) : {params_p['NAKL']:+.4f}  szt. / mln zl nakladow")
print(f"  b_WYNAGR (pooled) : {params_p['WYNAGR']:+.4f}  szt. / zl wynagrodzenia")
print(f"  Weryfikacja stochastyczna (pooled):")
print(f"    Normalnosc reszt (SW) : p = {p_sw:.4f}")
print(f"    Autokorelacja (BG)    : p = {bg_p:.4f}")
print(f"    Heteroskedastycznosc  : p = {bp_p:.4f}")
print(f"  Prognoza bazowa 2030 (agregat): {fc_nat['Bazowy']['mean'][-1]:,.0f} szt.")
print(f"  (ostatni rok historyczny {yr_last}: {total_last:,.0f} szt.)")
print("  Wygenerowane pliki: mw01..mw08_*.png")
