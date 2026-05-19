# -*- coding: utf-8 -*-
# ============================================================
# ANALIZA I PROGNOZA ZUŻYCIA ENERGII – DANE PANELOWE
# 16 województw, lata 2004–2024
# ============================================================
# Wymagane pakiety:
# pip install pandas numpy matplotlib seaborn statsmodels scipy openpyxl pmdarima

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
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
    print("pmdarima niedostepne – sekcja ARIMA pominieta.")

# ── SCIEZKA ──────────────────────────────────────────────────
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = r"C:\Users\Patryk\Desktop\studia magisterskie\Prognozowanie i symulacje\Projekt\Projekt_PiS"
DATA_FILE = os.path.join(SCRIPT_DIR, "Zuzycie_energii_wojewodztwa.xlsx")

# ── KOLORY ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "font.family": "DejaVu Sans",
})
BLUE   = "#1a5c96"; RED    = "#c0392b"; GREEN  = "#27ae60"
ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY   = "#7f8c8d"

# Paleta 16 kolorów dla województw
PALETTE = plt.cm.tab20.colors[:16]

# ── 1. WCZYTANIE DANYCH ──────────────────────────────────────
df = pd.read_excel(DATA_FILE)

print("=" * 60)
print("WCZYTANE DANE – PODGLAD")
print("=" * 60)
print(f"Wymiary: {df.shape[0]} wierszy x {df.shape[1]} kolumn")
print(f"Kolumny: {list(df.columns)}")
print(f"Województwa: {sorted(df['wojewodztwo'].unique())}")
print(f"Lata: {df['rok'].min()}–{df['rok'].max()}")

# ── 1b. ZMIENNE POCHODNE ─────────────────────────────────────
df = df.sort_values(["wojewodztwo", "rok"]).reset_index(drop=True)

df["pkb_per_capita"] = df["pkb_mln_zl"] * 1e6 / df["ludnosc"]
df["ln_pkb_pc"]      = np.log(df["pkb_per_capita"])
df["ln_zuzycie"]     = np.log(df["zuzycie_energii_GWh"])
df["ln_cena"]        = np.log(df["cena_energii_zl_kWh"])
df["ln_ludnosc"]     = np.log(df["ludnosc"])
df["trend"]          = df.groupby("wojewodztwo").cumcount()

# Opóźnienia wewnątrz każdego województwa
df["ln_zuzycie_lag1"] = df.groupby("wojewodztwo")["ln_zuzycie"].shift(1)

PROVINCES = sorted(df["wojewodztwo"].unique())
N_PROV    = len(PROVINCES)

# ── 2. ANALIZA OPISOWA ───────────────────────────────────────
print("\n" + "=" * 60)
print("STATYSTYKI OPISOWE – CAŁY PANEL")
print("=" * 60)
desc_cols = ["zuzycie_energii_GWh", "cena_energii_zl_kWh",
             "pkb_mln_zl", "ludnosc", "urbanizacja_pct", "hdd", "cdd"]
desc = df[desc_cols].describe().T
desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(2)
print(desc.round(3).to_string())

print("\n" + "=" * 60)
print("SREDNIE ZUŻYCIE ENERGII PER WOJEWÓDZTWO [GWh]")
print("=" * 60)
avg = (df.groupby("wojewodztwo")["zuzycie_energii_GWh"]
         .mean().sort_values(ascending=False))
for woj, val in avg.items():
    print(f"  {woj:<25} {val:,.0f} GWh")

# ── 3. WYKRESY SZEREGÓW CZASOWYCH ────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
fig.suptitle("Zużycie energii elektrycznej w województwach (2004–2024)",
             fontsize=14, fontweight="bold", y=1.01)

for i, (ax, prov) in enumerate(zip(axes.flat, PROVINCES)):
    dp = df[df["wojewodztwo"] == prov]
    ax.plot(dp["rok"], dp["zuzycie_energii_GWh"],
            marker="o", color=PALETTE[i], linewidth=2, markersize=4)
    ax.set_title(prov, fontsize=9, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    if i >= 12:
        ax.set_xlabel("Rok", fontsize=8)
    if i % 4 == 0:
        ax.set_ylabel("GWh", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w01_szeregi_czasowe.png"),
            bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: w01_szeregi_czasowe.png")

# ── 3b. PORÓWNAWCZY WYKRES LINIOWY ───────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
for i, prov in enumerate(PROVINCES):
    dp = df[df["wojewodztwo"] == prov]
    ax.plot(dp["rok"], dp["zuzycie_energii_GWh"],
            color=PALETTE[i], linewidth=1.8, label=prov)
ax.set_title("Zużycie energii elektrycznej – wszystkie województwa",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Rok")
ax.set_ylabel("Zużycie energii [GWh]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(fontsize=7, ncol=4, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w02_porownanie_woj.png"),
            bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: w02_porownanie_woj.png")

# ── 4. MACIERZ KORELACJI (cały panel) ────────────────────────
corr_cols   = ["zuzycie_energii_GWh", "pkb_per_capita",
               "cena_energii_zl_kWh", "urbanizacja_pct",
               "ludnosc", "hdd", "cdd"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 10})
ax.set_title("Macierz korelacji Pearsona – cały panel", fontsize=13, fontweight="bold")
labels = ["Zużycie energii", "PKB per capita", "Cena energii",
          "Urbanizacja", "Ludność", "HDD", "CDD"]
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w03_korelacja.png"), bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: w03_korelacja.png")

# ── 5. WYKRESY ROZRZUTU ──────────────────────────────────────
scatter_vars = [
    ("pkb_per_capita",      "PKB per capita [zł]",   GREEN),
    ("cena_energii_zl_kWh", "Cena energii [zł/kWh]", RED),
    ("urbanizacja_pct",     "Urbanizacja [%]",        ORANGE),
    ("ludnosc",             "Ludność [os.]",          PURPLE),
    ("hdd",                 "HDD",                   "#2980b9"),
    ("cdd",                 "CDD",                   "#e74c3c"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Zależności zużycia energii od zmiennych objaśniających – panel",
             fontsize=13, fontweight="bold")
for ax, (col, xlabel, color) in zip(axes.flat, scatter_vars):
    ax.scatter(df[col], df["zuzycie_energii_GWh"],
               color=color, alpha=0.3, s=30, edgecolors="none")
    tmp = df[[col, "zuzycie_energii_GWh"]].dropna()
    z = np.polyfit(tmp[col], tmp["zuzycie_energii_GWh"], 1)
    x_ln = np.linspace(tmp[col].min(), tmp[col].max(), 100)
    ax.plot(x_ln, np.poly1d(z)(x_ln), "--", color="black", linewidth=1.5)
    r, pv = stats.pearsonr(tmp[col], tmp["zuzycie_energii_GWh"])
    ax.set_xlabel(xlabel); ax.set_ylabel("Zużycie energii [GWh]")
    ax.set_title(f"r = {r:.3f}  (p = {pv:.3f})", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w04_scatter.png"), bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: w04_scatter.png")

# ── 6. POOLED OLS – MODEL 1 ──────────────────────────────────
#
# Model log-liniowy na danych panelowych (pooled OLS):
#   ln(ZUZYCIE_it) = β₀ + β₁·ln(PKB_pc_it) + β₂·ln(CENA_it)
#                  + β₃·URBANIZACJA_it + β₄·HDD_it + β₅·CDD_it + ε_it
#
# i = województwo, t = rok
# Pooled OLS traktuje wszystkie 336 obserwacji jednakowo.
# Uwaga: nie kontroluje efektów stałych województw.

X_cols = ["ln_pkb_pc", "ln_cena", "urbanizacja_pct", "hdd", "cdd"]
y_pool  = df["ln_zuzycie"]
X_pool  = sm.add_constant(df[X_cols])

model_pool = sm.OLS(y_pool, X_pool).fit()

print("\n" + "=" * 60)
print("MODEL 1 – POOLED OLS (wyniki estymacji)")
print("=" * 60)
print("  SPECYFIKACJA:")
print("    ln(ZUZYCIE_it) = b0 + b1*ln(PKB_pc) + b2*ln(CENA)")
print("                   + b3*URBANIZACJA + b4*HDD + b5*CDD + e")
print("  Dane panelowe: 16 wojewodztw x 21 lat = 336 obserwacji")
print("  Pooled OLS: brak efektow stalych wojewodztw.")
print(model_pool.summary())

# ── 6b. POOLED OLS Z EFEKTAMI STAŁYMI WOJEWÓDZTW ─────────────
#
# Fixed Effects OLS: dodajemy zmienne zero-jedynkowe dla każdego
# województwa, aby kontrolować nieobserwowaną heterogeniczność.
# To podejście "within" eliminuje stały efekt każdego województwa.

df["woj_cat"] = pd.Categorical(df["wojewodztwo"])
X_fe_cols = X_cols.copy()
X_fe = sm.add_constant(df[X_cols])

# Dodanie dummy dla województw (pomijamy pierwsze jako bazowe)
dummies = pd.get_dummies(df["wojewodztwo"], drop_first=True, prefix="woj").astype(float)
X_fe_dummies = pd.concat([X_fe, dummies], axis=1)

model_fe = sm.OLS(y_pool, X_fe_dummies).fit()

print("\n" + "=" * 60)
print("MODEL 1b – OLS Z EFEKTAMI STALYMI WOJEWÓDZTW (FE)")
print("=" * 60)
print("  Dodano zmienne zero-jedynkowe dla 15 wojewodztw (dolnoslaskie = baza).")
print("  Efekty stale kontroluja stale roznice miedzy wojewodztwami")
print("  (np. struktura przemyslu, klimat regionalny).")
print(model_fe.summary())

# ── 7. VIF – POOLED OLS ──────────────────────────────────────
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

print(f"  R²              = {R2:.4f}")
print(f"  R² adj.         = {R2_adj:.4f}")
print(f"  AIC             = {AIC_p:.3f}")
print(f"  BIC             = {BIC_p:.3f}")
print(f"  F-statystyka    = {F_stat:.3f}  (p = {F_pval:.6f})")
print(f"  n = {n_pool},  k = {k_pool}")

print("\n  Istotność parametrów:")
for name, coef, pv in zip(model_pool.params.index,
                           model_pool.params, model_pool.pvalues):
    if not name.startswith("woj_"):
        sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else
              ("*" if pv < 0.1 else ""))
        print(f"    {name:<22} β = {coef:+.4f}   p = {pv:.4f}  {sig}")

vif_data = pd.DataFrame({
    "Zmienna": X_cols,
    "VIF": [variance_inflation_factor(X_pool.values, i+1)
            for i in range(len(X_cols))]
})
print("\n  VIF (Pooled OLS):")
print(vif_data.to_string(index=False))

# ── 8. WERYFIKACJA STOCHASTYCZNA – POOLED OLS ─────────────────
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
print("  sa normalne. W razie wykrycia zalecane bledy standardowe HAC")
print("  lub PCSE (panel-corrected standard errors).")

# ── 9. MODELE OLS PER WOJEWÓDZTWO ────────────────────────────
print("\n" + "=" * 60)
print("MODELE OLS PER WOJEWÓDZTWO")
print("=" * 60)
print("  Specyfikacja per wojewodztwo:")
print("    ln(ZUZYCIE_t) = b0 + b1*ln(PKB_pc_t) + b2*ln(CENA_t)")
print("                  + b3*URBANIZACJA_t + b4*HDD_t + b5*CDD_t + e")
print("  n = 21 obserwacji na wojewodztwo")

prov_models = {}
prov_results = []

for prov in PROVINCES:
    dp = df[df["wojewodztwo"] == prov].copy()
    y_p = dp["ln_zuzycie"]
    X_p = sm.add_constant(dp[X_cols])
    mdl = sm.OLS(y_p, X_p).fit()
    prov_models[prov] = mdl

    sw_p_v  = shapiro(mdl.resid)[1]
    dw_v    = durbin_watson(mdl.resid)
    bg_p_v  = acorr_breusch_godfrey(mdl, nlags=2)[1]
    bp_p_v  = het_breuschpagan(mdl.resid, X_p)[1]

    prov_results.append({
        "województwo": prov,
        "R²":          round(mdl.rsquared, 4),
        "R²_adj":      round(mdl.rsquared_adj, 4),
        "AIC":         round(mdl.aic, 2),
        "β_pkb":       round(mdl.params["ln_pkb_pc"], 3),
        "β_cena":      round(mdl.params["ln_cena"], 3),
        "β_urban":     round(mdl.params["urbanizacja_pct"], 4),
        "DW":          round(dw_v, 3),
        "BG_p":        round(bg_p_v, 3),
        "SW_p":        round(sw_p_v, 3),
    })

    print(f"  {prov.upper():<25}  R²={mdl.rsquared:.3f}  "
          f"β_pkb={mdl.params['ln_pkb_pc']:+.3f}  "
          f"β_cena={mdl.params['ln_cena']:+.3f}  "
          f"DW={dw_v:.2f}  BG_p={bg_p_v:.3f}")

df_prov = pd.DataFrame(prov_results)

# ── 10. ZBIORCZY WYKRES WSPÓŁCZYNNIKÓW ELASTYCZNOŚCI ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Elastyczność dochodowa
axes[0].barh(df_prov["województwo"], df_prov["β_pkb"],
             color=[GREEN if v > 0 else RED for v in df_prov["β_pkb"]],
             alpha=0.85, edgecolor="white")
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_title("Elastyczność dochodowa β₁ (ln_pkb_pc)\nper województwo",
                  fontweight="bold")
axes[0].set_xlabel("β₁")

# Elastyczność cenowa
axes[1].barh(df_prov["województwo"], df_prov["β_cena"],
             color=[GREEN if v > 0 else RED for v in df_prov["β_cena"]],
             alpha=0.85, edgecolor="white")
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Elastyczność cenowa β₂ (ln_cena)\nper województwo",
                  fontweight="bold")
axes[1].set_xlabel("β₂")

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w05_elastycznosci_woj.png"),
            bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: w05_elastycznosci_woj.png")

# ── 11. WYKRESY DIAGNOSTYCZNE – POOLED OLS ───────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Diagnostyka – Pooled OLS (cały panel)", fontsize=14,
             fontweight="bold")

fitted_pool = model_pool.fittedvalues

ax = axes[0, 0]
ax.scatter(fitted_pool, res_pool, color=BLUE, alpha=0.4, s=20)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Wartości dopasowane (ln)"); ax.set_ylabel("Reszty")
ax.set_title("Reszty vs Dopasowane")

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
actual_gwh = np.exp(y_pool)
fitted_gwh = np.exp(fitted_pool)
ax.scatter(actual_gwh, fitted_gwh, color=BLUE, alpha=0.4, s=20)
mn = min(actual_gwh.min(), fitted_gwh.min())
mx = max(actual_gwh.max(), fitted_gwh.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Idealne dopasowanie")
ax.set_xlabel("Rzeczywiste [GWh]"); ax.set_ylabel("Dopasowane [GWh]")
ax.set_title("Rzeczywiste vs Dopasowane")
ax.legend()

ax = axes[1, 2]
cusum = np.cumsum(res_pool.values)
ax.plot(range(len(cusum)), cusum, color=PURPLE, linewidth=1.5)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
std_r = res_pool.std()
ax.axhline(+2 * std_r * np.sqrt(n_pool), color="red", linestyle=":", label="±2σ√n")
ax.axhline(-2 * std_r * np.sqrt(n_pool), color="red", linestyle=":")
ax.set_title("CUSUM reszt")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w06_diagnostyka.png"), bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: w06_diagnostyka.png")

# ── 12. INTERPRETACJA POOLED OLS ─────────────────────────────
print("\n" + "=" * 60)
print("INTERPRETACJA – POOLED OLS")
print("=" * 60)
params_p = model_pool.params
pvals_p  = model_pool.pvalues
print(f"  b1 (ln_pkb_pc)   = {params_p['ln_pkb_pc']:+.4f}")
print(f"    -> wzrost PKB per capita o 1% => zmiana zuzycia o {params_p['ln_pkb_pc']:+.2f}%")
print(f"  b2 (ln_cena)     = {params_p['ln_cena']:+.4f}")
print(f"    -> wzrost ceny o 1% => zmiana zuzycia o {params_p['ln_cena']:+.2f}%")
print(f"  b3 (urbanizacja) = {params_p['urbanizacja_pct']:+.5f}")
print(f"    -> wzrost urbanizacji o 1pp => zmiana ln(zuzycie) o {params_p['urbanizacja_pct']:+.5f}")
print(f"  b4 (HDD)         = {params_p['hdd']:+.6f}")
print(f"  b5 (CDD)         = {params_p['cdd']:+.6f}")

# ── 13. ARIMA PER WOJEWÓDZTWO ─────────────────────────────────
if PMDARIMA_OK:
    print("\n" + "=" * 60)
    print("ARIMA PER WOJEWÓDZTWO")
    print("=" * 60)

    arima_models = {}
    for prov in PROVINCES:
        dp = df[df["wojewodztwo"] == prov].copy().sort_values("rok")
        y_ts = dp["ln_zuzycie"].values
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
    zuzycie_energii_GWh  = ("zuzycie_energii_GWh", "sum"),
    ludnosc              = ("ludnosc", "sum"),
    pkb_mln_zl           = ("pkb_mln_zl", "sum"),
    cena_energii_zl_kWh  = ("cena_energii_zl_kWh", "mean"),
    urbanizacja_pct      = ("urbanizacja_pct", "mean"),
    hdd                  = ("hdd", "mean"),
    cdd                  = ("cdd", "mean"),
).reset_index())

df_nat["pkb_per_capita"] = df_nat["pkb_mln_zl"] * 1e6 / df_nat["ludnosc"]
df_nat["ln_pkb_pc"]      = np.log(df_nat["pkb_per_capita"])
df_nat["ln_zuzycie"]     = np.log(df_nat["zuzycie_energii_GWh"])
df_nat["ln_cena"]        = np.log(df_nat["cena_energii_zl_kWh"])

# Model OLS na agregacie
y_nat = df_nat["ln_zuzycie"]
X_nat = sm.add_constant(df_nat[X_cols])
model_nat = sm.OLS(y_nat, X_nat).fit()

print(f"\n  Model OLS na agregacie krajowym (n={len(df_nat)}):")
print(f"  R² = {model_nat.rsquared:.4f},  R²adj = {model_nat.rsquared_adj:.4f}")
print(f"  AIC = {model_nat.aic:.2f},  BIC = {model_nat.bic:.2f}")

# Scenariusze
forecast_years = list(range(2025, 2031))
last = df_nat[df_nat["rok"] == df_nat["rok"].max()].iloc[0]

scenarios = {
    "Pesymistyczny": dict(pkb=0.015, cena=0.06, urban=0.10, hdd=3100, cdd=25,
                          color=RED,   ls="--"),
    "Bazowy":        dict(pkb=0.030, cena=0.03, urban=0.20, hdd=2900, cdd=35,
                          color=BLUE,  ls="-"),
    "Optymistyczny": dict(pkb=0.045, cena=0.01, urban=0.30, hdd=2700, cdd=50,
                          color=GREEN, ls="-."),
}

fc_nat = {}
for sc_name, sc in scenarios.items():
    vals, lo_vals, hi_vals = [], [], []
    for i, yr in enumerate(forecast_years, 1):
        x_pred = pd.DataFrame({
            "const":           [1.0],
            "ln_pkb_pc":       [np.log(last["pkb_per_capita"] * (1+sc["pkb"])**i)],
            "ln_cena":         [np.log(last["cena_energii_zl_kWh"] * (1+sc["cena"])**i)],
            "urbanizacja_pct": [last["urbanizacja_pct"] + sc["urban"]*i],
            "hdd":             [float(sc["hdd"])],
            "cdd":             [float(sc["cdd"])],
        })
        pr    = model_nat.get_prediction(x_pred).summary_frame(alpha=0.05)
        vals.append(np.exp(pr["mean"].values[0]))
        lo_vals.append(np.exp(pr["mean_ci_lower"].values[0]))
        hi_vals.append(np.exp(pr["mean_ci_upper"].values[0]))

    fc_nat[sc_name] = dict(years=forecast_years, mean=vals,
                            lo=lo_vals, hi=hi_vals,
                            color=sc["color"], ls=sc["ls"])

    print(f"\n  Scenariusz: {sc_name}")
    for yr, gw, lo, hi in zip(forecast_years, vals, lo_vals, hi_vals):
        print(f"    {yr}: {gw:,.0f} GWh  [95% CI: {lo:,.0f}–{hi:,.0f}]")

# Udziały regionalne (2024)
yr_last = df["rok"].max()
last_prov = df[df["rok"] == yr_last][["wojewodztwo", "zuzycie_energii_GWh"]].copy()
total_2024 = last_prov["zuzycie_energii_GWh"].sum()
last_prov["udzial"] = last_prov["zuzycie_energii_GWh"] / total_2024

print("\n" + "=" * 60)
print("DEKOMPOZYCJA PROGNOZY NA UDZIALY REGIONALNE (scen. bazowy)")
print("=" * 60)
print(f"  {'Województwo':<25} {'Udział 2024':>12}  " +
      "  ".join(str(y) for y in forecast_years))
for _, row in last_prov.sort_values("udzial", ascending=False).iterrows():
    fc_prov = [f"{row['udzial'] * gw:,.0f}" for gw in fc_nat["Bazowy"]["mean"]]
    print(f"  {row['wojewodztwo']:<25} {row['udzial']*100:>10.1f}%  " +
          "  ".join(fc_prov))

# ── 15. WYKRES PROGNOZY ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df_nat["rok"], df_nat["zuzycie_energii_GWh"],
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
    "Prognoza zużycia energii elektrycznej – agregat z 16 województw 2025–2030\n"
    "Model OLS log-liniowy z czynnikami ekonomicznymi i klimatycznymi",
    fontsize=13, fontweight="bold")
ax.set_xlabel("Rok", fontsize=11)
ax.set_ylabel("Zużycie energii [GWh]", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xlim(2003, 2031)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w07_prognoza.png"), bbox_inches="tight")
print("Zapisano: w07_prognoza.png")

# ── 16. WYKRES MAP – UDZIAŁY REGIONALNE ──────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
df_share = last_prov.sort_values("udzial", ascending=True)
bars = ax.barh(df_share["wojewodztwo"], df_share["udzial"] * 100,
               color=BLUE, alpha=0.85, edgecolor="white")
ax.set_xlabel("Udział w krajowym zużyciu energii [%]")
ax.set_title(f"Udziały województw w zużyciu energii elektrycznej ({yr_last})",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, df_share["udzial"] * 100):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "w08_udzialy_regionalne.png"),
            bbox_inches="tight")
print("Zapisano: w08_udzialy_regionalne.png")

# ── 17. PODSUMOWANIE ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PODSUMOWANIE")
print("=" * 60)
print(f"  Dane          : panel 16 wojewodztw x 21 lat = 336 obserwacji")
print(f"  Model pooled  : ln(ZUZYCIE) ~ ln(PKB_pc) + ln(CENA) + URBANIZACJA + HDD + CDD")
print(f"  R2 (pooled)   : {R2:.4f}")
print(f"  R2 (FE)       : {model_fe.rsquared:.4f}")
print(f"  Elastycznosc dochodowa (pooled) : {params_p['ln_pkb_pc']:+.3f}")
print(f"  Elastycznosc cenowa    (pooled) : {params_p['ln_cena']:+.3f}")
print(f"  Weryfikacja stochastyczna (pooled):")
print(f"    Normalnosc reszt (SW) : p = {p_sw:.4f}")
print(f"    Autokorelacja (BG)    : p = {bg_p:.4f}")
print(f"    Heteroskedastycznosc  : p = {bp_p:.4f}")
print(f"  Prognoza bazowa 2030 (agregat): {fc_nat['Bazowy']['mean'][-1]:,.0f} GWh")
print(f"  (2024: {total_2024:,.0f} GWh)")
print("  Wygenerowane pliki: w01..w08_*.png")
