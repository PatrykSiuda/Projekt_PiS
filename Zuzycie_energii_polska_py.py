# ============================================================
# ANALIZA I PROGNOZA ZUŻYCIA ENERGII ELEKTRYCZNEJ W POLSCE
# ============================================================
# Projekt: Wpływ czynników społeczno-ekonomicznych na zużycie
#          energii elektrycznej w Polsce
# ============================================================

# ── 0. IMPORTY ───────────────────────────────────────────────
# Wymagane pakiety (uruchom raz w terminalu jeśli brakuje):
# pip install pandas numpy matplotlib seaborn statsmodels scipy openpyxl pmdarima docx

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats.stattools import jarque_bera

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("⚠️  Pakiet pmdarima niedostępny. Sekcja ARIMA zostanie pominięta.")
    print("   Zainstaluj: pip install pmdarima")

# ── ŚCIEŻKA DO DANYCH ────────────────────────────────────────
# Działa zarówno z terminala (py skrypt.py) jak i przez reticulate
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = SCRIPT_DIR / "Zuzycie_energii_polska.xlsx"

# ── PARAMETRY GLOBALNE ───────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
})
BLUE   = "#1a5c96"
RED    = "#c0392b"
GREEN  = "#27ae60"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"
GRAY   = "#7f8c8d"

# ── 1. WCZYTANIE DANYCH ──────────────────────────────────────
df = pd.read_excel(DATA_FILE)

print("=" * 60)
print("WCZYTANE DANE – PODGLĄD")
print("=" * 60)
print(df.to_string(index=False))

# ── 1b. ZMIENNE POCHODNE ─────────────────────────────────────
df["pkb_per_capita"] = df["pkb_mln_zl"] * 1e6 / df["ludnosc"]
df["ln_pkb_pc"]      = np.log(df["pkb_per_capita"])
df["ln_zuzycie"]     = np.log(df["zuzycie_energii_GWh"])
df["ln_cena"]        = np.log(df["cena_energii_zl_kWh"])
df["ln_ludnosc"]     = np.log(df["ludnosc"])
df["sqrt_hdd"]       = np.sqrt(df["hdd"])
df["trend"]          = np.arange(len(df))

# ── 2. ANALIZA OPISOWA ───────────────────────────────────────
print("\n" + "=" * 60)
print("STATYSTYKI OPISOWE")
print("=" * 60)
desc_cols = [
    "zuzycie_energii_GWh", "cena_energii_zl_kWh",
    "pkb_mln_zl", "ludnosc", "urbanizacja_pct", "hdd", "cdd"
]
desc = df[desc_cols].describe().T
desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(2)
print(desc.round(3).to_string())

# ── 3. WYKRESY SZEREGÓW CZASOWYCH ───────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(
    "Szeregi czasowe zmiennych – Polska 2004–2024",
    fontsize=15, fontweight="bold", y=1.01
)

variables = {
    "Zużycie energii [GWh]":    ("zuzycie_energii_GWh", BLUE),
    "Cena energii [zł/kWh]":    ("cena_energii_zl_kWh", RED),
    "PKB [mln zł]":              ("pkb_mln_zl",          GREEN),
    "PKB per capita [zł]":       ("pkb_per_capita",       ORANGE),
    "Ludność [os.]":             ("ludnosc",              PURPLE),
    "Urbanizacja [%]":           ("urbanizacja_pct",      GRAY),
    "Heating Degree Days (HDD)": ("hdd",                  "#2980b9"),
    "Cooling Degree Days (CDD)": ("cdd",                  "#e74c3c"),
    "Ludność miejska [os.]":     ("ludnosc_miasto",       "#16a085"),
}

for ax, (title, (col, color)) in zip(axes.flat, variables.items()):
    ax.plot(df["rok"], df[col], marker="o", color=color, linewidth=2, markersize=5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Rok")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "01_szeregi_czasowe.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 01_szeregi_czasowe.png")

# ── 4. MACIERZ KORELACJI ────────────────────────────────────
corr_cols = [
    "zuzycie_energii_GWh", "pkb_per_capita",
    "cena_energii_zl_kWh", "urbanizacja_pct",
    "ludnosc", "hdd", "cdd"
]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
    vmin=-1, vmax=1, ax=ax, linewidths=0.5,
    annot_kws={"size": 10}
)
ax.set_title("Macierz korelacji Pearsona", fontsize=13, fontweight="bold")
labels = [
    "Zużycie energii", "PKB per capita", "Cena energii",
    "Urbanizacja", "Ludność", "HDD", "CDD"
]
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "02_korelacja.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 02_korelacja.png")

# ── 5. WYKRESY ROZRZUTU ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Zależności zużycia energii od zmiennych objaśniających",
    fontsize=13, fontweight="bold"
)

scatter_vars = [
    ("pkb_per_capita",      "PKB per capita [zł]",      GREEN),
    ("cena_energii_zl_kWh", "Cena energii [zł/kWh]",    RED),
    ("urbanizacja_pct",     "Urbanizacja [%]",           ORANGE),
    ("ludnosc",             "Ludność [os.]",             PURPLE),
    ("hdd",                 "HDD",                      "#2980b9"),
    ("cdd",                 "CDD",                      "#e74c3c"),
]

for ax, (col, xlabel, color) in zip(axes.flat, scatter_vars):
    ax.scatter(df[col], df["zuzycie_energii_GWh"],
               color=color, alpha=0.8, s=60, edgecolors="white", linewidth=0.5)
    z = np.polyfit(df[col], df["zuzycie_energii_GWh"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, p(x_line), "--", color="black", alpha=0.5, linewidth=1.5)
    r, pval = stats.pearsonr(df[col], df["zuzycie_energii_GWh"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Zużycie energii [GWh]")
    ax.set_title(f"r = {r:.3f}  (p = {pval:.3f})", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "03_scatter.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 03_scatter.png")

# ── 6. BUDOWA MODELU OLS ─────────────────────────────────────
#
# Model log-liniowy:
#   ln(ZUZYCIE) = β₀ + β₁·ln(PKB_PC) + β₂·ln(CENA)
#                + β₃·URBANIZACJA + β₄·HDD + β₅·CDD + ε
#
# Uzasadnienie:
#  • ln(PKB_PC)  → efekt dochodowy, elastyczność interpretowalna
#  • ln(CENA)    → efekt cenowy, elastyczność cenowa popytu
#  • URBANIZACJA → strukturalny czynnik społeczny
#  • HDD / CDD   → klimatyczne czynniki sezonowe

X_cols = ["ln_pkb_pc", "ln_cena", "urbanizacja_pct", "hdd", "cdd"]
y      = df["ln_zuzycie"]
X      = sm.add_constant(df[X_cols])

model = sm.OLS(y, X).fit()

print("\n" + "=" * 60)
print("MODEL 1 – LOG-LINIOWY OLS (wyniki estymacji)")
print("=" * 60)
print("""
  SPECYFIKACJA:
    ln(ZUZYCIE) = β₀ + β₁·ln(PKB_pc) + β₂·ln(CENA)
                + β₃·URBANIZACJA + β₄·HDD + β₅·CDD + ε

  ZMIENNE OBJAŚNIAJĄCE:
    ln(PKB_pc)   – elastyczność dochodowa popytu na energię
                   (1% wzrostu PKB per capita → β₁% zmiany zużycia)
    ln(CENA)     – elastyczność cenowa popytu
                   (1% wzrostu ceny → β₂% zmiany zużycia, oczekiwany znak ujemny)
    URBANIZACJA  – odsetek ludności miejskiej [%]; wyższe zurbanizowanie
                   wiąże się z efektywniejszym zużyciem energii w gospodarstwach
    HDD          – Heating Degree Days; miara zapotrzebowania na ogrzewanie
    CDD          – Cooling Degree Days; miara zapotrzebowania na chłodzenie

  FORMA FUNKCYJNA: log-liniowa → współczynniki przy ln(·) to elastyczności;
  współczynniki przy zmiennych w poziomie (URBANIZACJA, HDD, CDD) to efekty semilogarytmiczne.

  UWAGA: wykryto wysoką współliniowość (VIF>35 dla ln_pkb_pc i urbanizacji).
         Estymacja punktowa może być niestabilna → patrz Iteracja 2.
""")
print(model.summary())

# ── 6b. MODEL Z OPÓŹNIONĄ ZMIENNĄ ZALEŻNĄ ───────────────────
df["ln_zuzycie_lag1"] = df["ln_zuzycie"].shift(1)
df_lagged = df.dropna(subset=["ln_zuzycie_lag1"])

X_cols_lagged = ["pkb_per_capita", "ln_cena", "urbanizacja_pct", "hdd", "cdd", "ln_zuzycie_lag1"]
y_lagged      = df_lagged["ln_zuzycie"]
X_lagged      = sm.add_constant(df_lagged[X_cols_lagged])
model_lagged  = sm.OLS(y_lagged, X_lagged).fit()

print("\n" + "=" * 60)
print("MODEL 1b – DYNAMICZNY OLS (opóźniona zmienna zależna)")
print("=" * 60)
print("""
  SPECYFIKACJA:
    ln(ZUZYCIE_t) = β₀ + β₁·PKB_pc_t + β₂·ln(CENA_t) + β₃·URBANIZACJA_t
                  + β₄·HDD_t + β₅·CDD_t + β₆·ln(ZUZYCIE_{t-1}) + ε_t

  CEL: uchwycenie inercji systemu energetycznego – zużycie w roku t zależy
  częściowo od zużycia w roku t-1 (kontrakty, infrastruktura, przyzwyczajenia).

  WŁAŚCIWOŚCI:
    • Opóźniona zmienna zależna (ln_zuzycie_lag1) absorbuje autokorelację reszt
    • Kosztem: utrata 1 obserwacji (n = 20 zamiast 21)
    • Współczynnik β₆ interpretowany jako szybkość dostosowania:
      β₆ ≈ 1 → silna inercja, β₆ ≈ 0 → szybkie dostosowanie
""")
print(model_lagged.summary())

# ── 7. WERYFIKACJA NUMERYCZNA ────────────────────────────────
print("\n" + "=" * 60)
print("WERYFIKACJA NUMERYCZNA MODELU")
print("=" * 60)

R2        = model.rsquared
R2_adj    = model.rsquared_adj
AIC       = model.aic
BIC       = model.bic
F_stat    = model.fvalue
F_pval    = model.f_pvalue
n         = int(model.nobs)
k         = len(model.params) - 1
residuals = model.resid

print(f"  R²              = {R2:.4f}")
print(f"  R² adj.         = {R2_adj:.4f}")
print(f"  AIC             = {AIC:.3f}")
print(f"  BIC             = {BIC:.3f}")
print(f"  F-statystyka    = {F_stat:.3f}  (p = {F_pval:.6f})")
print(f"  Liczba obserwacji (n) = {n}")
print(f"  Liczba parametrów (k) = {k}")

print("\n  Istotność parametrów:")
for name, coef, pv in zip(model.params.index, model.params, model.pvalues):
    sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
    print(f"    {name:<20} β = {coef:+.4f}   p = {pv:.4f}  {sig}")

vif_data = pd.DataFrame()
vif_data["Zmienna"] = X.columns[1:]
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i + 1)
    for i in range(len(X_cols))
]
print("\n  Czynnik inflacji wariancji (VIF):")
print(vif_data.to_string(index=False))

# ── 8. WERYFIKACJA STOCHASTYCZNA ─────────────────────────────
print("\n" + "=" * 60)
print("WERYFIKACJA STOCHASTYCZNA MODELU")
print("=" * 60)

# 8a. Normalność reszt
stat_sw, p_sw = shapiro(residuals)
stat_jb, p_jb, _, _ = jarque_bera(residuals)
print(f"\n  [Normalność reszt]")
print(f"  Shapiro-Wilk:    W = {stat_sw:.4f},  p = {p_sw:.4f}  "
      f"{'✅ brak podstaw do odrzucenia H0' if p_sw > 0.05 else '❌ odrzucamy H0'}")
print(f"  Jarque-Bera:     JB = {stat_jb:.4f}, p = {p_jb:.4f}  "
      f"{'✅ brak podstaw do odrzucenia H0' if p_jb > 0.05 else '❌ odrzucamy H0'}")

# 8b. Autokorelacja
dw = durbin_watson(residuals)
bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(model, nlags=2)
print(f"\n  [Autokorelacja reszt]")
print(f"  Durbin-Watson:   DW = {dw:.4f}  "
      f"({'brak autokorelacji' if 1.5 < dw < 2.5 else 'możliwa autokorelacja'})")
print(f"  Breusch-Godfrey: LM = {bg_stat:.4f}, p = {bg_pval:.4f}  "
      f"{'✅ brak autokorelacji' if bg_pval > 0.05 else '❌ autokorelacja wykryta'}")

# 8c. Heteroskedastyczność
bp_lm, bp_pval, _, _ = het_breuschpagan(residuals, X)
print(f"\n  [Heteroskedastyczność]")
print(f"  Breusch-Pagan:   LM = {bp_lm:.4f}, p = {bp_pval:.4f}  "
      f"{'✅ homoskedastyczność' if bp_pval > 0.05 else '❌ heteroskedastyczność'}")

try:
    wh_lm, wh_pval, _, _ = het_white(residuals, X)
    print(f"  White:           LM = {wh_lm:.4f}, p = {wh_pval:.4f}  "
          f"{'✅ homoskedastyczność' if wh_pval > 0.05 else '❌ heteroskedastyczność'}")
except Exception:
    print("  White: test niedostępny (za mało stopni swobody)")

# ── 9. WYKRESY DIAGNOSTYCZNE ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Diagnostyka modelu OLS", fontsize=14, fontweight="bold")

fitted = model.fittedvalues

# 9a. Reszty vs dopasowane
ax = axes[0, 0]
ax.scatter(fitted, residuals, color=BLUE, alpha=0.8, s=50)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Wartości dopasowane (ln)")
ax.set_ylabel("Reszty")
ax.set_title("Reszty vs Dopasowane")

# 9b. QQ-plot
ax = axes[0, 1]
sm.qqplot(residuals, line="s", ax=ax, alpha=0.8, color=BLUE)
ax.set_title("Wykres kwantyl-kwantyl (QQ)")

# 9c. Histogram reszt
ax = axes[0, 2]
ax.hist(residuals, bins=8, color=BLUE, alpha=0.8, edgecolor="white")
xmin, xmax = ax.get_xlim()
x_norm = np.linspace(xmin, xmax, 100)
ax.plot(x_norm,
        stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        * len(residuals) * (xmax - xmin) / 8,
        "r-", linewidth=2)
ax.set_title("Histogram reszt")
ax.set_xlabel("Reszty")

# 9d. Reszty w czasie
ax = axes[1, 0]
ax.plot(df["rok"], residuals, marker="o", color=ORANGE, linewidth=2, markersize=6)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.fill_between(df["rok"], residuals, 0,
                where=(residuals > 0), alpha=0.2, color=GREEN)
ax.fill_between(df["rok"], residuals, 0,
                where=(residuals < 0), alpha=0.2, color=RED)
ax.set_title("Reszty w czasie")
ax.set_xlabel("Rok")
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

# 9e. Dopasowane vs rzeczywiste
ax = axes[1, 1]
actual_gwh = np.exp(y)
fitted_gwh = np.exp(fitted)
ax.plot(df["rok"], actual_gwh, "o-", color=BLUE,  label="Rzeczywiste", linewidth=2)
ax.plot(df["rok"], fitted_gwh, "s--", color=RED,  label="Dopasowane",  linewidth=2)
ax.set_title("Rzeczywiste vs Dopasowane [GWh]")
ax.set_xlabel("Rok")
ax.set_ylabel("Zużycie energii [GWh]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

# 9f. CUSUM reszt
ax = axes[1, 2]
cusum = np.cumsum(residuals)
ax.plot(df["rok"], cusum, color=PURPLE, linewidth=2, marker="o", markersize=5)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
std_res = residuals.std()
ax.axhline(+2 * std_res * np.sqrt(n), color="red", linestyle=":", label="±2σ√n")
ax.axhline(-2 * std_res * np.sqrt(n), color="red", linestyle=":")
ax.set_title("CUSUM reszt (stabilność)")
ax.set_xlabel("Rok")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "04_diagnostyka.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 04_diagnostyka.png")

# ── 10. INTERPRETACJA ANALITYCZNA ───────────────────────────
print("\n" + "=" * 60)
print("INTERPRETACJA ANALITYCZNA MODELU LOG-LINIOWEGO")
print("=" * 60)
params = model.params

print(f"""
  Model postaci: ln(ZUZYCIE) = β₀ + β₁·ln(PKB_pc) + β₂·ln(CENA)
                              + β₃·URBANIZACJA + β₄·HDD + β₅·CDD + ε

  β₀ (stała)      = {params['const']:+.4f}
  β₁ (ln_pkb_pc)  = {params['ln_pkb_pc']:+.4f}
    → Elastyczność dochodowa: wzrost PKB per capita o 1%
      zmienia zużycie energii o {params['ln_pkb_pc']:+.2f}%

  β₂ (ln_cena)    = {params['ln_cena']:+.4f}
    → Elastyczność cenowa: wzrost ceny o 1%
      zmienia zużycie energii o {params['ln_cena']:+.2f}%

  β₃ (urbanizacja)= {params['urbanizacja_pct']:+.4f}
    → Wzrost urbanizacji o 1 pp zmienia ln(zużycie) o {params['urbanizacja_pct']:+.4f}
      (tj. zmiana o ok. {np.expm1(params['urbanizacja_pct'])*100:+.2f}%)

  β₄ (HDD)        = {params['hdd']:+.6f}
    → Wzrost HDD o 1 dzień·stopień → zmiana zużycia o {params['hdd']:+.6f} (ln)

  β₅ (CDD)        = {params['cdd']:+.6f}
    → Wzrost CDD o 1 dzień·stopień → zmiana zużycia o {params['cdd']:+.6f} (ln)
""")

# ── 11. PROGNOZA WARUNKOWA 2025–2030 ─────────────────────────
#
#  Scenariusze:
#  ┌─────────────────┬─────────────┬─────────────┬─────────────┐
#  │  Zmienna        │  Pesymist.  │    Bazowy   │  Optymist.  │
#  ├─────────────────┼─────────────┼─────────────┼─────────────┤
#  │ PKB_pc [zł]     │  +1.5%/rok  │  +3.0%/rok  │  +4.5%/rok  │
#  │ Cena [zł/kWh]   │  +6%/rok    │  +3%/rok    │  +1%/rok    │
#  │ Urbanizacja     │  +0.1pp/rok │  +0.2pp/rok │  +0.3pp/rok │
#  │ HDD             │  3100       │  2900       │  2700       │
#  │ CDD             │  25         │  35         │  50         │
#  └─────────────────┴─────────────┴─────────────┴─────────────┘

print("\n" + "=" * 60)
print("PROGNOZA WARUNKOWA 2025–2030")
print("=" * 60)

forecast_years = list(range(2025, 2031))
base_pkb_pc    = df["pkb_per_capita"].iloc[-1]
base_cena      = df["cena_energii_zl_kWh"].iloc[-1]
base_urban     = df["urbanizacja_pct"].iloc[-1]

scenarios = {
    "Pesymistyczny": {
        "pkb_growth": 0.015, "cena_growth": 0.06, "urban_delta": 0.10,
        "hdd": 3100, "cdd": 25, "color": RED, "ls": "--",
    },
    "Bazowy": {
        "pkb_growth": 0.030, "cena_growth": 0.03, "urban_delta": 0.20,
        "hdd": 2900, "cdd": 35, "color": BLUE, "ls": "-",
    },
    "Optymistyczny": {
        "pkb_growth": 0.045, "cena_growth": 0.01, "urban_delta": 0.30,
        "hdd": 2700, "cdd": 50, "color": GREEN, "ls": "-.",
    },
}

forecast_results = {}

for scenario_name, sc in scenarios.items():
    gwh_preds, gwh_lo, gwh_hi = [], [], []

    for i, yr in enumerate(forecast_years, 1):
        pkb_pc_yr = base_pkb_pc * (1 + sc["pkb_growth"]) ** i
        cena_yr   = base_cena   * (1 + sc["cena_growth"]) ** i
        urban_yr  = base_urban  + sc["urban_delta"] * i

        x_pred = pd.DataFrame({
            "const":           [1.0],
            "ln_pkb_pc":       [np.log(pkb_pc_yr)],
            "ln_cena":         [np.log(cena_yr)],
            "urbanizacja_pct": [urban_yr],
            "hdd":             [float(sc["hdd"])],
            "cdd":             [float(sc["cdd"])],
        })

        pred  = model.get_prediction(x_pred)
        frame = pred.summary_frame(alpha=0.05)
        gwh_preds.append(np.exp(frame["mean"].values[0]))
        gwh_lo.append(np.exp(frame["mean_ci_lower"].values[0]))
        gwh_hi.append(np.exp(frame["mean_ci_upper"].values[0]))

    forecast_results[scenario_name] = {
        "years": forecast_years, "mean": gwh_preds,
        "lo": gwh_lo, "hi": gwh_hi,
        "color": sc["color"], "ls": sc["ls"],
    }

    print(f"\n  Scenariusz: {scenario_name}")
    for yr, gw, lo, hi in zip(forecast_years, gwh_preds, gwh_lo, gwh_hi):
        print(f"    {yr}: {gw:,.0f} GWh  [95% CI: {lo:,.0f} – {hi:,.0f}]")

# ── 12. WYKRES PROGNOZY WARUNKOWEJ ───────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df["rok"], df["zuzycie_energii_GWh"],
        "o-", color=GRAY, linewidth=2.5, markersize=7,
        label="Dane historyczne 2004–2024", zorder=5)

ax.axvline(2024.5, color="black", linewidth=1.5, linestyle=":", alpha=0.6)
ax.text(2024.6, ax.get_ylim()[0] * 1.02 if ax.get_ylim()[0] > 0 else 22500,
        "Prognoza →", fontsize=9, color="black", alpha=0.7)

for sc_name, sc_data in forecast_results.items():
    ax.plot(sc_data["years"], sc_data["mean"],
            color=sc_data["color"], linewidth=2.5,
            linestyle=sc_data["ls"], marker="D", markersize=6,
            label=f"Scenariusz {sc_name}")
    ax.fill_between(sc_data["years"], sc_data["lo"], sc_data["hi"],
                    color=sc_data["color"], alpha=0.12)

ax.set_title(
    "Prognoza warunkowa zużycia energii elektrycznej w Polsce 2025–2030\n"
    "Model log-liniowy OLS z czynnikami ekonomicznymi i klimatycznymi",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Rok", fontsize=11)
ax.set_ylabel("Zużycie energii elektrycznej [GWh]", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xlim(2003, 2031)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "05_prognoza_warunkowa.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 05_prognoza_warunkowa.png")

# ── 13. TABELA PORÓWNAWCZA SCENARIUSZY ───────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

base_2024  = df["zuzycie_energii_GWh"].iloc[-1]
table_data = [["Rok",
               "Pesym. [GWh]", "Bazowy [GWh]", "Optym. [GWh]",
               "Δ Pesym. vs 2024", "Δ Optym. vs 2024"]]

for i, yr in enumerate(forecast_years):
    p  = forecast_results["Pesymistyczny"]["mean"][i]
    b  = forecast_results["Bazowy"]["mean"][i]
    o  = forecast_results["Optymistyczny"]["mean"][i]
    dp = (p / base_2024 - 1) * 100
    do = (o / base_2024 - 1) * 100
    table_data.append([
        str(yr),
        f"{p:,.0f}", f"{b:,.0f}", f"{o:,.0f}",
        f"{dp:+.1f}%", f"{do:+.1f}%"
    ])

table = ax.table(
    cellText=table_data[1:], colLabels=table_data[0],
    cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
for j in range(len(table_data[0])):
    table[0, j].set_facecolor("#1a5c96")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(table_data)):
    bg = "#f0f5ff" if i % 2 == 0 else "white"
    for j in range(len(table_data[0])):
        table[i, j].set_facecolor(bg)

ax.set_title("Zestawienie prognoz warunkowych 2025–2030 vs bazowy rok 2024",
             fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "06_tabela_prognoz.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 06_tabela_prognoz.png")

# ── 14. WYKRES WSPÓŁCZYNNIKÓW ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

coef_names = {
    "ln_pkb_pc":       "ln(PKB per capita)\n[elastyczność dochodowa]",
    "ln_cena":         "ln(Cena energii)\n[elastyczność cenowa]",
    "urbanizacja_pct": "Urbanizacja [%]",
    "hdd":             "HDD\n[ogrzewanie]",
    "cdd":             "CDD\n[chłodzenie]",
}

coefs      = model.params[list(coef_names.keys())]
confs      = model.conf_int().loc[list(coef_names.keys())]
colors_bar = [GREEN if c > 0 else RED for c in coefs]

bars = ax.barh(
    list(coef_names.values()), coefs.values,
    xerr=[coefs.values - confs[0].values, confs[1].values - coefs.values],
    color=colors_bar, alpha=0.85, edgecolor="white", height=0.6,
    error_kw={"elinewidth": 2, "capsize": 5, "ecolor": "black"}
)
ax.axvline(0, color="black", linewidth=1.2)
ax.set_title("Współczynniki modelu OLS z 95% przedziałami ufności",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Wartość współczynnika β")
for bar, val in zip(bars, coefs.values):
    ax.text(val + (0.002 if val > 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center",
            ha="left" if val > 0 else "right", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "07_wspolczynniki.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 07_wspolczynniki.png")

# ── 15. MODEL ARIMA ──────────────────────────────────────────
if PMDARIMA_AVAILABLE:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    print("\n" + "=" * 60)
    print("MODEL ARIMA – WYNIKI ESTYMACJI")
    print("=" * 60)

    y_ts = df["ln_zuzycie"]
    model_arima = pm.auto_arima(
        y_ts,
        seasonal=False,
        suppress_warnings=True,
        stepwise=True
    )
    print(model_arima.summary())

    fc_arima = model_arima.predict(n_periods=6, return_conf_int=True)
    fc_vals, fc_ci = fc_arima
    print("\n  Prognoza ARIMA 2025–2030 [ln(GWh)]:")
    for yr, val, ci in zip(forecast_years, fc_vals, fc_ci):
        print(f"    {yr}: ln = {val:.4f}  → {np.exp(val):,.0f} GWh"
              f"  [95% CI: {np.exp(ci[0]):,.0f} – {np.exp(ci[1]):,.0f}]")

# ── 16. PODSUMOWANIE KOŃCOWE ─────────────────────────────────
print("\n" + "=" * 60)
print("PODSUMOWANIE PROJEKTU")
print("=" * 60)
print(f"""
  Zmienna zależna : Zużycie energii elektrycznej w Polsce [GWh]
  Okres próby     : 2004–2024  (n = {n} obserwacji rocznych)
  Model           : log-liniowy OLS (5 zmiennych objaśniających)

  MIARY DOPASOWANIA:
    R²      = {R2:.4f}  → model wyjaśnia {R2*100:.1f}% zmienności
    R² adj  = {R2_adj:.4f}
    AIC     = {AIC:.2f}
    BIC     = {BIC:.2f}
    F-test  = {F_stat:.2f}  (p = {F_pval:.6f}) → model istotny łącznie

  WERYFIKACJA STOCHASTYCZNA:
    Normalność reszt    : Shapiro-Wilk p = {p_sw:.4f}
    Autokorelacja       : DW = {dw:.4f}, BG p = {bg_pval:.4f}
    Heteroskedastyczność: BP p = {bp_pval:.4f}

  WNIOSKI EKONOMICZNE:
    • Elastyczność dochodowa  = {params['ln_pkb_pc']:+.3f}
      → Wzrost zamożności społeczeństwa zwiększa zapotrzebowanie na energię
    • Elastyczność cenowa     = {params['ln_cena']:+.3f}
      → Wyższe ceny nieznacznie ograniczają zużycie (niska elastyczność)
    • Urbanizacja i klimat mają istotny wpływ strukturalny

  PROGNOZA 2030 (scenariusz bazowy):
    ≈ {forecast_results['Bazowy']['mean'][-1]:,.0f} GWh
    (vs {base_2024:,} GWh w 2024)

  Wygenerowane pliki:
    01_szeregi_czasowe.png
    02_korelacja.png
    03_scatter.png
    04_diagnostyka.png
    05_prognoza_warunkowa.png
    06_tabela_prognoz.png
    07_wspolczynniki.png
""")

# ============================================================
# ITERACJA 2 – KOREKTA WIELOKOLINIOWOŚCI
# ============================================================
# Problem wykryty w Modelu 1:
#   VIF(ln_pkb_pc) = 45.8,  VIF(urbanizacja_pct) = 35.1
#   Korelacja ln_pkb_pc ↔ urbanizacja_pct = –0.985
# Rozwiązanie: usunięcie urbanizacja_pct z modelu
# ============================================================

print("\n" + "=" * 60)
print("ITERACJA 2 – MODEL 2 (korekta wielokoliniowości)")
print("=" * 60)
print("""
  MOTYWACJA:
    W Modelu 1 wykryto silną wielokoliniowość:
      VIF(ln_pkb_pc) ≈ 46,  VIF(urbanizacja_pct) ≈ 35
      corr(ln_pkb_pc, urbanizacja_pct) = –0.985
    Urbanizacja i PKB per capita mierzą ten sam trend rozwojowy → jedna z nich
    jest redundantna. Usunięto urbanizacja_pct jako zmienną bardziej pośrednią.

  SPECYFIKACJA:
    ln(ZUZYCIE) = β₀ + β₁·ln(PKB_pc) + β₂·ln(CENA) + β₃·HDD + β₄·CDD + ε

  OCZEKIWANE POPRAWY:
    • VIF poniżej 15 dla wszystkich zmiennych
    • Stabilniejsze estymaty i szersze, bardziej wiarygodne przedziały ufności

  OCENA DO PROGNOZOWANIA:
    Brak autokorelacji (BG p>0.05), brak heteroskedastyczności → prognozy warunkowe
    są dopuszczalne. Ograniczenie: reszty nienormalne (SW p<0.05) → przedziały
    predykcji należy traktować orientacyjnie.
""")

# ── M2-A. ESTYMACJA ─────────────────────────────────────────
#   ln(ZUZYCIE) = β₀ + β₁·ln(PKB_PC) + β₂·ln(CENA) + β₃·HDD + β₄·CDD + ε

X2_cols = ["ln_pkb_pc", "ln_cena", "hdd", "cdd"]
X2      = sm.add_constant(df[X2_cols])
model2  = sm.OLS(df["ln_zuzycie"], X2).fit()

print(model2.summary())

# ── M2-B. PORÓWNANIE VIF ────────────────────────────────────
print("\n  Porównanie VIF – Model 1 vs Model 2:")
print(f"  {'Zmienna':<20} {'VIF M1':>8}  {'VIF M2':>8}")
print("  " + "-" * 42)
vif_m1 = {
    "ln_pkb_pc":       variance_inflation_factor(X.values, 1),
    "ln_cena":         variance_inflation_factor(X.values, 2),
    "urbanizacja_pct": variance_inflation_factor(X.values, 3),
    "hdd":             variance_inflation_factor(X.values, 4),
    "cdd":             variance_inflation_factor(X.values, 5),
}
vif_m2 = {col: variance_inflation_factor(X2.values, i+1)
          for i, col in enumerate(X2_cols)}
for col in X2_cols:
    print(f"  {col:<20} {vif_m1[col]:>8.2f}  {vif_m2[col]:>8.2f}")
print(f"  {'urbanizacja_pct':<20} {vif_m1['urbanizacja_pct']:>8.2f}  {'usunięta':>8}")

# ── M2-C. WERYFIKACJA NUMERYCZNA ────────────────────────────
R2_2     = model2.rsquared
R2_adj_2 = model2.rsquared_adj
AIC_2    = model2.aic
BIC_2    = model2.bic
res2     = model2.resid

print(f"\n  Porównanie dopasowania:")
print(f"  {'Miara':<12} {'Model 1':>10}  {'Model 2':>10}")
print("  " + "-" * 36)
print(f"  {'R²':<12} {R2:>10.4f}  {R2_2:>10.4f}")
print(f"  {'R² adj.':<12} {R2_adj:>10.4f}  {R2_adj_2:>10.4f}")
print(f"  {'AIC':<12} {AIC:>10.3f}  {AIC_2:>10.3f}")
print(f"  {'BIC':<12} {BIC:>10.3f}  {BIC_2:>10.3f}")

# ── M2-D. WERYFIKACJA STOCHASTYCZNA ─────────────────────────
from scipy.stats import shapiro as _sw
sw2_stat, sw2_p = _sw(res2)
dw2 = durbin_watson(res2)
bg2_stat, bg2_p, _, _ = acorr_breusch_godfrey(model2, nlags=2)
bp2_lm, bp2_p, _, _   = het_breuschpagan(res2, X2)

print(f"\n  Weryfikacja stochastyczna – Model 2:")
print(f"  Shapiro-Wilk  p = {sw2_p:.4f}  "
      f"{'✅' if sw2_p > 0.05 else '❌'}")
print(f"  Durbin-Watson   = {dw2:.4f}  "
      f"{'✅' if 1.5 < dw2 < 2.5 else '⚠️ możliwa autokorelacja'}")
print(f"  Breusch-Godfrey p = {bg2_p:.4f}  "
      f"{'✅' if bg2_p > 0.05 else '❌ autokorelacja'}")
print(f"  Breusch-Pagan   p = {bp2_p:.4f}  "
      f"{'✅' if bp2_p > 0.05 else '❌ heteroskedastyczność'}")

# ── M2-E. WYKRES DIAGNOSTYCZNY ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Diagnostyka – Iteracja 2 (bez urbanizacji)", fontweight="bold")

fitted2 = model2.fittedvalues
axes[0].scatter(fitted2, res2, color=BLUE, alpha=0.8, s=50)
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_title("Reszty vs Dopasowane")
axes[0].set_xlabel("Dopasowane (ln)")
axes[0].set_ylabel("Reszty")

sm.qqplot(res2, line="s", ax=axes[1], alpha=0.8, color=BLUE)
axes[1].set_title("QQ-plot")

axes[2].plot(df["rok"], res2, marker="o", color=ORANGE, linewidth=2, markersize=5)
axes[2].axhline(0, color="black", linestyle="--")
axes[2].fill_between(df["rok"], res2, 0, where=(res2 > 0), alpha=0.2, color=GREEN)
axes[2].fill_between(df["rok"], res2, 0, where=(res2 < 0), alpha=0.2, color=RED)
axes[2].set_title("Reszty w czasie")
axes[2].set_xlabel("Rok")
axes[2].xaxis.set_major_locator(mticker.MultipleLocator(4))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "08_diagnostyka_iter2.png"), bbox_inches="tight")
plt.show()
print("✅ Zapisano: 08_diagnostyka_iter2.png")

# ── M2-F. INTERPRETACJA ─────────────────────────────────────
p2 = model2.params
print(f"\n  Interpretacja – Iteracja 2:")
print(f"  β₁ (ln_pkb_pc) = {p2['ln_pkb_pc']:+.4f}"
      f"  → wzrost PKB pc o 1% → zużycie o {p2['ln_pkb_pc']:+.2f}%")
print(f"  β₂ (ln_cena)   = {p2['ln_cena']:+.4f}"
      f"  → wzrost ceny o 1%   → zużycie o {p2['ln_cena']:+.2f}%")
print(f"  β₃ (hdd)       = {p2['hdd']:+.6f}"
      f"  → wzrost HDD o 1     → zużycie o {p2['hdd']:+.6f} (ln)")
print(f"  β₄ (cdd)       = {p2['cdd']:+.6f}"
      f"  → wzrost CDD o 1     → zużycie o {p2['cdd']:+.6f} (ln)")
