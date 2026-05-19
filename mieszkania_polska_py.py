# -*- coding: utf-8 -*-
# ============================================================
# ANALIZA MIESZKAŃ ODDANYCH DO UŻYTKOWANIA W POLSCE
# ============================================================
# Projekt: Wpływ czynników społeczno-ekonomicznych na liczbę
#          mieszkań oddanych do użytkowania w Polsce
# ============================================================

# ── 0. IMPORTY ───────────────────────────────────────────────
# Wymagane pakiety (uruchom raz w !!!!!!R!!!!! NIE PYTHON jeśli brakuje):
# reticulate::py_install(c("pandas", "numpy", "matplotlib", "seaborn","statsmodels", "scipy", "openpyxl", "pmdarima", "python-docx"))

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
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = SCRIPT_DIR / "mieszkania_polska.xlsx"

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

# Ujednolicenie nazwy kolumny nakładów (może mieć różne kodowanie znaku Ł)
for col in df.columns:
    if col.startswith("NAK"):
        df = df.rename(columns={col: "NAKL"})
        break

print("=" * 60)
print("WCZYTANE DANE – PODGLĄD")
print("=" * 60)
print(df.to_string(index=False))

# ── 1b. ZMIENNE POCHODNE ─────────────────────────────────────
df["trend"]    = np.arange(len(df))
df["MO_lag1"]  = df["MO"].shift(1)
df_lagged      = df.dropna(subset=["MO_lag1"])

# ── 2. ANALIZA OPISOWA ───────────────────────────────────────
print("\n" + "=" * 60)
print("STATYSTYKI OPISOWE")
print("=" * 60)
desc_cols = ["MO", "NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM"]
desc = df[desc_cols].describe().T
desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(2)
print(desc.round(3).to_string())

# ── 3. WYKRESY SZEREGÓW CZASOWYCH ───────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Szeregi czasowe zmiennych – Polska 2004–2024",
    fontsize=15, fontweight="bold", y=1.01
)

variables = {
    "Mieszkania oddane [szt.]":        ("MO",       BLUE),
    "Nakłady na budownictwo [mln zł]": ("NAKL",     GREEN),
    "Wynagrodzenie [zł]":              ("WYNAGR",   ORANGE),
    "Udział wieku 25–34":              ("WSK25-34", PURPLE),
    "Urbanizacja":                     ("WSK_URB",  GRAY),
    "Saldo migracji [‰]":              ("SM",       RED),
}

for ax, (title, (col, color)) in zip(axes.flat, variables.items()):
    ax.plot(df["rok"], df[col], marker="o", color=color, linewidth=2, markersize=5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Rok")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:.3f}")
    )

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "m01_szeregi_czasowe.png"), bbox_inches="tight")
print("✅ Zapisano: m01_szeregi_czasowe.png")

# ── 4. MACIERZ KORELACJI ────────────────────────────────────
corr_cols   = ["MO", "NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
    vmin=-1, vmax=1, ax=ax, linewidths=0.5,
    annot_kws={"size": 10}
)
ax.set_title("Macierz korelacji Pearsona", fontsize=13, fontweight="bold")
labels = ["Mieszkania oddane", "Nakłady", "Wynagrodzenie",
          "Udział 25–34", "Urbanizacja", "Saldo migracji"]
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "m02_korelacja.png"), bbox_inches="tight")
print("✅ Zapisano: m02_korelacja.png")

# ── 5. WYKRESY ROZRZUTU ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Zależności liczby mieszkań oddanych od zmiennych objaśniających",
    fontsize=13, fontweight="bold"
)

scatter_vars = [
    ("NAKL",     "Nakłady na budownictwo [mln zł]", GREEN),
    ("WYNAGR",   "Wynagrodzenie [zł]",               ORANGE),
    ("WSK25-34", "Udział wieku 25–34",               PURPLE),
    ("WSK_URB",  "Urbanizacja",                      GRAY),
    ("SM",       "Saldo migracji [‰]",               RED),
    ("trend",    "Trend",                            "#2980b9"),
]

for ax, (col, xlabel, color) in zip(axes.flat, scatter_vars):
    ax.scatter(df[col], df["MO"],
               color=color, alpha=0.8, s=60, edgecolors="white", linewidth=0.5)
    z = np.polyfit(df[col], df["MO"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, p(x_line), "--", color="black", alpha=0.5, linewidth=1.5)
    r, pval = stats.pearsonr(df[col], df["MO"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mieszkania oddane [szt.]")
    ax.set_title(f"r = {r:.3f}  (p = {pval:.3f})", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "m03_scatter.png"), bbox_inches="tight")
print("✅ Zapisano: m03_scatter.png")

# ── 6. BUDOWA MODELU OLS ─────────────────────────────────────
#
# Model liniowy:
#   MO = β₀ + β₁·NAKL + β₂·WYNAGR
#          + β₃·WSK25-34 + β₄·WSK_URB + β₅·SM + ε
#
# Uzasadnienie:
#  • NAKL     → efekt podażowy (nakłady na budownictwo)
#  • WYNAGR   → efekt dochodowy / siła nabywcza
#  • WSK25-34 → czynnik demograficzny (główni nabywcy mieszkań)
#  • WSK_URB  → czynnik strukturalny (koncentracja miejska)
#  • SM       → saldo migracji (presja popytowa)

X_cols = ["NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM"]
y      = df["MO"]
X      = sm.add_constant(df[X_cols])

model = sm.OLS(y, X).fit()

print("\n" + "=" * 60)
print("MODEL 1 – LINIOWY OLS (wyniki estymacji)")
print("=" * 60)
print("""
  SPECYFIKACJA:
    MO = β₀ + β₁·NAKL + β₂·WYNAGR + β₃·WSK25-34 + β₄·WSK_URB + β₅·SM + ε

  ZMIENNA ZALEŻNA:
    MO – mieszkania oddane do użytkowania [szt./rok]

  ZMIENNE OBJAŚNIAJĄCE:
    NAKL      – nakłady na budownictwo mieszkaniowe [mln zł]; czynnik podażowy;
                wzrost inwestycji → więcej ukończonych mieszkań
    WYNAGR    – przeciętne wynagrodzenie brutto [zł]; siła nabywcza gospodarstw
                domowych → wzrost zamożności → wyższy popyt
    WSK25-34  – udział osób w wieku 25–34 lat w populacji; główna grupa
                nabywców i najemców mieszkań; czynnik demograficzny
    WSK_URB   – wskaźnik urbanizacji; koncentracja popytu w miastach
    SM        – saldo migracji [‰]; napływ ludności zwiększa popyt na mieszkania

  FORMA FUNKCYJNA: liniowa (bez logarytmów) → współczynniki β to efekty
  marginalne w jednostkach oryginalnych.

  UWAGA: wykryto wysoką współliniowość (VIF>23 dla NAKL i WYNAGR).
         Estymacja punktowa może być niestabilna → patrz Iteracja 2.
""")
print(model.summary())

# ── 6b. MODEL Z OPÓŹNIONĄ ZMIENNĄ ZALEŻNĄ ───────────────────
X_cols_lagged = ["NAKL", "WYNAGR", "WSK25-34", "WSK_URB", "SM", "MO_lag1"]
y_lagged      = df_lagged["MO"]
X_lagged      = sm.add_constant(df_lagged[X_cols_lagged])
model_lagged  = sm.OLS(y_lagged, X_lagged).fit()

print("\n" + "=" * 60)
print("MODEL 1b – DYNAMICZNY OLS (opóźniona zmienna zależna)")
print("=" * 60)
print("""
  SPECYFIKACJA:
    MO_t = β₀ + β₁·NAKL_t + β₂·WYNAGR_t + β₃·WSK25-34_t
          + β₄·WSK_URB_t + β₅·SM_t + β₆·MO_{t-1} + ε_t

  CEL: uchwycenie wieloletnich cykli budowlanych – liczba oddanych mieszkań
  w roku t zależy częściowo od liczby w roku t-1 (projekty realizowane
  przez kilka lat, opóźnienia decyzyjne i administracyjne).

  WŁAŚCIWOŚCI:
    • MO_lag1 absorbuje autokorelację reszt i poprawia DW
    • Kosztem: utrata 1 obserwacji (n = 20 zamiast 21)
    • Współczynnik β₆ to miara inercji rynku mieszkaniowego
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
ax.set_xlabel("Wartości dopasowane [szt.]")
ax.set_ylabel("Reszty")
ax.set_title("Reszty vs Dopasowane")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

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
ax.plot(df["rok"], y,      "o-",  color=BLUE, label="Rzeczywiste", linewidth=2)
ax.plot(df["rok"], fitted, "s--", color=RED,  label="Dopasowane",  linewidth=2)
ax.set_title("Rzeczywiste vs Dopasowane [szt.]")
ax.set_xlabel("Rok")
ax.set_ylabel("Mieszkania oddane [szt.]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

# 9f. CUSUM reszt
ax = axes[1, 2]
cusum   = np.cumsum(residuals)
std_res = residuals.std()
ax.plot(df["rok"], cusum, color=PURPLE, linewidth=2, marker="o", markersize=5)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.axhline(+2 * std_res * np.sqrt(n), color="red", linestyle=":", label="±2σ√n")
ax.axhline(-2 * std_res * np.sqrt(n), color="red", linestyle=":")
ax.set_title("CUSUM reszt (stabilność)")
ax.set_xlabel("Rok")
ax.legend()
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "m04_diagnostyka.png"), bbox_inches="tight")
print("✅ Zapisano: m04_diagnostyka.png")

# ── 10. INTERPRETACJA ANALITYCZNA ───────────────────────────
print("\n" + "=" * 60)
print("INTERPRETACJA ANALITYCZNA MODELU LINIOWEGO")
print("=" * 60)
params = model.params

print(f"""
  Model postaci: MO = β₀ + β₁·NAKL + β₂·WYNAGR
                    + β₃·WSK25-34 + β₄·WSK_URB + β₅·SM + ε

  β₀ (stała)       = {params['const']:+.2f}
  β₁ (NAKL)        = {params['NAKL']:+.4f}
    → Wzrost nakładów o 1 mln zł zmienia liczbę mieszkań o {params['NAKL']:+.2f} szt.

  β₂ (WYNAGR)      = {params['WYNAGR']:+.4f}
    → Wzrost wynagrodzenia o 1 zł zmienia liczbę mieszkań o {params['WYNAGR']:+.2f} szt.

  β₃ (WSK25-34)    = {params['WSK25-34']:+.4f}
    → Wzrost udziału osób 25–34 o 0.001 zmienia liczbę mieszkań o {params['WSK25-34'] * 0.001:+.2f} szt.

  β₄ (WSK_URB)     = {params['WSK_URB']:+.4f}
    → Wzrost urbanizacji o 0.001 zmienia liczbę mieszkań o {params['WSK_URB'] * 0.001:+.2f} szt.

  β₅ (SM)          = {params['SM']:+.4f}
    → Wzrost salda migracji o 1‰ zmienia liczbę mieszkań o {params['SM']:+.2f} szt.
""")

# ── 11. WYKRES WSPÓŁCZYNNIKÓW ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

coef_names = {
    "NAKL":     "Nakłady na budownictwo\n[mln zł]",
    "WYNAGR":   "Wynagrodzenie\n[zł]",
    "WSK25-34": "Udział 25–34 lat\n[czynnik demograficzny]",
    "WSK_URB":  "Urbanizacja\n[czynnik strukturalny]",
    "SM":       "Saldo migracji\n[presja popytowa]",
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
    ax.text(val + (abs(coefs.values).max() * 0.01 if val > 0 else -abs(coefs.values).max() * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center",
            ha="left" if val > 0 else "right", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "m05_wspolczynniki.png"), bbox_inches="tight")
print("✅ Zapisano: m05_wspolczynniki.png")

# ── 12. MODEL ARIMA ──────────────────────────────────────────
if PMDARIMA_AVAILABLE:
    print("\n" + "=" * 60)
    print("MODEL ARIMA – WYNIKI ESTYMACJI")
    print("=" * 60)

    y_ts = df["MO"]
    model_arima = pm.auto_arima(
        y_ts,
        seasonal=False,
        suppress_warnings=True,
        stepwise=True
    )
    print(model_arima.summary())

    fc_vals, fc_ci = model_arima.predict(n_periods=6, return_conf_int=True)
    forecast_years = list(range(2025, 2031))
    print("\n  Prognoza ARIMA 2025–2030 [szt.]:")
    for yr, val, ci in zip(forecast_years, fc_vals, fc_ci):
        print(f"    {yr}: {val:,.0f} szt."
              f"  [95% CI: {ci[0]:,.0f} – {ci[1]:,.0f}]")

# ── 13. PODSUMOWANIE KOŃCOWE ─────────────────────────────────
print("\n" + "=" * 60)
print("PODSUMOWANIE PROJEKTU")
print("=" * 60)
print(f"""
  Zmienna zależna : Liczba mieszkań oddanych do użytkowania w Polsce [szt.]
  Okres próby     : 2004–2024  (n = {n} obserwacji rocznych)
  Model           : liniowy OLS (5 zmiennych objaśniających)

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
    • Nakłady (NAKL)       β = {params['NAKL']:+.4f}
      → Główny czynnik podażowy – inwestycje napędzają budownictwo
    • Wynagrodzenia (WYNAGR) β = {params['WYNAGR']:+.4f}
      → Wzrost zamożności zwiększa popyt na mieszkania
    • Demografia (WSK25-34) β = {params['WSK25-34']:+.4f}
      → Malejący udział grupy 25–34 lat wywiera presję na spadek popytu

  Wygenerowane pliki:
    m01_szeregi_czasowe.png
    m02_korelacja.png
    m03_scatter.png
    m04_diagnostyka.png
    m05_wspolczynniki.png
""")

# ============================================================
# ITERACJA 2 – KOREKTA WIELOKOLINIOWOŚCI I AUTOKORELACJI
# ============================================================
# Problemy wykryte w Modelu 1:
#   1. VIF(WYNAGR) = 27.7,  VIF(NAKL) = 23.6
#      Korelacja NAKL ↔ WYNAGR = 0.974  → usunięcie WYNAGR
#   2. Durbin-Watson = 1.015, BG p = 0.013 → autokorelacja reszt
#      Korekta: błędy standardowe HAC (Newey-West)
# ============================================================

print("\n" + "=" * 60)
print("ITERACJA 2 – MODEL 2 (korekta wielokoliniowości + HAC)")
print("=" * 60)
print("""
  MOTYWACJA:
    W Modelu 1 wykryto dwa problemy:
      1. Wielokoliniowość: VIF(WYNAGR)=27.7, VIF(NAKL)=23.6
         corr(NAKL, WYNAGR) = 0.974 → oba odzwierciedlają wzrost gosp.
         Usunięto WYNAGR jako zmienną bardziej pośrednią (efekt dochodowy
         jest już częściowo zawarty w NAKL i poziomie aktywności budowlanej).
      2. Autokorelacja reszt: DW ≈ 1.0, BG p < 0.05
         Zastosowano błędy standardowe HAC (Newey-West, maxlags=2)
         jako korektę wnioskowania statystycznego.

  SPECYFIKACJA:
    MO = β₀ + β₁·NAKL + β₂·WSK25-34 + β₃·WSK_URB + β₄·SM + ε
    Błędy standardowe: HAC Newey-West (maxlags=2)

  OCENA DO PROGNOZOWANIA:
    Reszty normalne (SW p>0.05), brak heteroskedastyczności → OK.
    Autokorelacja nadal obecna w resztach OLS (DW≈0.75) – HAC poprawia
    wnioskowanie, ale nie strukturę błędów prognoz. Do prognozowania
    zalecany wariant z MO_lag1 lub model ARIMA (sekcja 12).
""")

# ── M2-A. ESTYMACJA ─────────────────────────────────────────
#   MO = β₀ + β₁·NAKL + β₂·WSK25-34 + β₃·WSK_URB + β₄·SM + ε
#   Błędy standardowe: HAC (Newey-West, nlags=2)

X2_cols = ["NAKL", "WSK25-34", "WSK_URB", "SM"]
X2      = sm.add_constant(df[X2_cols])
model2_ols = sm.OLS(df["MO"], X2).fit()
model2     = model2_ols.get_robustcov_results(cov_type="HAC", maxlags=2)

print(model2.summary())

# ── M2-B. PORÓWNANIE VIF ────────────────────────────────────
print("\n  Porównanie VIF – Model 1 vs Model 2:")
print(f"  {'Zmienna':<20} {'VIF M1':>8}  {'VIF M2':>8}")
print("  " + "-" * 42)
vif_m1_dict = {
    "NAKL":     variance_inflation_factor(X.values, 1),
    "WYNAGR":   variance_inflation_factor(X.values, 2),
    "WSK25-34": variance_inflation_factor(X.values, 3),
    "WSK_URB":  variance_inflation_factor(X.values, 4),
    "SM":       variance_inflation_factor(X.values, 5),
}
vif_m2_dict = {col: variance_inflation_factor(X2.values, i+1)
               for i, col in enumerate(X2_cols)}
for col in X2_cols:
    print(f"  {col:<20} {vif_m1_dict[col]:>8.2f}  {vif_m2_dict[col]:>8.2f}")
print(f"  {'WYNAGR':<20} {vif_m1_dict['WYNAGR']:>8.2f}  {'usunięta':>8}")

# ── M2-C. WERYFIKACJA NUMERYCZNA ────────────────────────────
R2_2     = model2_ols.rsquared
R2_adj_2 = model2_ols.rsquared_adj
AIC_2    = model2_ols.aic
BIC_2    = model2_ols.bic
res2     = model2_ols.resid

print(f"\n  Porównanie dopasowania:")
print(f"  {'Miara':<12} {'Model 1':>10}  {'Model 2':>10}")
print("  " + "-" * 36)
print(f"  {'R²':<12} {R2:>10.4f}  {R2_2:>10.4f}")
print(f"  {'R² adj.':<12} {R2_adj:>10.4f}  {R2_adj_2:>10.4f}")
print(f"  {'AIC':<12} {AIC:>10.3f}  {AIC_2:>10.3f}")
print(f"  {'BIC':<12} {BIC:>10.3f}  {BIC_2:>10.3f}")

# ── M2-D. WERYFIKACJA STOCHASTYCZNA ─────────────────────────
from scipy.stats import shapiro as _sw2
sw2_stat, sw2_p = _sw2(res2)
dw2 = durbin_watson(res2)
bg2_stat, bg2_p, _, _ = acorr_breusch_godfrey(model2_ols, nlags=2)
bp2_lm, bp2_p, _, _   = het_breuschpagan(res2, X2)

print(f"\n  Weryfikacja stochastyczna – Model 2 (OLS bez WYNAGR):")
print(f"  Shapiro-Wilk  p = {sw2_p:.4f}  "
      f"{'✅' if sw2_p > 0.05 else '❌'}")
print(f"  Durbin-Watson   = {dw2:.4f}  "
      f"{'✅' if 1.5 < dw2 < 2.5 else '⚠️ autokorelacja → stosujemy HAC'}")
print(f"  Breusch-Godfrey p = {bg2_p:.4f}  "
      f"{'✅' if bg2_p > 0.05 else '❌ autokorelacja → błędy HAC aktywne'}")
print(f"  Breusch-Pagan   p = {bp2_p:.4f}  "
      f"{'✅' if bp2_p > 0.05 else '❌ heteroskedastyczność'}")
print(f"\n  ℹ️  Błędy standardowe w Modelu 2 są korygowane metodą HAC")
print(f"     (Newey-West, maxlags=2) – odporne na autokorelację i heteroskedastyczność.")

# ── M2-E. WYKRES DIAGNOSTYCZNY ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Diagnostyka – Iteracja 2 (bez WYNAGR, HAC)", fontweight="bold")

fitted2 = model2_ols.fittedvalues
axes[0].scatter(fitted2, res2, color=BLUE, alpha=0.8, s=50)
axes[0].axhline(0, color="black", linestyle="--")
axes[0].set_title("Reszty vs Dopasowane")
axes[0].set_xlabel("Dopasowane [szt.]")
axes[0].set_ylabel("Reszty")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

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
plt.savefig(os.path.join(SCRIPT_DIR, "m06_diagnostyka_iter2.png"), bbox_inches="tight")
print("✅ Zapisano: m06_diagnostyka_iter2.png")

# ── M2-F. INTERPRETACJA ─────────────────────────────────────
p2 = pd.Series(model2.params, index=model2.model.exog_names)

print(f"\n  Interpretacja – Iteracja 2 (błędy HAC):")
print(f"  β₁ (NAKL)     = {p2['NAKL']:+.4f}"
      f"  → wzrost nakładów o 1 mln zł → {p2['NAKL']:+.2f} szt.")
print(f"  β₂ (WSK25-34) = {p2['WSK25-34']:+.4f}"
      f"  → wzrost udziału 25-34 o 0.001 → {p2['WSK25-34']*0.001:+.2f} szt.")
print(f"  β₃ (WSK_URB)  = {p2['WSK_URB']:+.4f}"
      f"  → wzrost urbanizacji o 0.001 → {p2['WSK_URB']*0.001:+.2f} szt.")
print(f"  β₄ (SM)       = {p2['SM']:+.4f}"
      f"  → wzrost salda migracji o 1‰ → {p2['SM']:+.2f} szt.")
