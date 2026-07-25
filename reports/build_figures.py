#!/usr/bin/env python3
"""
Generate the EDA figures and summary statistics for the AAVAIL revenue
case-study, part 1, report.

Reads the processed artifacts written by scripts/ingest_data.py
(data/processed/*.csv) and produces:
  * PNG figures in reports/figures/
  * a JSON of summary statistics in reports/content/eda_stats.json, used to
    populate the narrative text of the PDF report.

Usage: python reports/build_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
CONTENT_DIR = ROOT / "reports" / "content"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette / chart chrome (validated categorical + status palette)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
WARNING = "#fab219"
MUTED_FILL = "#c3c2b7"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK_PRIMARY,
})


def style_axes(ax, y_zero_baseline=True):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    if y_zero_baseline:
        ax.axhline(0, color=BASELINE, linewidth=1.0)


def money(x, _pos=None):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:,.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:,.0f}K"
    return f"${x:,.0f}"


MONEY_FMT = mticker.FuncFormatter(money)


# ---------------------------------------------------------------------------
# Load processed data
# ---------------------------------------------------------------------------
transactions = pd.read_csv(PROCESSED / "transactions_clean.csv", parse_dates=["date"])
ts_all = pd.read_csv(PROCESSED / "ts_all.csv", parse_dates=["date"], index_col="date")
ts_top = pd.read_csv(PROCESSED / "ts_top_countries.csv", parse_dates=["date"])
top_countries = pd.read_csv(PROCESSED / "top_countries.csv")["country"].tolist()

stats = {}

# ---------------------------------------------------------------------------
# Fig 1: daily revenue over time, with rolling mean and the holiday-gap callout
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
rolling = ts_all["revenue"].rolling(30, min_periods=7).mean()
ax.plot(ts_all.index, ts_all["revenue"], color=BLUE, alpha=0.28, linewidth=1.0, label="Daily revenue")
ax.plot(ts_all.index, rolling, color=BLUE_DARK, linewidth=2.0, label="30-day rolling mean")

for year in (2017, 2018):
    gap_start = pd.Timestamp(year=year, month=12, day=21)
    gap_end = pd.Timestamp(year=year, month=12, day=31)
    if gap_start >= ts_all.index.min() and gap_end <= ts_all.index.max():
        ax.axvspan(gap_start, gap_end, color=WARNING, alpha=0.18, linewidth=0)

ax.axvspan(ts_all.index.min(), pd.Timestamp("2017-12-01"), color=MUTED_FILL, alpha=0.15, linewidth=0)
ax.text(0.01, 0.96, "Shaded: partial/gap coverage (see data-quality note)", transform=ax.transAxes,
        fontsize=8.5, color=INK_MUTED, va="top")

ax.yaxis.set_major_formatter(MONEY_FMT)
ax.set_title("Daily revenue, all countries (Nov 2017 – Jul 2019)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
style_axes(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig01_daily_revenue.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: monthly revenue bar chart (the business metric), partial months flagged
# ---------------------------------------------------------------------------
monthly = ts_all["revenue"].resample("MS").sum()
day_counts = ts_all["revenue"].resample("MS").size()
days_in_month = day_counts.index.days_in_month
coverage = ts_all["revenue"].resample("MS").apply(lambda s: (s > 0).sum() + (s == 0).sum())
is_partial = pd.Series(False, index=monthly.index)
is_partial.iloc[0] = True  # Nov 2017: only 3 days observed
dec_months = [ts for ts in monthly.index if ts.month == 12]
for ts in dec_months:
    is_partial[ts] = True  # Dec: 11-day end-of-year gap in every observed year

fig, ax = plt.subplots(figsize=(11, 4.5))
colors = [WARNING if p else BLUE for p in is_partial]
bars = ax.bar(monthly.index, monthly.values, width=20, color=colors)
ax.yaxis.set_major_formatter(MONEY_FMT)
ax.set_title("Monthly revenue — the metric this service predicts one month ahead")
ax.set_xticks(monthly.index)
ax.set_xticklabels([d.strftime("%b\n%Y") for d in monthly.index], fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Full month"), Patch(color=WARNING, label="Partial / gapped month")],
          frameon=False, loc="upper left", fontsize=9)
style_axes(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig02_monthly_revenue.png", dpi=200)
plt.close(fig)

full_months = monthly[~is_partial]
stats["n_full_months"] = int(len(full_months))
stats["n_partial_months"] = int(is_partial.sum())
stats["monthly_revenue_mean"] = float(full_months.mean())
stats["monthly_revenue_min"] = float(full_months.min())
stats["monthly_revenue_max"] = float(full_months.max())
stats["monthly_revenue_cv"] = float(full_months.std() / full_months.mean())

# ---------------------------------------------------------------------------
# Fig 3: countries ranked by total revenue, top 10 highlighted
# ---------------------------------------------------------------------------
rev_by_country = (
    transactions[~transactions["country"].str.lower().isin({"unspecified", "european community"})]
    .groupby("country")["price"].sum().sort_values(ascending=False)
)
total_revenue = float(transactions["price"].sum())
top10_revenue = float(rev_by_country.head(10).sum())
uk_revenue = float(rev_by_country.get("United Kingdom", 0.0))

stats["total_revenue_all_countries"] = total_revenue
stats["top10_share_of_revenue"] = top10_revenue / total_revenue
stats["uk_share_of_revenue"] = uk_revenue / total_revenue
stats["n_countries_observed"] = int(rev_by_country.shape[0])

non_uk_total = total_revenue - uk_revenue
top9_ex_uk_revenue = float(rev_by_country.head(10).drop("United Kingdom").sum())
stats["top9_ex_uk_share_of_non_uk_revenue"] = top9_ex_uk_revenue / non_uk_total
stats["top9_ex_uk_share_of_total_revenue"] = top9_ex_uk_revenue / total_revenue

monthly_full = ts_all.resample("MS").sum().iloc[1:-1]
stats["monthly_corr_purchases_revenue"] = float(monthly_full["purchases"].corr(monthly_full["revenue"]))
stats["monthly_corr_streams_revenue"] = float(monthly_full["unique_streams"].corr(monthly_full["revenue"]))
stats["monthly_corr_customers_revenue"] = float(monthly_full["unique_customers"].corr(monthly_full["revenue"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1, 2.1]})

# Left panel: UK vs. rest-of-world split (UK's scale otherwise crushes every
# other bar to invisibility on a shared linear axis).
uk_vs_rest = pd.Series({"United Kingdom": uk_revenue, "All other countries\n(41 total)": total_revenue - uk_revenue})
ax1.bar(uk_vs_rest.index, uk_vs_rest.values, color=[BLUE, MUTED_FILL], width=0.55)
ax1.yaxis.set_major_formatter(MONEY_FMT)
ax1.set_title(f"UK = {stats['uk_share_of_revenue']*100:.0f}% of all revenue", fontsize=11.5)
ax1.tick_params(axis="x", labelsize=9)
style_axes(ax1, y_zero_baseline=False)
ax1.spines["left"].set_visible(True)
ax1.spines["left"].set_color(BASELINE)

# Right panel: the other 9 modeled countries plus a few more, ranked, so
# their relative differences are actually visible.
rest = rev_by_country.drop("United Kingdom").head(14)
colors = [BLUE if c in top_countries else MUTED_FILL for c in rest.index]
y_pos = np.arange(len(rest))[::-1]
ax2.barh(y_pos, rest.values, color=colors, height=0.62)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(rest.index, fontsize=9.5)
ax2.xaxis.set_major_formatter(MONEY_FMT)
ax2.set_title("Ranked, excluding the UK — blue = the other 9 modeled countries", fontsize=11.5)
ax2.text(0.98, 0.04,
         f"Top 10 (incl. UK) = {stats['top10_share_of_revenue']*100:.1f}% of all revenue\n"
         f"across {stats['n_countries_observed']} countries observed",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)
style_axes(ax2, y_zero_baseline=False)
ax2.spines["bottom"].set_visible(True)

fig.suptitle("Total revenue by country (Nov 2017 – Jul 2019)", fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig03_country_revenue_ranked.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: small multiples of monthly revenue trend, one per top-10 country
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(15, 5.5), sharex=False)
for ax, country in zip(axes.flat, top_countries):
    sub = ts_top[ts_top["country"] == country].set_index("date")["revenue"]
    monthly_c = sub.resample("MS").sum()
    if len(monthly_c) > 2:
        monthly_c = monthly_c.iloc[1:-1]  # drop partial first/last month for this small view
    ax.plot(monthly_c.index, monthly_c.values, color=BLUE, linewidth=1.8)
    ax.fill_between(monthly_c.index, monthly_c.values, color=BLUE, alpha=0.10)
    ax.set_title(country, fontsize=10, fontweight="bold", color=INK_PRIMARY, pad=4)
    ax.yaxis.set_major_formatter(MONEY_FMT)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(color=GRIDLINE, linewidth=0.8)
fig.suptitle("Monthly revenue trend, top-10 countries (each panel own scale)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig04_top10_country_trends.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5: daily purchases vs daily revenue (relationship to the target)
# ---------------------------------------------------------------------------
mask = ts_all["revenue"] > 0
x = ts_all.loc[mask, "purchases"].values
y = ts_all.loc[mask, "revenue"].values
corr = float(np.corrcoef(x, y)[0, 1])
stats["corr_purchases_revenue"] = corr

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(x, y, s=22, color=BLUE, alpha=0.45, edgecolors="none")
coeffs = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, np.polyval(coeffs, xs), color=ORANGE, linewidth=2.0)
ax.yaxis.set_major_formatter(MONEY_FMT)
ax.set_xlabel("Distinct purchases (invoices) that day")
ax.set_ylabel("")
ax.set_title(f"Daily purchase volume vs. daily revenue  (r = {corr:.2f})")
style_axes(ax, y_zero_baseline=False)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color(BASELINE)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig05_purchases_vs_revenue.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6: day-of-week seasonality
# ---------------------------------------------------------------------------
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = ts_all.copy()
dow["dow"] = dow.index.day_name()
dow = dow[dow.index >= "2018-01-01"]  # skip the short first partial month
dow = dow[~((dow.index.month == 12) & (dow.index.day >= 21))]  # exclude the known holiday gap

fig, ax = plt.subplots(figsize=(8.5, 5))
data_by_dow = [dow.loc[dow["dow"] == d, "revenue"].values for d in dow_order]
bp = ax.boxplot(data_by_dow, tick_labels=dow_order, patch_artist=True, showfliers=False, widths=0.55)
for patch in bp["boxes"]:
    patch.set_facecolor(BLUE)
    patch.set_alpha(0.35)
    patch.set_edgecolor(BLUE_DARK)
for element in ("whiskers", "caps"):
    for line in bp[element]:
        line.set_color(BASELINE)
for median in bp["medians"]:
    median.set_color(BLUE_DARK)
    median.set_linewidth(2)
ax.yaxis.set_major_formatter(MONEY_FMT)
ax.set_title("Daily revenue by day of week (holiday gap excluded)")
style_axes(ax, y_zero_baseline=False)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color(BASELINE)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig06_day_of_week.png", dpi=200)
plt.close(fig)

sat_median = float(dow.loc[dow["dow"] == "Saturday", "revenue"].median())
weekday_median = float(dow.loc[dow["dow"].isin(dow_order[:5]), "revenue"].median())
stats["saturday_median_revenue"] = sat_median
stats["weekday_median_revenue"] = weekday_median

# ---------------------------------------------------------------------------
# Fig 7: transaction price distribution (log scale)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
prices = transactions.loc[transactions["price"] > 0, "price"]
bins = np.logspace(np.log10(max(prices.min(), 0.01)), np.log10(prices.quantile(0.999)), 40)
ax.hist(prices.clip(upper=prices.quantile(0.999)), bins=bins, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=0.5)
ax.set_xscale("log")
ax.set_xlabel("Transaction price (log scale, $)")
ax.set_ylabel("Number of transactions")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos=None: f"{x:,.0f}"))
ax.set_title("Distribution of per-transaction price")
style_axes(ax, y_zero_baseline=False)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color(BASELINE)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig07_price_distribution.png", dpi=200)
plt.close(fig)

stats["price_median"] = float(prices.median())
stats["price_mean"] = float(prices.mean())
stats["price_p95"] = float(prices.quantile(0.95))

# ---------------------------------------------------------------------------
# Fig 8: customer_id completeness over time (data-quality caveat)
# ---------------------------------------------------------------------------
transactions["ym"] = transactions["date"].dt.to_period("M")
missing_rate = transactions.groupby("ym")["customer_id"].apply(lambda s: s.isna().mean())
missing_rate = missing_rate.iloc[1:-1] if len(missing_rate) > 2 else missing_rate

fig, ax = plt.subplots(figsize=(10, 4))
xs = missing_rate.index.to_timestamp()
ax.plot(xs, missing_rate.values * 100, color=WARNING, linewidth=2.0, marker="o", markersize=5,
        markerfacecolor=WARNING, markeredgecolor=SURFACE, markeredgewidth=1.2)
ax.fill_between(xs, missing_rate.values * 100, color=WARNING, alpha=0.12)
ax.set_ylabel("% of transactions missing customer_id")
ax.set_title("Data-quality caveat: share of transactions with no customer_id, by month")
ax.set_ylim(bottom=0)
style_axes(ax, y_zero_baseline=False)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color(BASELINE)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig08_missing_customer_id.png", dpi=200)
plt.close(fig)

stats["overall_missing_customer_id_rate"] = float(transactions["customer_id"].isna().mean())

# ---------------------------------------------------------------------------
# Persist stats for the report text
# ---------------------------------------------------------------------------
stats["top_countries"] = top_countries
stats["date_min"] = str(transactions["date"].min().date())
stats["date_max"] = str(transactions["date"].max().date())
stats["n_transactions"] = int(len(transactions))
stats["n_invoices"] = int(transactions["invoice_clean"].nunique())

with open(CONTENT_DIR / "eda_stats.json", "w") as fh:
    json.dump(stats, fh, indent=2)

print("wrote figures to", FIG_DIR)
print("wrote stats to", CONTENT_DIR / "eda_stats.json")
print(json.dumps(stats, indent=2))
