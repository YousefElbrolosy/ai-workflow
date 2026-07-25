#!/usr/bin/env python3
"""
Assemble the Part 1 case-study deliverable: a single PDF report combining
the business understanding narrative, the ideal-data rationale, the data
ingestion/cleaning summary, and the EDA figures with interpretation.

Depends on reports/build_figures.py having already been run (it produces
reports/figures/*.png and reports/content/eda_stats.json), which in turn
depends on scripts/ingest_data.py having produced data/processed/*.

Usage: python reports/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
CONTENT_DIR = ROOT / "reports" / "content"
OUT_PATH = ROOT / "reports" / "AAVAIL_Revenue_Part1_Report.pdf"

with open(CONTENT_DIR / "eda_stats.json") as fh:
    S = json.load(fh)
with open(ROOT / "data" / "processed" / "ingestion_report.json") as fh:
    ING = json.load(fh)["cs-train"]

# ---------------------------------------------------------------------------
# Palette (matches reports/build_figures.py)
# ---------------------------------------------------------------------------
BLUE = colors.HexColor("#2a78d6")
BLUE_DARK = colors.HexColor("#184f95")
WARNING = colors.HexColor("#fab219")
INK_PRIMARY = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
INK_MUTED = colors.HexColor("#898781")
GRIDLINE = colors.HexColor("#e1e0d9")
SURFACE = colors.HexColor("#fcfcfb")


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
base = getSampleStyleSheet()

styles = {
    "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold",
                             fontSize=25, leading=30, textColor=INK_PRIMARY, spaceAfter=6),
    "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica",
                                fontSize=13, leading=18, textColor=INK_SECONDARY, spaceAfter=4),
    "Meta": ParagraphStyle("Meta", parent=base["Normal"], fontName="Helvetica",
                            fontSize=9.5, leading=13, textColor=INK_MUTED),
    "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold",
                          fontSize=16.5, leading=21, textColor=INK_PRIMARY,
                          spaceBefore=4, spaceAfter=10, borderColor=GRIDLINE,
                          borderWidth=0, borderPadding=0),
    "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                          fontSize=12.5, leading=16, textColor=BLUE_DARK,
                          spaceBefore=14, spaceAfter=6),
    "Body": ParagraphStyle("Body", parent=base["Normal"], fontName="Helvetica",
                            fontSize=10, leading=14.5, textColor=INK_PRIMARY,
                            spaceAfter=8, alignment=4),  # justified
    "Bullet": ParagraphStyle("Bullet", parent=base["Normal"], fontName="Helvetica",
                              fontSize=10, leading=14, textColor=INK_PRIMARY,
                              spaceAfter=4),
    "Caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Helvetica-Oblique",
                               fontSize=9, leading=12.5, textColor=INK_SECONDARY,
                               spaceBefore=4, spaceAfter=14),
    "Callout": ParagraphStyle("Callout", parent=base["Normal"], fontName="Helvetica-Bold",
                               fontSize=10.5, leading=15, textColor=BLUE_DARK,
                               spaceBefore=4, spaceAfter=10, leftIndent=10,
                               borderColor=BLUE, borderWidth=0, backColor=colors.HexColor("#eaf2fc"),
                               borderPadding=8),
}

PAGE_W, PAGE_H = LETTER
MARGIN = 0.72 * inch
USABLE_W = PAGE_W - 2 * MARGIN


def fitted_image(path: Path, max_width=USABLE_W, max_height=None) -> Image:
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    w = max_width
    h = w * ih / iw
    if max_height and h > max_height:
        h = max_height
        w = h * iw / ih
    return Image(str(path), width=w, height=h)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(INK_MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, 0.45 * inch, "AAVAIL Revenue Forecasting — Part 1")
    canvas.drawRightString(PAGE_W - MARGIN, 0.45 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(GRIDLINE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 0.62 * inch, PAGE_W - MARGIN, 0.62 * inch)
    canvas.restoreState()


def title_page_deco(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, PAGE_H - 0.35 * inch, PAGE_W, 0.35 * inch, stroke=0, fill=1)
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT_PATH), pagesize=LETTER,
                       leftMargin=MARGIN, rightMargin=MARGIN,
                       topMargin=0.85 * inch, bottomMargin=0.8 * inch,
                       title="AAVAIL Revenue Forecasting — Part 1 Report",
                       author="AAVAIL Data Science")

body_frame = Frame(MARGIN, 0.8 * inch, USABLE_W, PAGE_H - 0.85 * inch - 0.8 * inch, id="body")
title_frame = Frame(MARGIN, 0.8 * inch, USABLE_W, PAGE_H - 0.85 * inch - 0.8 * inch, id="title")

doc.addPageTemplates([
    PageTemplate(id="Title", frames=[title_frame], onPage=title_page_deco),
    PageTemplate(id="Body", frames=[body_frame], onPage=header_footer),
])

story = []
P = lambda text, style="Body": story.append(Paragraph(text, styles[style]))
SP = lambda h=10: story.append(Spacer(1, h))


def bullets(items, style="Bullet"):
    story.append(ListFlowable(
        [ListItem(Paragraph(t, styles[style]), bulletColor=BLUE, value="circle") for t in items],
        bulletType="bullet", start="circle", leftIndent=14,
    ))


def numbered(items, style="Bullet"):
    story.append(ListFlowable(
        [ListItem(Paragraph(t, styles[style])) for t in items],
        bulletType="1", leftIndent=16, start=1,
    ))


def figure(fname, caption, max_height=None):
    story.append(fitted_image(FIG_DIR / fname, max_height=max_height))
    story.append(Paragraph(caption, styles["Caption"]))


def figure_section(heading, fname, caption, body, max_height=None):
    """An H2 heading, its figure, and the interpretation paragraph that
    follows, kept together so nothing ends up orphaned at a page break."""
    story.append(KeepTogether([
        Paragraph(heading, styles["H2"]),
        fitted_image(FIG_DIR / fname, max_height=max_height),
        Paragraph(caption, styles["Caption"]),
        Paragraph(body, styles["Body"]),
    ]))


# ===========================================================================
# TITLE PAGE
# ===========================================================================
story.append(NextPageTemplate("Body"))
SP(70)
P("AAVAIL Revenue Forecasting", "Title")
P("Part 1 — Business Understanding &amp; Exploratory Data Analysis", "Subtitle")
SP(18)
P("A feasibility investigation into predicting next-month revenue, overall and for "
  "AAVAIL's ten highest-revenue countries, ahead of a company-wide move to an "
  "à la carte pricing model.", "Subtitle")
SP(40)
meta_rows = [
    ["Prepared for", "AAVAIL management (finance &amp; operations planning)"],
    ["Data window", f"{S['date_min']} to {S['date_max']} ({S['n_full_months']} complete months, "
                     f"{S['n_partial_months']} partial)"],
    ["Source data", "cs-train — 21 monthly transaction exports, 38+ countries"],
    ["Transactions analyzed", f"{S['n_transactions']:,} (of {ING['rows_read']:,} raw records)"],
]
t = Table([[Paragraph(f"<b>{a}</b>", styles["Meta"]), Paragraph(b, styles["Meta"])] for a, b in meta_rows],
          colWidths=[1.6 * inch, USABLE_W - 1.6 * inch])
t.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LINEBELOW", (0, 0), (-1, -2), 0.5, GRIDLINE),
]))
story.append(t)
story.append(PageBreak())

# ===========================================================================
# 1. BUSINESS OPPORTUNITY
# ===========================================================================
P("1. The Business Opportunity", "H1")
P(
    "AAVAIL built its business on a tiered subscription model. That model showed promise, but "
    "conversations with users — especially outside the US — surfaced consistent friction: "
    "customers who watch inconsistently or occasionally don't want to pay a flat recurring fee "
    "for access they under-use. In response, AAVAIL ran an à la carte, pay-per-title pricing "
    "experiment, mostly in non-US markets, for close to two years. That experiment now has "
    "transaction-level data from a few thousand active users across 38 countries, invoiced in "
    "batches.", "Body"
)
P(
    "Management is now leaning toward rolling the à la carte model out company-wide, but they are "
    "blocked on a planning problem, not a pricing problem: <b>they cannot reliably predict next "
    "month's revenue under this model.</b> Today, individual managers build their own ad hoc "
    "forecasts. That has two costs. First, it consumes manager time that could go toward running "
    "the business. Second, because these managers are not data scientists, their forecasts carry "
    "avoidable error — and that error propagates directly into staffing plans and budget "
    "projections, both of which depend on knowing, in advance, roughly how much revenue is coming.",
    "Body"
)
P(
    "The ask is a revenue-forecasting service that, run at any point in time, predicts total "
    "revenue for the following calendar month — and can do the same for any of the ten countries "
    "that generate the most revenue, since a global number alone will not tell a country manager "
    "what to expect in their own market. Restated as a product requirement: replace a set of "
    "manual, inconsistent, manager-built estimates with one automated, reproducible forecast that "
    "is measurably more accurate, at the two grains (company-wide and per-country) the business "
    "actually plans against.",
    "Body"
)

P("Testable hypotheses", "H2")
P(
    "Framed so each can be checked against data already in hand or against forecasts once a model "
    "exists:", "Body"
)
numbered([
    "<b>H1 — Beats the status quo.</b> A model trained on transaction history predicts next-month "
    "revenue with lower error than a naive baseline (e.g. “same as last month”), which stands "
    "in for the accuracy manual estimates are implicitly competing against.",
    "<b>H2 — Ten countries are enough.</b> A small set of top-revenue countries accounts for the "
    "large majority of total revenue, so restricting modeling scope to ten countries sacrifices "
    "little forecast coverage.",
    "<b>H3 — Activity predicts revenue, not just past revenue.</b> Recent purchasing activity "
    "(purchase counts, unique buyers, unique titles streamed) carries information about near-term "
    "revenue beyond what the historical revenue series alone provides, which would justify "
    "feature-engineered supervised learning rather than a pure univariate time-series model.",
    "<b>H4 — Revenue is seasonal on a predictable calendar.</b> Systematic day-of-week and "
    "holiday-period effects exist, such that a model blind to calendar structure will "
    "mis-predict around those periods.",
    "<b>H5 — Countries do not all behave alike.</b> The shape of monthly revenue (steady vs. "
    "lumpy/concentrated in a few large invoices) differs enough by country that a single "
    "global model may not serve every one of the ten target countries equally well.",
    "<b>H6 — The experiment is not fading.</b> Over its observed history, à la carte revenue is "
    "flat-to-growing rather than declining, which would support management's inclination to "
    "expand rather than retreat from the model.",
])

story.append(PageBreak())

# ===========================================================================
# 2. IDEAL DATA
# ===========================================================================
P("2. Ideal Data for This Problem", "H1")
P(
    "Before looking at what AAVAIL actually has on hand, it's worth being explicit about what a "
    "revenue-forecasting service like this would ideally be built on — that is the yardstick the "
    "real data gets measured against, and it shapes what the feature matrix should try to capture.",
    "Body"
)
P("To forecast next-month revenue, company-wide and per top-10 country, the ideal dataset would provide:", "Body")
bullets([
    "<b>Clean, granular transaction records</b> — one row per purchase with a precise timestamp, "
    "country, customer identifier, product/title identifier, and net amount actually recognized as "
    "revenue (after any return or cancellation), all under one consistent schema over time.",
    "<b>Long, gap-free history spanning multiple full seasonal cycles</b> — at least two to three "
    "years of continuous daily coverage, so weekly and annual/holiday seasonality can be estimated "
    "and then validated against held-out months, rather than guessed from a single cycle.",
    "<b>An unambiguous sign convention for returns and cancellations</b>, kept in the same revenue "
    "stream as ordinary purchases (rather than filed as unrelated accounting entries), so that net "
    "monthly revenue is never in question.",
    "<b>Complete customer identifiers on every transaction</b>, enabling customer-level features — "
    "active customer counts, new-vs-returning mix, repeat-purchase rate — which typically carry "
    "more forward-looking signal than raw transaction counts alone.",
    "<b>Exogenous / calendar context</b>: a holiday and business-closure calendar, marketing or "
    "promotional campaign dates, historical price-list changes, and, for cross-country modeling, "
    "currency exchange rates and basic macroeconomic indicators (e.g. disposable income proxies) "
    "for each market.",
    "<b>A model/pricing-plan flag</b> distinguishing subscription from à la carte revenue on the "
    "same customer base, so any future comparison between the two pricing strategies is a query, "
    "not a fresh data-collection effort.",
    "<b>Low-latency availability</b> — since the service must be able to forecast “at any point in "
    "time,” the underlying data pipeline should reflect transactions with at most a few days of lag.",
])
P(
    "The rationale in each case ties back to the same target: monthly revenue, at two levels of "
    "granularity, predicted before the month is over. Trend and seasonality cannot be separated "
    "from noise without enough continuous history; the country-level requirement means the data "
    "must stay reliable even for markets with a fraction of the UK's volume; and the accuracy bar "
    "management is holding this service to means every plausible source of forward-looking signal "
    "(not just lagged revenue) is worth having available for feature engineering in Part 2.",
    "Body"
)

story.append(PageBreak())

# ===========================================================================
# 3. DATA INGESTION
# ===========================================================================
P("3. Data Ingestion — Automating Extraction from Multiple Sources", "H1")
P(
    "The actual data arrive as monthly JSON exports — 21 files in <font face='Courier'>cs-train</font> "
    f"covering {S['date_min']} through {S['date_max']}, {ING['rows_read']:,} raw transaction records "
    f"across {S['n_countries_observed']} countries after removing non-country labels — plus a second, "
    "structurally identical <font face='Courier'>cs-production</font> export covering the months "
    "immediately after, reserved for later phases of this project. Because more than one export "
    "process produced these files, field names are not uniform across months: price appears as "
    "either <font face='Courier'>price</font> or <font face='Courier'>total_price</font>, and the "
    "streaming fields appear as either <font face='Courier'>stream_id</font>/<font face='Courier'>"
    "times_viewed</font> or <font face='Courier'>StreamID</font>/<font face='Courier'>TimesViewed</font>, "
    "depending on the batch.",
    "Body"
)
P(
    "<font face='Courier'>aavail/ingestion.py</font> handles this as a library: "
    "<font face='Courier'>fetch_data()</font> discovers every file in a directory, normalizes "
    "the schema variants onto one canonical set of columns, coerces and validates types, and "
    "returns a single clean transaction-level DataFrame — the feature matrix EDA and, later, "
    "modeling build on. <font face='Courier'>scripts/ingest_data.py</font> is the automation entry "
    "point: it accepts one or more source directories (validated against both cs-train and "
    "cs-train+cs-production together), runs ingestion, determines the top-N countries by revenue, "
    "and writes the processed transaction table and daily time series to disk.",
    "Body"
)
P("Errors and data-quality issues the ingestion function specifically catches:", "H2")
bullets([
    "<b>Unreadable or malformed files</b> — corrupt JSON, non-list payloads, or files missing a "
    "required field under every known alias are logged and skipped rather than aborting the run.",
    "<b>Invalid dates and prices</b> — records with an out-of-range year/month/day or a "
    "non-numeric price are dropped and counted rather than silently coerced.",
    "<b>Bad-debt adjustment records</b> — a small number of invoices (prefixed <font face='Courier'>"
    "A</font>) are extreme-magnitude negative entries for written-off debt, not customer "
    "purchases; left in, three records alone would swing several individual days' revenue by "
    "tens of thousands of dollars. They are excluded from the revenue series.",
    "<b>Exact duplicate rows</b> — introduced by the export process itself, removed via "
    "<font face='Courier'>drop_duplicates</font>.",
    "<b>Invoice-id noise</b> — invoice numbers carry an incidental letter prefix (e.g. "
    "<font face='Courier'>C512770</font>); the ingestion function splits this into its own field "
    "so invoices can be matched and counted on their numeric id alone, per the case-study guidance.",
])

ing_rows = [
    ["Files found / loaded", f"{ING['files_found']} / {ING['files_loaded']}"],
    ["Raw rows read", f"{ING['rows_read']:,}"],
    ["Dropped — bad-debt adjustment (A-*)", f"{ING['rows_dropped_bad_debt_adjustment']:,}"],
    ["Dropped — exact duplicate row", f"{ING['rows_dropped_exact_duplicate']:,}"],
    ["Dropped — invalid date / price / missing field", f"{ING['rows_dropped_invalid_date'] + ING['rows_dropped_invalid_price'] + ING['rows_dropped_missing_required_field']:,}"],
    ["Rows kept (the feature matrix)", f"{ING['rows_kept']:,}"],
]
t = Table([["Ingestion step", "Count"]] + ing_rows, colWidths=[USABLE_W - 1.6 * inch, 1.6 * inch])
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 0), (-1, 0), BLUE_DARK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, colors.white]),
    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ("LINEBELOW", (0, 0), (-1, -2), 0.5, GRIDLINE),
    ("LINEBELOW", (0, -1), (-1, -1), 1, BLUE_DARK),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(t)
P("Table 1. Ingestion outcome for the cs-train source (see reports/content/eda_stats.json and "
  "data/processed/ingestion_report.json for the machine-readable version).", "Caption")

P(
    f"Even after cleaning, {pct(S['overall_missing_customer_id_rate'])} of transactions carry no "
    "customer_id (Figure 8) — the ingestion function keeps these rows, since they are still valid "
    "revenue events, but this caps how much customer-level feature engineering (e.g. new-vs-returning "
    "buyer mix) Part 2 can rely on without first improving upstream data capture.",
    "Body"
)

# ===========================================================================
# 4. EDA
# ===========================================================================
P("4. Exploratory Data Analysis", "H1")
P(
    "The remainder of this report investigates the relationship between the cleaned transaction "
    "data, the forecasting target (next month's revenue), and the business metric management "
    "actually plans against.",
    "Body"
)

figure_section("4.1 Revenue over time, and a real data-completeness gap",
       "fig01_daily_revenue.png",
       "Figure 1. Daily revenue across all countries, with a 30-day rolling mean. Shaded regions "
       "mark the first (3-day) partial month and an 11-day end-of-year data gap present in both "
       "observed Decembers.",
       "The first month of the export (Nov 2017) contains only three days of data, and both "
       "Decembers in the window stop recording on the 20th and resume on Jan 1st — an identical "
       "11-day gap two years running. That regularity is the tell that this is a genuine reporting "
       "gap around year-end rather than random missingness, and it matters operationally: those days "
       "read as zero revenue, not missing revenue, in a naive daily aggregation, which would bias any "
       "model trained naively across a year boundary. Part 2's feature engineering needs to treat "
       "this window as missing, not as zero demand.")
figure("fig02_monthly_revenue.png",
       "Figure 2. Monthly revenue — the metric this service predicts one month ahead. Amber bars "
       "are the three partial/gapped months excluded from the summary statistics below.")
P(
    f"Across the {S['n_full_months']} complete months, monthly revenue averages "
    f"{money(S['monthly_revenue_mean'])}, ranging from {money(S['monthly_revenue_min'])} to "
    f"{money(S['monthly_revenue_max'])} — a coefficient of variation of "
    f"{S['monthly_revenue_cv']:.2f}. That's the volatility managers are currently trying to "
    "predict by hand: revenue swings roughly ±29% month to month around its average, with a "
    "visible autumn build-up (Sep–Nov 2018) ahead of the holiday season. A month-to-month swing "
    "that large, on a metric staffing and budget plans depend on, is exactly the kind of pattern "
    "a trend/seasonality-aware model should be able to anticipate better than a flat guess.",
    "Body"
)

figure_section("4.2 Revenue concentration by country — sizing the top-10 scope",
       "fig03_country_revenue_ranked.png",
       "Figure 3. Total revenue by country. Left: the UK alone against all 40 other countries "
       "combined. Right: the other nine modeled countries, ranked, with the UK removed so their "
       "relative sizes are actually visible.",
       f"The UK — AAVAIL's home market — accounts for {pct(S['uk_share_of_revenue'])} of all revenue "
       f"in this data on its own. The ten countries selected for modeling together capture "
       f"{pct(S['top10_share_of_revenue'])} of total revenue across all {S['n_countries_observed']} "
       f"countries observed, which is direct support for H2: scoping the service to ten countries "
       "gives up very little forecast coverage. Restricted to non-UK markets, the other nine modeled "
       f"countries still account for {pct(S['top9_ex_uk_share_of_non_uk_revenue'])} of all non-UK "
       "revenue — so within \"the rest of the world,\" the selection is not just UK-plus-noise; it is "
       "genuinely capturing the international markets the à la carte experiment was aimed at.")

figure_section("4.3 Country-level trends are not all the same shape",
       "fig04_top10_country_trends.png",
       "Figure 4. Monthly revenue trend per top-10 country, each on its own scale.",
       "The UK, Germany, France, and the Netherlands show continuous month-over-month activity with "
       "a recognizable trend — the kind of series a standard time-series or regression approach "
       "handles well. Norway, Hong Kong, Portugal, and Singapore look different: long stretches at "
       "zero punctuated by a single dominant month. That pattern means their \"top-10 by cumulative "
       "revenue\" ranking is driven by one or two large invoices rather than sustained demand — "
       "evidence for H5. Part 2 should not assume one modeling approach generalizes across all ten "
       "countries without checking per-country error, and may want a specific plan (e.g. wider "
       "prediction intervals, or a simpler baseline) for the lumpy markets.")

figure_section("4.4 Purchase activity as a leading signal for revenue",
       "fig05_purchases_vs_revenue.png",
       "Figure 5. Daily distinct purchases vs. daily revenue, all countries, with a linear fit.",
       f"Day to day, the relationship is real but noisy (r = {S['corr_purchases_revenue']:.2f}) — a "
       "handful of very large single invoices (visible as the points well above the trend line, "
       "consistent with the heavy right tail in Figure 7) mean purchase count alone is an imperfect "
       f"proxy for revenue at daily granularity. At monthly granularity the same relationship "
       f"strengthens substantially: purchases correlate with revenue at r = "
       f"{S['monthly_corr_purchases_revenue']:.2f}, unique titles streamed at r = "
       f"{S['monthly_corr_streams_revenue']:.2f}, and unique customers at r = "
       f"{S['monthly_corr_customers_revenue']:.2f}. That supports H3: activity-based features carry "
       "real signal at the monthly grain this service actually predicts at, which argues for a "
       "feature-engineered supervised approach over a model that only ever looks at past revenue.",
       max_height=3.3 * inch)

figure_section("4.5 Weekly seasonality",
       "fig06_day_of_week.png",
       "Figure 6. Daily revenue by day of week (the end-of-year gap excluded so it doesn't distort "
       "the Saturday/Sunday comparison).",
       f"Saturday revenue is essentially zero (median {money(S['saturday_median_revenue'])}, against a "
       f"weekday median of {money(S['weekday_median_revenue'])}); Sunday is depressed but not zero. "
       "This is a strong, mechanical calendar effect — direct evidence for H4 — and a cheap, "
       "high-value feature for Part 2: any model, even a simple one, should know what day of the "
       "week a given date falls on.")

figure_section("4.6 Transaction price: right-skewed, with a heavy tail",
       "fig07_price_distribution.png",
       "Figure 7. Distribution of per-transaction price (log x-axis).",
       f"The typical transaction is small — a {money(S['price_median'])} median against a "
       f"{money(S['price_mean'])} mean, with the 95th percentile at {money(S['price_p95'])} — but the "
       "distribution has a real tail out past $10,000 (Table 1's outlier rows, and the spikes visible "
       "in Figure 1). Aggregate daily/monthly revenue is therefore sensitive to a small number of "
       "large invoices. Part 2 should favor loss functions and validation metrics that are not overly "
       "sensitive to a handful of extreme days, and should consider whether those large invoices "
       "represent a distinct transaction type (e.g. bulk/business purchases) worth a feature of "
       "their own.",
       max_height=3.3 * inch)

figure_section("4.7 Data-quality caveat: incomplete customer identification",
       "fig08_missing_customer_id.png",
       "Figure 8. Share of transactions with no recorded customer_id, by month.",
       f"customer_id is missing on {pct(S['overall_missing_customer_id_rate'])} of transactions "
       "overall, and the rate is not stable — it spikes above 35% around the Nov/Dec 2018 peak "
       "season. Revenue itself is unaffected (every row has a price), but this is exactly the gap "
       "called out in Section 2: it limits how much customer-level signal (repeat-purchase rate, "
       "active-customer counts) Part 2's feature engineering can extract with confidence, "
       "particularly for the highest-volume months.",
       max_height=3.3 * inch)

story.append(PageBreak())

# ===========================================================================
# 5. SUMMARY
# ===========================================================================
P("5. Summary of Findings and Implications for Part 2", "H1")

P("Where the data landed on each hypothesis", "H2")
bullets([
    "<b>H2 (ten countries are enough) — supported.</b> The top 10 countries capture "
    f"{pct(S['top10_share_of_revenue'])} of all revenue; scoping the service to them is a safe "
    "simplification.",
    "<b>H3 (activity predicts revenue) — supported at the monthly grain.</b> Purchases, unique "
    f"titles, and unique customers all correlate with monthly revenue at r ≥ "
    f"{min(S['monthly_corr_purchases_revenue'], S['monthly_corr_streams_revenue'], S['monthly_corr_customers_revenue']):.2f}, "
    "well above their noisy daily-level correlation.",
    "<b>H4 (predictable calendar seasonality) — supported.</b> Saturday revenue is effectively "
    "zero and the year-end window has a genuine multi-day reporting gap in both observed years.",
    "<b>H5 (countries are not interchangeable) — supported.</b> Four of the ten modeled countries "
    "show continuous trend-like revenue; four others are dominated by one or two outsized months.",
    "<b>H1 (beats the status quo) and H6 (not fading)</b> — not yet testable from EDA alone; both "
    "require an actual model and a longer look at the full monthly series, and are the direct "
    "subject of Part 2.",
])

P("What this means for Part 2 (feature engineering &amp; modeling)", "H2")
bullets([
    "Aggregate transactions to a daily series (as <font face='Courier'>convert_to_ts()</font> "
    "already does) and engineer lag/rolling-window revenue and activity features — Section 4.4 "
    "shows activity counts carry signal beyond lagged revenue alone.",
    "Add calendar features (day-of-week, and an explicit end-of-year flag) rather than relying on "
    "the model to infer them — Section 4.5 and 4.1.",
    "Treat the two 11-day December gaps as missing data, not as zero-demand days, when building "
    "training windows that cross a year boundary.",
    "Evaluate the model per country as well as in aggregate, given the steady-vs-lumpy split in "
    "Section 4.3; a single global model's overall error can hide poor performance on the lumpy "
    "markets.",
    "Use an evaluation metric and, if relevant, a loss function that is not dominated by the "
    "handful of extreme-value invoices identified in Section 4.6.",
    "Scope customer-level features to what the data can actually support given the missing-"
    "customer_id rate in Section 4.7, or flag forecasts in high-missingness months as lower-"
    "confidence.",
])

SP(8)
story.append(Paragraph(
    "All figures, statistics, and processed data referenced above are reproducible via "
    "<font face='Courier'>scripts/ingest_data.py</font> and <font face='Courier'>"
    "reports/build_figures.py</font>; source numbers live in "
    "<font face='Courier'>reports/content/eda_stats.json</font> and "
    "<font face='Courier'>data/processed/ingestion_report.json</font>.",
    styles["Callout"]
))

doc.build(story)
print("wrote", OUT_PATH)
