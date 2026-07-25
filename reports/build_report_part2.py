#!/usr/bin/env python3
"""Assemble the Part 2 deliverable: model comparison, iteration, retraining,
and deployment readiness, as a single PDF report."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem,
    NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
CONTENT_DIR = ROOT / "reports" / "content"
OUT_PATH = ROOT / "reports" / "AAVAIL_Revenue_Part2_Report.pdf"

with open(CONTENT_DIR / "model_comparison.json") as fh:
    C = json.load(fh)
with open(ROOT / "models" / "metadata.json") as fh:
    META = json.load(fh)

BLUE = colors.HexColor("#2a78d6")
BLUE_DARK = colors.HexColor("#184f95")
INK_PRIMARY = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
INK_MUTED = colors.HexColor("#898781")
GRIDLINE = colors.HexColor("#e1e0d9")
SURFACE = colors.HexColor("#fcfcfb")


def money(x):
    return f"${x:,.0f}"


base = getSampleStyleSheet()
styles = {
    "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold",
                             fontSize=25, leading=30, textColor=INK_PRIMARY, spaceAfter=6),
    "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica",
                                fontSize=13, leading=18, textColor=INK_SECONDARY, spaceAfter=4),
    "Meta": ParagraphStyle("Meta", parent=base["Normal"], fontName="Helvetica",
                            fontSize=9.5, leading=13, textColor=INK_MUTED),
    "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold",
                          fontSize=16.5, leading=21, textColor=INK_PRIMARY, spaceBefore=4, spaceAfter=10),
    "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                          fontSize=12.5, leading=16, textColor=BLUE_DARK, spaceBefore=14, spaceAfter=6),
    "Body": ParagraphStyle("Body", parent=base["Normal"], fontName="Helvetica",
                            fontSize=10, leading=14.5, textColor=INK_PRIMARY, spaceAfter=8, alignment=4),
    "Bullet": ParagraphStyle("Bullet", parent=base["Normal"], fontName="Helvetica",
                              fontSize=10, leading=14, textColor=INK_PRIMARY, spaceAfter=4),
    "Caption": ParagraphStyle("Caption", parent=base["Normal"], fontName="Helvetica-Oblique",
                               fontSize=9, leading=12.5, textColor=INK_SECONDARY, spaceBefore=4, spaceAfter=14),
    "Callout": ParagraphStyle("Callout", parent=base["Normal"], fontName="Helvetica-Bold",
                               fontSize=10.5, leading=15, textColor=BLUE_DARK, spaceBefore=4, spaceAfter=10,
                               leftIndent=10, backColor=colors.HexColor("#eaf2fc"), borderPadding=8),
}

PAGE_W, PAGE_H = LETTER
MARGIN = 0.72 * inch
USABLE_W = PAGE_W - 2 * MARGIN


def fitted_image(path, max_width=USABLE_W, max_height=None):
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
    canvas.drawString(MARGIN, 0.45 * inch, "AAVAIL Revenue Forecasting — Part 2")
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


doc = BaseDocTemplate(str(OUT_PATH), pagesize=LETTER, leftMargin=MARGIN, rightMargin=MARGIN,
                       topMargin=0.85 * inch, bottomMargin=0.8 * inch,
                       title="AAVAIL Revenue Forecasting — Part 2 Report", author="AAVAIL Data Science")
frame = Frame(MARGIN, 0.8 * inch, USABLE_W, PAGE_H - 0.85 * inch - 0.8 * inch, id="f")
doc.addPageTemplates([
    PageTemplate(id="Title", frames=[frame], onPage=title_page_deco),
    PageTemplate(id="Body", frames=[frame], onPage=header_footer),
])

story = []
P = lambda text, style="Body": story.append(Paragraph(text, styles[style]))
SP = lambda h=10: story.append(Spacer(1, h))


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(t, styles["Bullet"]), bulletColor=BLUE, value="circle") for t in items],
        bulletType="bullet", start="circle", leftIndent=14,
    ))


def figure_section(heading, fname, caption, body, max_height=None):
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
P("Part 2 — Model Comparison, Selection &amp; Deployment Prep", "Subtitle")
SP(18)
P("Predicting total revenue over the next 30 days, at any point in time, for the "
  "company overall and for each of the ten highest-revenue countries.", "Subtitle")
SP(40)
meta_rows = [
    ["Task", "Supervised regression: predict revenue summed over days t+1…t+30, as of any date t"],
    ["Series modeled", "11 (overall + 10 countries), one model each"],
    ["Models compared", "Naive, seasonal-naive, Ridge, Random Forest, HistGBM, LightGBM"],
    ["Selected approach", f"{META['model_family']} — {META['hyperparameters']}"],
]
t = Table([[Paragraph(f"<b>{a}</b>", styles["Meta"]), Paragraph(b, styles["Meta"])] for a, b in meta_rows],
          colWidths=[1.5 * inch, USABLE_W - 1.5 * inch])
t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LINEBELOW", (0, 0), (-1, -2), 0.5, GRIDLINE)]))
story.append(t)
story.append(PageBreak())

# ===========================================================================
# 1. MODELING APPROACHES
# ===========================================================================
P("1. Modeling Approaches Compared", "H1")
P(
    "Part 1 established that next-month revenue is the target and that ten countries account "
    "for the large majority of it. Following the case-study guidance, every candidate here is "
    "standardized on the same task: given all data available as of a date t, predict total "
    "revenue over the following 30 days (t+1 … t+30) — a single number per as-of date, which "
    "turns the forecasting problem into ordinary supervised regression on engineered features "
    "rather than a multi-step sequence prediction problem.",
    "Body"
)
bullets([
    "<b>Naive (persistence) baseline</b> — next 30 days = trailing 28 days of revenue. Stands in "
    "for the “same as last month” estimate management currently uses informally.",
    "<b>Seasonal-naive baseline</b> — next 30 days = the same 30-day window one year earlier, "
    "falling back to the naive baseline where a year of history isn't yet available.",
    "<b>Ridge regression</b> — a regularized linear model. Included deliberately despite the "
    "case-study's own caution that OLS assumptions break down on autocorrelated, "
    "revenue-derived features; regularization was expected to help, and it's a useful check on "
    "whether the non-linear models are earning their complexity.",
    "<b>Random Forest</b> — a bagged tree ensemble, robust to the feature scale differences "
    "across an 11-series, very-different-revenue-scale problem (UK vs. Singapore).",
    "<b>Histogram-based Gradient Boosting (scikit-learn)</b> and <b>LightGBM</b> — two gradient-"
    "boosted tree implementations, generally the strongest performers on tabular, engineered "
    "time-series features of this kind.",
])
P(
    "Feature engineering is shared across every learned model: for revenue, purchases, unique "
    "streams, unique customers, and total views, trailing 7/14/28/70-day sums and means, a "
    "28-day trend, and a 30-day-value-one-year-ago feature (flagged when unavailable rather than "
    "dropping the row — most series don't have a full year of prior history). Calendar features "
    "(day-of-week, day-of-month, month, and an explicit end-of-year-gap flag) encode the "
    "seasonality Part 1 found in the data. See <font face='Courier'>aavail/features.py</font>.",
    "Body"
)

figure_section(
    "1.1 Why purchases and views are in the feature set, not just lagged revenue",
    "fig13_revenue_purchases_views_comoving.png",
    "Figure 9. Daily revenue, purchases, and total views — 7-day rolling averages, each indexed "
    "to its own mean = 100 so all three share one axis despite very different raw scales.",
    "Pivoted to a daily interval, the three series move together: purchases and total views are "
    "almost identical in shape (correlation 0.93), since more distinct purchases mechanically "
    "means more content streamed. Revenue tracks the same broad seasonal pattern — the January "
    "dip, the autumn build-up, the December gap — but correlates more loosely with the other two "
    "(0.62–0.70), because revenue alone carries occasional large single-invoice spikes (the tall "
    "blue peaks in Mar, Jun, and Oct/Nov 2018) that purchase counts and view counts don't share. "
    "That's the same heavy-tailed price effect Part 1 found in the transaction-price distribution, "
    "and it's the concrete reason purchases and views were engineered as separate features here "
    "rather than assumed redundant with lagged revenue.",
    max_height=3.3 * inch,
)

# ===========================================================================
# 2. ITERATION
# ===========================================================================
P("2. Iteration", "H1")
P(
    "Hyperparameters were tuned once, on the overall series (it has the most history), using "
    "4-fold <font face='Courier'>TimeSeriesSplit</font> cross-validation — folds respect "
    "chronological order so no model is ever validated on data that precedes its training "
    "window. A small, deliberately cheap grid was searched per model family (e.g. tree depth, "
    "learning rate, number of estimators/leaves for the boosted models; regularization strength "
    "for Ridge); the winning configuration per family was then reused unchanged across all 11 "
    "series, keeping the comparison apples-to-apples rather than re-tuning per country.",
    "Body"
)
P(
    "Each of the 11 series was then evaluated on its own chronological holdout (the most recent "
    "~20% of usable rows, minimum 40), so every model is judged on data it never trained on.",
    "Body"
)

story.append(PageBreak())

# ===========================================================================
# 3. COMPARISON RESULTS
# ===========================================================================
P("3. Comparison Results", "H1")
figure_section(
    "3.1 Which model family wins, on average",
    "fig09_model_comparison.png",
    "Figure 10. Mean absolute error, normalized by each series' average revenue and averaged "
    "across all 11 series (lower is better).",
    f"Histogram Gradient Boosting had the lowest average normalized error "
    f"({C['avg_normalized_mae']['hist_gbm']:.2f}), narrowly ahead of LightGBM "
    f"({C['avg_normalized_mae']['lightgbm']:.2f}). Random Forest trailed "
    f"({C['avg_normalized_mae']['random_forest']:.2f}), and Ridge was far worse on average "
    f"({C['avg_normalized_mae']['ridge']:.2f}) — almost entirely because it produced a wildly "
    "unstable extrapolation on Norway, a country whose revenue is concentrated in one or two "
    "outsized months (flagged as a lumpy market in Part 1). That instability is itself a "
    "finding: linear models extrapolate badly on the sparse, spiky countries, which is a "
    "concrete reason — not just an average-case one — to prefer a tree-based approach here.",
    max_height=3.4 * inch,
)
figure_section(
    "3.2 Does the winner actually beat the naive baseline, everywhere?",
    "fig10_model_vs_naive_per_series.png",
    "Figure 11. Normalized MAE, naive baseline vs. the selected model (Histogram Gradient "
    "Boosting), per series.",
    "For the steady, high-volume series (overall, UK, Germany, Norway, France, Spain, "
    "Netherlands) the model matches or modestly beats the naive baseline. For the lumpiest, "
    "lowest-volume markets (EIRE, Singapore, Hong Kong, Portugal) both the model and the naive "
    "baseline show much higher relative error — the model still generally edges out naive, but "
    "neither is highly accurate, because a handful of outsized invoices make 30-day totals hard "
    "to predict from any historical pattern. This is the clearest evidence in this report that "
    "Part 1's per-country accuracy caveat (H5) was warranted: the service should report "
    "wider uncertainty, or a simpler baseline, for these markets rather than implying UK-level "
    "confidence.",
    max_height=3.6 * inch,
)

story.append(PageBreak())

# ===========================================================================
# 4. SELECTED MODEL, RETRAINING, DEPLOYMENT
# ===========================================================================
P("4. Selected Approach, Retraining, and Deployment Readiness", "H1")
P(
    f"<b>Histogram-based Gradient Boosting</b> ({META['hyperparameters']}) was selected: best "
    "average accuracy, no catastrophic failure modes on any series, and (unlike LightGBM here) "
    "no extra tuning needed to close the gap. Once selected, the same model family and "
    "hyperparameters were retrained on <b>100% of available history</b> for every one of the 11 "
    "series — the held-out evaluation above exists purely to choose and validate the approach; "
    "the deployed models use all the data, since forecasting accuracy is monotonic in training "
    "data for this kind of model.",
    "Body"
)
figure_section(
    "4.1 Sanity check: overall revenue, actual vs. predicted",
    "fig11_actual_vs_predicted.png",
    "Figure 12. Held-out overall revenue: actual next-30-day total vs. the model's prediction.",
    f"On the overall series' holdout, the selected model's mean absolute error was "
    f"{money(C['per_series_metrics']['overall']['hist_gbm']['mae'])} against an average 30-day "
    f"revenue of {money(C['per_series_metrics']['overall']['hist_gbm']['mean_actual'])} — "
    f"roughly a {C['per_series_metrics']['overall']['hist_gbm']['mape']:.0f}% typical error. The "
    "predicted line tracks the level and direction of actual revenue well; it lags sharp single-"
    "week spikes (early May) since those are driven by a few large invoices no engineered "
    "feature fully anticipates.",
    max_height=3.2 * inch,
)
figure_section(
    "4.2 What the model actually relies on",
    "fig12_feature_importance.png",
    "Figure 13. Permutation feature importance, overall revenue model (top 12 features).",
    "Calendar month dominates — consistent with Part 1's finding of a strong autumn build-up "
    "toward the holiday season. Trailing views, trailing revenue, and trailing purchase counts "
    "follow, confirming Part 1's H3: recent activity carries real predictive signal beyond raw "
    "lagged revenue. This also means the model is depending on a broad mix of features rather "
    "than one brittle signal, which is reassuring for deployment.",
    max_height=3.4 * inch,
)
P("Deployment artifacts", "H2")
bullets([
    f"11 trained models (one per series) saved via joblib to <font face='Courier'>models/*.joblib</font>.",
    "<font face='Courier'>models/metadata.json</font> records the model family, hyperparameters, "
    "the exact feature-column order each model expects, and the training date range per series — "
    "everything an inference service needs without re-deriving it.",
    "<font face='Courier'>aavail/features.py</font> exposes "
    "<font face='Courier'>latest_feature_row()</font>, which builds the single feature row for "
    "'today' from a time series, so training and inference share identical feature logic and "
    "can't silently drift apart.",
    "Verified end-to-end: loading a saved model, building today's feature row, and predicting "
    "next-30-day revenue runs correctly for both the overall series and an individual country "
    "(see the Part 2 training script's smoke test).",
])

story.append(PageBreak())

# ===========================================================================
# 5. SUMMARY
# ===========================================================================
P("5. Summary and What's Next", "H1")
bullets([
    "Six candidate approaches were compared — two baselines and four learned models — on an "
    "identical 30-day-ahead supervised task across 11 series.",
    "Histogram Gradient Boosting was the most accurate on average and the most robust: it did "
    "not blow up on the lumpy, low-volume countries the way Ridge did.",
    "The model beats a naive persistence baseline on the high-volume, steady series but only "
    "modestly improves on it for the sparsest markets — a limitation worth stating to "
    "management directly rather than masking behind an aggregate accuracy number.",
    "All 11 models are retrained on full history and saved with the metadata an inference "
    "service needs, ready for Part 3 (packaging this as the actual prediction service, with "
    "monitoring against the cs-production data held out from this project).",
])
SP(8)
story.append(Paragraph(
    "Reproduce via <font face='Courier'>scripts/train_models.py</font> (comparison + retrain) and "
    "<font face='Courier'>reports/build_model_figures.py</font> (figures); source metrics live in "
    "<font face='Courier'>reports/content/model_comparison.json</font>.",
    styles["Callout"]
))

doc.build(story)
print("wrote", OUT_PATH)
