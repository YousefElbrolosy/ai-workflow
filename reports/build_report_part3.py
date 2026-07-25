#!/usr/bin/env python3
"""Assemble the Part 3 deliverable: API/Docker/TDD architecture, the
production simulation, and the post-production performance analysis, as a
single PDF report."""

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
OUT_PATH = ROOT / "reports" / "AAVAIL_Revenue_Part3_Report.pdf"

with open(CONTENT_DIR / "post_production_stats.json") as fh:
    STATS = json.load(fh)
with open(CONTENT_DIR / "post_production_summary.json") as fh:
    SUMMARY = {row["country"]: row for row in json.load(fh)}
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
    canvas.drawString(MARGIN, 0.45 * inch, "AAVAIL Revenue Forecasting — Part 3")
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
                       title="AAVAIL Revenue Forecasting — Part 3 Report", author="AAVAIL Data Science")
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
P("Part 3 — Deployment, Monitoring &amp; Post-Production Analysis", "Subtitle")
SP(18)
P("Serving the Part 2 model behind a tested, Dockerized API; simulating real "
  "day-by-day usage; and comparing predictions against revenue observed since.", "Subtitle")
SP(40)
meta_rows = [
    ["API", "Flask: POST /train, POST /predict, GET /drift, GET /logs, GET /health"],
    ["Packaging", "Docker image bundling the API, the trained models, and the unit test suite"],
    ["Simulation", f"{STATS['n_logged']} logged /predict calls; {STATS['n_evaluated']} have a fully-observed actual"],
    ["Gold-standard check", (f"overall MAE {money(STATS['overall_mae'])} ({STATS['overall_mape']:.1f}% typical error)"
                              if STATS.get("overall_mae") is not None else "n/a")],
]
t = Table([[Paragraph(f"<b>{a}</b>", styles["Meta"]), Paragraph(b, styles["Meta"])] for a, b in meta_rows],
          colWidths=[1.5 * inch, USABLE_W - 1.5 * inch])
t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LINEBELOW", (0, 0), (-1, -2), 0.5, GRIDLINE)]))
story.append(t)
story.append(PageBreak())

# ===========================================================================
# 1. API DESIGN
# ===========================================================================
P("1. API Design", "H1")
P(
    "The API standardizes on the same task Part 2 trained for: given a country and an as-of "
    "date, return the total revenue predicted over the following 30 days. Deliberately, "
    "<b>predict does not assume 'today' is the latest date in the data</b> — leadership makes "
    "projections on the 15th as often as at month-end, so every request rebuilds the engineered "
    "feature row for the exact date supplied, truncating history to that date first "
    "(<font face='Courier'>aavail/features.py: feature_row_for_date</font>) rather than serving "
    "a single precomputed forecast.",
    "Body"
)
bullets([
    "<b>POST /train</b> — retrains one model per series (overall + the current top-N countries "
    "by revenue) on every file in the given data directories, optionally truncated to an "
    "<font face='Courier'>as_of</font> date. Per the case study's own suggestion, this is "
    "intentionally simple: point it at a directory, it retrains on everything there, cheap "
    "enough to run nightly or weekly with a handful of active users.",
    "<b>POST /predict</b> — <font face='Courier'>{country, date}</font> → predicted 30-day "
    "revenue, plus the model's training timestamp and an embedded drift verdict.",
    "<b>GET /drift</b> — the same feature-drift check as /predict, callable on its own (e.g. "
    "from a monitoring cron) without generating a prediction log entry.",
    "<b>GET /logs</b> — read back logged train/predict/drift events, filterable by type and "
    "date range — this is the audit trail the post-production analysis below is built from.",
])
P("Anticipating scale, load, and drift", "H2")
bullets([
    "<b>Scale/load:</b> trained models and ingested transactions are cached in memory after "
    "first use and only invalidated on the next successful /train — repeated predictions don't "
    "re-read 21+ JSON files per request. Verified directly in "
    "<font face='Courier'>tests/test_api.py</font> "
    "(<font face='Courier'>test_predict_reuses_cached_transactions_across_calls</font>). The "
    "case study is explicit that this service has only a handful of active users, so a "
    "single-process in-memory cache was chosen deliberately over a heavier store — correctness "
    "and simplicity over throughput this project doesn't need.",
    "<b>Drift:</b> each trained model stores a per-feature mean/std snapshot "
    "(<font face='Courier'>aavail/monitoring.py</font>). Both /predict and /drift compare the "
    "requested date's engineered features against that snapshot and flag anything more than 3 "
    "training standard deviations away — cheap, dependency-free, and exercised directly in "
    "Section 4 below.",
])

story.append(PageBreak())

# ===========================================================================
# 2. TDD
# ===========================================================================
P("2. Test-Driven Iteration", "H1")
P(
    "16 tests in <font face='Courier'>tests/test_api.py</font> (26 total with Part 1's ingestion "
    "tests) pin down the API's contract before treating it as done. Beyond the happy path, the "
    "suite specifically covers the edge cases this business problem actually has:",
    "Body"
)
bullets([
    "Predicting for a <b>mid-month date</b> (the 15th), not just the latest available date — "
    "the exact requirement leadership stated.",
    "<b>Case-insensitive country matching</b>, and a clear 400 with the valid option list for an "
    "unrecognized country, rather than a generic 500.",
    "A date with <b>insufficient trailing history</b> (under 70 days of prior data) and a date "
    "<b>outside the ingested range entirely</b> both fail with an explanatory 400 instead of a "
    "silent wrong answer.",
    "Calling /predict with <b>no trained model present</b> returns 503, not a crash.",
    "/train <b>invalidates</b> every in-memory cache, and a subsequent /predict is proven to "
    "reuse the transaction cache across two different countries in the same session — the two "
    "load/scale properties above are asserted, not just assumed.",
])

story.append(PageBreak())

# ===========================================================================
# 3. DOCKER
# ===========================================================================
P("3. Docker Packaging", "H1")
P(
    "The image bundles exactly what the case study asks for: the API, the trained models, and "
    "the unit tests, so the same artifact that serves traffic can also prove itself.",
    "Body"
)
bullets([
    "<font face='Courier'>python:3.11-slim</font> base; <font face='Courier'>requirements.txt</font> "
    "pins flask, gunicorn, pandas, scikit-learn, lightgbm, joblib, and pytest to the exact "
    "versions used in Part 2, so the served model's predictions can't drift from a library "
    "version change.",
    "<font face='Courier'>aavail/</font>, <font face='Courier'>api/</font>, "
    "<font face='Courier'>models/</font> (the Part 2 artifacts), "
    "<font face='Courier'>cs-train/</font>, and <font face='Courier'>tests/</font> are all "
    "copied into the image.",
    "<font face='Courier'>docker run aavail-revenue-api pytest tests/</font> runs the full suite "
    "inside the container — the same bundle that serves requests is the one being tested, not a "
    "look-alike host environment.",
    "Serving uses gunicorn (2 workers) rather than Flask's dev server, on port 8080.",
    "cs-production (the holdout months used for the simulation below) is intentionally "
    "<b>not</b> baked into the image — it's mounted as a read-only volume at run time "
    "(<font face='Courier'>-v $(pwd)/cs-production:/app/cs-production</font>), modeling how new "
    "production data actually arrives after a deploy rather than being part of the shipped "
    "artifact.",
])

story.append(PageBreak())

# ===========================================================================
# 4. SIMULATION
# ===========================================================================
P("4. Simulating Production Usage", "H1")
P(
    "<font face='Courier'>scripts/simulate_queries.py</font> walks forward day by day and calls "
    "the real, running (Dockerized) API's /predict for every series, as if that day were "
    "'today'. Because /predict rebuilds features from history truncated to the requested date, "
    "no future data can leak into any single prediction. A retrain is triggered periodically "
    "(weekly, via /train with a matching <font face='Courier'>as_of</font>), mimicking a "
    "nightly/weekly retraining job rather than one model frozen for the entire simulated period.",
    "Body"
)
P(
    f"Every call was captured by the API's own logs: {STATS['n_logged']} /predict events were "
    f"logged, of which {STATS['n_evaluated']} now have a fully-observed 30-day actual to compare "
    "against (the remainder fall too close to the end of the available data for their 30-day "
    "window to be fully known yet — an expected, honest limitation of evaluating a forward-"
    "looking forecast against a fixed historical dataset, not a bug).",
    "Body"
)

story.append(PageBreak())

# ===========================================================================
# 5. POST-PRODUCTION ANALYSIS
# ===========================================================================
P("5. Post-Production Analysis: Predictions vs. the Gold Standard", "H1")
figure_section(
    "5.1 Overall revenue: predicted vs. known",
    "fig14_production_predicted_vs_known.png",
    "Figure 14. Every logged overall-series prediction against the revenue that was actually "
    "observed for that 30-day window.",
    f"Against a gold standard of {STATS['n_evaluated']} predictions with a fully-known actual, "
    f"the deployed overall model's mean absolute error was "
    f"{money(STATS['overall_mae']) if STATS['overall_mae'] else float('nan')} — a typical error of "
    f"{STATS['overall_mape']:.1f}% — while being queried under the same realistic constraints "
    "management actually operates under (asking mid-month, not just at month-end, and behind a "
    "periodically-retrained model rather than one fit once and never touched again). More "
    "importantly than the average, the chart reveals a <b>systematic overprediction bias from "
    "early September through early November 2019</b>: the model consistently forecast more "
    "revenue than actually arrived for about two months, before actual revenue caught up (and "
    "briefly overtook the forecast) in early November. Part 2 found that calendar month was, by "
    "a wide margin, the model's most important feature — driven by a large Oct/Nov 2018 holiday "
    "spike in the training data. This looks like exactly that dependency showing up in "
    "production: the model expected 2019's autumn build-up to echo 2018's, and 2019 built up "
    "more gradually. That is a concrete, actionable finding for management, not just an accuracy "
    "number — a forecast this reliant on 'last year's seasonal shape repeating' should carry a "
    "wider uncertainty band heading into every Q4.",
    max_height=3.2 * inch,
)
figure_section(
    "5.2 Does error track the business metric itself?",
    "fig15_error_vs_business_metric.png",
    "Figure 15. Prediction error (predicted − known) against known next-30-day revenue, overall "
    "and per country.",
    f"The correlation between absolute error and revenue level is "
    f"{STATS['corr_abs_error_vs_revenue']:.2f}, and the shape here is the same finding as Figure "
    "14 from a different angle: errors are positive (overprediction) in the "
    "$180K–230K range — the September/early-October weeks — and swing negative "
    "(underprediction) above roughly $250K, where November's actual revenue ultimately landed. "
    "This is not random noise scaling with size; it's the same autumn-timing miss, visible "
    "across the individual countries (grey) as well as the overall series (blue), confirming "
    "it's a shared seasonal-anchoring effect rather than one series' fluke.",
    max_height=3.4 * inch,
)

P("Per-series accuracy in production", "H2")
rows = [["Series", "n evaluated", "MAE", "MAPE"]]
for name, row in sorted(SUMMARY.items(), key=lambda kv: -kv[1]["mean_actual"]):
    rows.append([name, str(row["n"]), money(row["mae"]), f"{row['mape']:.1f}%"])
t = Table(rows, colWidths=[USABLE_W * 0.34, USABLE_W * 0.18, USABLE_W * 0.24, USABLE_W * 0.24])
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 0), (-1, 0), BLUE_DARK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, colors.white]),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("LINEBELOW", (0, 0), (-1, -2), 0.5, GRIDLINE),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(t)
P("Table 2. Mean absolute error and mean absolute percentage error, per series, over every "
  "production-simulated prediction with a known actual.", "Caption")
P(
    "MAPE is not a trustworthy number for Portugal and Singapore here (428% and over 11,000%) — "
    "the same issue Part 2 ran into: a percentage error explodes when the known actual it divides "
    "by is close to zero, which happens often for these sparse, lumpy markets (Part 1's finding, "
    "again). Their MAE figures — a few hundred dollars — are the honest read: small in absolute "
    "terms, just not meaningfully expressible as a percentage of a near-zero base.",
    "Body"
)

story.append(PageBreak())

# ===========================================================================
# 6. FINDINGS
# ===========================================================================
P("6. Summary and Recommendations", "H1")
bullets([
    "The API, Docker packaging, and TDD suite are complete and verified: the containerized "
    "image serves /train, /predict, /drift, and /logs, and runs its own test suite from inside "
    "the same image that ships.",
    "The production simulation validated the exact usage pattern management needs — projections "
    "made on arbitrary dates, not only at month-end — behind a periodically retrained model, "
    "with zero look-ahead leakage by construction.",
    f"Measured against revenue observed after the fact, the deployed model's overall accuracy "
    f"({STATS['overall_mape']:.1f}% typical error) is close to what Part 2's offline holdout "
    "evaluation predicted (16.3%) — a useful confirmation the offline evaluation wasn't overly "
    "optimistic, and that a periodically-retrained model in production doesn't quietly perform "
    "worse than the one-off evaluation suggested.",
    "The one real miss the gold-standard comparison surfaced: a two-month <b>overprediction "
    "streak (Sep–early Nov 2019)</b>, most plausibly explained by the model over-anchoring on "
    "2018's outsized holiday build-up (Part 2's top feature was calendar month). This is the "
    "kind of failure mode only a post-production comparison against real outcomes reveals — it "
    "wasn't visible in Part 2's single historical holdout.",
    "Recommended next steps: schedule /train on a real cadence (cron or the container "
    "orchestrator's job scheduler) rather than a manual call; pipe /drift into an alerting "
    "channel so a flagged feature reaches a person, not just a log line; investigate whether "
    "dampening the model's reliance on the single prior autumn (e.g. averaging two prior years "
    "once more history exists, or a feature for deviation from trend rather than raw month) "
    "reduces the seasonal-anchoring bias found in Section 5; and revisit the lumpy-country "
    "caveat from Parts 1–2 before leadership relies on country-level forecasts for the sparsest "
    "markets specifically.",
])
SP(8)
story.append(Paragraph(
    "Reproduce via <font face='Courier'>docker build</font> + "
    "<font face='Courier'>scripts/simulate_queries.py</font> (production simulation) and "
    "<font face='Courier'>scripts/analyze_performance.py</font> (this analysis); source metrics "
    "live in <font face='Courier'>reports/content/post_production_*.json</font>.",
    styles["Callout"]
))

doc.build(story)
print("wrote", OUT_PATH)
