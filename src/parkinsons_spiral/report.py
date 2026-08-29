"""Clinician-readable research report for a completed local session."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
import json
from pathlib import Path
from statistics import median
from typing import Any

import joblib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .features import COLUMNS, extract_participant_features


INK = colors.HexColor("#123C35")
GREEN = colors.HexColor("#176459")
LIME = colors.HexColor("#B9D978")
PAPER = colors.HexColor("#FBFAF5")
PANEL = colors.HexColor("#F3F1E9")
LINE = colors.HexColor("#D8DDD6")
MUTED = colors.HexColor("#60716D")
ORANGE = colors.HexColor("#C77655")


def _register_fonts() -> tuple[str, str]:
    regular = "Helvetica"
    bold = "Helvetica-Bold"
    font_root = Path("C:/Windows/Fonts")
    regular_path = font_root / "arial.ttf"
    bold_path = font_root / "arialbd.ttf"
    if regular_path.exists() and bold_path.exists():
        if "ReportSans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("ReportSans", str(regular_path)))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold_path)))
        regular, bold = "ReportSans", "ReportSans-Bold"
    return regular, bold


class ScoreCard(Flowable):
    def __init__(self, label: str, score: float | None, threshold: float, status: str):
        super().__init__()
        self.label = label
        self.score = score
        self.threshold = threshold
        self.status = status
        self.width = 82 * mm
        self.height = 39 * mm

    def draw(self) -> None:
        canvas = self.canv
        canvas.setStrokeColor(LINE)
        canvas.setFillColor(PANEL)
        canvas.roundRect(0, 0, self.width, self.height, 2 * mm, fill=1, stroke=1)
        canvas.setFillColor(GREEN)
        canvas.setFont("ReportSans-Bold", 8)
        canvas.drawString(7 * mm, 30 * mm, self.label.upper())
        shown = "-" if self.score is None else str(round(self.score * 100))
        canvas.setFillColor(INK)
        canvas.setFont("ReportSans-Bold", 31)
        canvas.drawString(7 * mm, 15 * mm, shown)
        canvas.setFont("ReportSans", 10)
        canvas.setFillColor(MUTED)
        canvas.drawString(28 * mm, 18 * mm, "/ 100 model score")
        bar_x, bar_y, bar_w, bar_h = 7 * mm, 8 * mm, 68 * mm, 2.8 * mm
        canvas.setFillColor(colors.HexColor("#E1E6E0"))
        canvas.rect(bar_x, bar_y, bar_w, bar_h, fill=1, stroke=0)
        if self.score is not None:
            canvas.setFillColor(GREEN)
            canvas.rect(bar_x, bar_y, bar_w * min(max(self.score, 0), 1), bar_h, fill=1, stroke=0)
        canvas.setStrokeColor(ORANGE)
        x = bar_x + bar_w * self.threshold
        canvas.line(x, bar_y - 1.2 * mm, x, bar_y + bar_h + 1.2 * mm)
        canvas.setFillColor(GREEN)
        canvas.setFont("ReportSans-Bold", 7.5)
        canvas.drawString(7 * mm, 3.2 * mm, self.status.upper())


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold = _register_fonts()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=bold, fontSize=25, leading=29, textColor=INK, alignment=TA_LEFT, spaceAfter=3 * mm),
        "kicker": ParagraphStyle("kicker", parent=base["Normal"], fontName=bold, fontSize=8, leading=10, textColor=GREEN, tracking=1.5, spaceAfter=2 * mm),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=bold, fontSize=15, leading=19, textColor=INK, spaceBefore=2 * mm, spaceAfter=3 * mm),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=bold, fontSize=10, leading=13, textColor=GREEN, spaceBefore=2 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=regular, fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=2 * mm),
        "body_bold": ParagraphStyle("body_bold", parent=base["BodyText"], fontName=bold, fontSize=9, leading=12, textColor=INK, spaceAfter=2 * mm),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName=regular, fontSize=7.2, leading=9.5, textColor=MUTED),
        "small_bold": ParagraphStyle("small_bold", parent=base["BodyText"], fontName=bold, fontSize=7.2, leading=9.5, textColor=INK),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName=regular, fontSize=8.5, leading=12.5, textColor=INK, leftIndent=4 * mm, rightIndent=4 * mm, borderColor=LIME, borderWidth=0, borderPadding=3 * mm, backColor=PANEL),
        "center": ParagraphStyle("center", parent=base["BodyText"], fontName=regular, fontSize=7.2, leading=9, textColor=MUTED, alignment=TA_CENTER),
    }


def _p(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(value)), style)


def _table(data: list[list[Any]], widths: list[float], styles: dict[str, ParagraphStyle], repeat: bool = True) -> Table:
    formatted: list[list[Any]] = []
    for row_index, row in enumerate(data):
        if row_index == 0:
            style = ParagraphStyle(
                "table_header",
                parent=styles["small_bold"],
                textColor=colors.white,
            )
        else:
            style = styles["small"]
        formatted.append([cell if isinstance(cell, Flowable) else _p(cell, style) for cell in row])
    table = Table(formatted, colWidths=widths, repeatRows=1 if repeat else 0, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
    ]))
    return table


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value).replace("_", " ")


def _friendly(name: str) -> str:
    replacements = {
        "Jitter_rel": "Relative jitter",
        "Jitter_RAP": "Jitter RAP",
        "Jitter_PPQ": "Jitter PPQ",
        "Shim_loc": "Local shimmer",
        "Shim_dB": "Shimmer dB",
        "Shim_APQ3": "Shimmer APQ3",
        "Shim_APQ5": "Shimmer APQ5",
        "Shi_APQ11": "Shimmer APQ11",
        "pitch_median_hz": "Median pitch",
        "pitch_cv": "Pitch variability",
        "hnr_proxy_db": "Harmonicity proxy",
        "energy_cv": "Energy variability",
        "zero_crossing_rate": "Zero-crossing rate",
        "spectral_centroid_hz": "Spectral centroid",
        "spectral_bandwidth_hz": "Spectral bandwidth",
        "spectral_flatness": "Spectral flatness",
    }
    if name in replacements:
        return replacements[name]
    return name.replace("__", " - ").replace("_", " ").title()


def _contributions(bundle_path: Path, rows: list[dict[str, float]]) -> list[tuple[str, float]]:
    if not bundle_path.exists() or not rows:
        return []
    bundle = joblib.load(bundle_path)
    names = list(bundle["feature_names"])
    row = pd.DataFrame(rows).reindex(columns=names).median(axis=0).to_frame().T
    pipeline = bundle["pipeline"]
    if not hasattr(pipeline, "named_steps"):
        return []
    values = pipeline.named_steps["imputer"].transform(row)
    values = pipeline.named_steps["scaler"].transform(values)
    selector = pipeline.named_steps.get("selector")
    if selector is not None:
        keep = selector.get_support()
        names = [name for name, selected in zip(names, keep, strict=True) if selected]
        values = selector.transform(values)
    classifier = pipeline.named_steps["classifier"]
    contributions = values[0] * classifier.coef_[0]
    ranked = sorted(zip(names, contributions, strict=True), key=lambda item: abs(item[1]), reverse=True)
    return [(name, float(value)) for name, value in ranked[:6]]


def _load_metrics(model_path: Path) -> dict[str, Any]:
    path = model_path.with_name("metrics.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("metrics", {})


def _motor_feature_rows(session: dict[str, Any]) -> list[dict[str, float]]:
    valid = [trial for trial in session["trials"] if trial["quality"]["valid"]]
    indexed = {(trial["hand"], trial["repetition"], trial["mode"]): trial for trial in valid}
    rows: list[dict[str, float]] = []
    for hand in ("left", "right"):
        for repetition in range(1, 4):
            static = indexed.get((hand, repetition, "static"))
            dynamic = indexed.get((hand, repetition, "dynamic"))
            if not static or not dynamic:
                continue
            frames = []
            for trial, test_id in ((static, 0), (dynamic, 1)):
                points = []
                for point in trial["points"]:
                    tilt = min(float(np.hypot(point["tilt_x"], point["tilt_y"])), 90.0)
                    points.append((point["x"], point["y"], 0, point["pressure"] * 1024.0, (90.0 - tilt) * 10.0, point["t"], test_id))
                frames.append(pd.DataFrame(points, columns=COLUMNS))
            rows.append(extract_participant_features(pd.concat(frames, ignore_index=True)))
    return rows


def _header_footer(canvas: Any, doc: BaseDocTemplate) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(GREEN)
    canvas.rect(18 * mm, height - 14 * mm, width - 36 * mm, 1.2 * mm, fill=1, stroke=0)
    canvas.setFont("ReportSans", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Parkinson's multimodal research screening - not a diagnosis")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_session_report(
    session: dict[str, Any],
    motor_result: dict[str, Any],
    voice_result: dict[str, Any],
    motor_model_path: Path,
    voice_model_path: Path,
) -> bytes:
    """Return a multi-page PDF using only locally stored derived measurements."""
    styles = _styles()
    stream = BytesIO()
    doc = BaseDocTemplate(stream, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=17 * mm, title="Parkinson's Multimodal Research Report", author="Local Parkinson's Research Tool")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_header_footer)])
    story: list[Flowable] = []

    created = datetime.fromisoformat(session["created_at"]).astimezone().strftime("%d %b %Y, %H:%M %Z")
    threshold = float(motor_result["decision_threshold"])
    motor_score = float(motor_result["experimental_screening_score"])
    voice_score = voice_result.get("experimental_voice_score")
    agreement = voice_score is not None and ((motor_score >= threshold) == (float(voice_score) >= threshold))
    if voice_score is None:
        headline = "Spiral result available; voice was not safely scorable"
    elif agreement and motor_score < threshold:
        headline = "Both research signals are below the elevated-signal boundary"
    elif agreement:
        headline = "Both research signals meet the elevated-signal boundary"
    else:
        headline = "The two research signals are mixed"

    story += [
        _p("MULTIMODAL RESEARCH REPORT", styles["kicker"]),
        _p("Parkinson's pattern screening", styles["title"]),
        _p(headline, styles["body_bold"]),
        Spacer(1, 2 * mm),
        _table([
            ["Participant code", "Session", "Captured", "Protocol"],
            [session["participant_code"], session["id"][:12], created, f"{session['handedness'].title()} dominant hand + sustained voice"],
        ], [37 * mm, 36 * mm, 49 * mm, 52 * mm], styles, repeat=False),
        Spacer(1, 5 * mm),
        Table([[ScoreCard("Dominant-hand spiral", motor_score, threshold, f"{motor_result['pattern_signal']} pattern signal"), ScoreCard("Sustained voice", None if voice_score is None else float(voice_score), threshold, "unable to score safely" if voice_score is None else f"{voice_result['pattern_signal']} pattern signal")]], colWidths=[87 * mm, 87 * mm], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 5)])),
        Spacer(1, 5 * mm),
        _p("How to interpret this result", styles["h1"]),
        _p(
            f"The spiral score is {motor_score * 100:.1f}/100 and the voice score is " + ("unavailable" if voice_score is None else f"{float(voice_score) * 100:.1f}/100") + f". The pre-set elevated-signal boundary is {threshold * 100:.0f}/100. A score below that boundary is labelled a lower pattern signal; it does not rule out Parkinson's disease or another cause of symptoms.",
            styles["body"],
        ),
        _p("These are model similarity scores, not the percentage chance that this participant has Parkinson's disease and not diagnostic confidence. The two modalities were trained on separate, small research cohorts and must not be averaged into a diagnosis.", styles["callout"]),
        Spacer(1, 4 * mm),
        _p("Capture quality at a glance", styles["h1"]),
    ]
    quality_rows = [["Capture", "Duration", "Sampling", "Coverage / voicing", "Status"]]
    for trial in sorted(session["trials"], key=lambda x: x["mode"], reverse=True):
        q = trial["quality"]
        quality_rows.append([f"Spiral - {trial['mode']}", f"{q['duration_s']:.1f} s", f"{q['sample_rate_hz']:.1f} Hz", f"{q['turns']:.2f} turns", "Pass" if q["valid"] else "Review"])
    for trial in session["voice_trials"]:
        q = trial["quality"]
        quality_rows.append([f"Voice - take {trial['repetition']}", f"{q['duration_s']:.2f} s", f"{q['sample_rate_hz']} Hz", f"{q['voiced_ratio'] * 100:.1f}% voiced", "Pass" if q["valid"] and (trial.get("result") or {}).get("domain_match") else "Review"])
    story.append(_table(quality_rows, [38 * mm, 29 * mm, 31 * mm, 42 * mm, 34 * mm], styles))

    story += [PageBreak(), _p("VOICE ANALYSIS", styles["kicker"]), _p("Sustained-vowel acoustic measurements", styles["title"])]
    if voice_score is not None:
        scores = voice_result.get("recording_scores", [])
        story.append(_p(f"Three valid sustained-'ah' recordings were scored independently. Their scores were {', '.join(f'{float(v) * 100:.1f}' for v in scores)}/100; the reported voice result is the median ({float(voice_score) * 100:.1f}/100). All selected model inputs were within the accepted training-domain ranges.", styles["body"]))
    else:
        story.append(_p("The audio could be measured, but at least one selected model input fell outside the accepted training-domain ranges. The application therefore suppressed the voice score.", styles["body"]))
    voice_rows = [["Take", "Pitch median", "Pitch variability", "Jitter", "Shimmer", "Harmonicity", "Score"]]
    for trial in session["voice_trials"]:
        f = trial["features"]
        score = (trial.get("result") or {}).get("screening_score")
        voice_rows.append([
            str(trial["repetition"]), f"{f['pitch_median_hz']:.1f} Hz", f"{f['pitch_cv'] * 100:.2f}%", f"{f['Jitter_rel']:.2f}%", f"{f['Shim_loc'] * 100:.2f}%", f"{f['hnr_proxy_db']:.2f} dB", "Suppressed" if score is None else f"{float(score) * 100:.1f}/100",
        ])
    story += [_table(voice_rows, [15 * mm, 29 * mm, 29 * mm, 24 * mm, 25 * mm, 27 * mm, 25 * mm], styles), Spacer(1, 4 * mm)]
    secondary_rows = [["Take", "Energy variability", "Zero crossing", "Spectral centroid", "Bandwidth", "Flatness", "Clipping"]]
    for trial in session["voice_trials"]:
        f, q = trial["features"], trial["quality"]
        secondary_rows.append([str(trial["repetition"]), f"{f['energy_cv'] * 100:.2f}%", f"{f['zero_crossing_rate']:.4f}", f"{f['spectral_centroid_hz']:.1f} Hz", f"{f['spectral_bandwidth_hz']:.1f} Hz", f"{f['spectral_flatness']:.4f}", f"{q['clipping_fraction'] * 100:.3f}%"])
    story += [_p("Additional spectral and recording metrics", styles["h2"]), _table(secondary_rows, [15 * mm, 31 * mm, 25 * mm, 30 * mm, 27 * mm, 24 * mm, 22 * mm], styles), Spacer(1, 4 * mm)]
    voice_contrib = _contributions(voice_model_path, [trial["features"] for trial in session["voice_trials"] if trial["quality"]["valid"]])
    contribution_rows = [["Model input", "Direction", "Relative log-odds contribution", "Meaning"]]
    for name, value in voice_contrib:
        direction = "Toward PD-labelled pattern" if value > 0 else "Toward control-labelled pattern"
        contribution_rows.append([_friendly(name), direction, f"{value:+.3f}", "Association inside this fitted model; not a clinical cause"])
    story += [_p("Strongest model contributions", styles["h1"]), _p("Contribution values show how the median measured feature profile moved this logistic model's score. Larger absolute values had more influence. They are not clinical severity measurements.", styles["body"])]
    if voice_contrib:
        story.append(_table(contribution_rows, [39 * mm, 44 * mm, 39 * mm, 52 * mm], styles))
    else:
        story.append(_p("Contribution detail is unavailable for this model artifact.", styles["body"]))
    story += [Spacer(1, 4 * mm), _p("What the terms mean", styles["h2"]), _p("Pitch is the estimated fundamental frequency. Jitter measures cycle-to-cycle pitch-period variation. Shimmer measures cycle-to-cycle amplitude variation. The harmonicity proxy compares periodic structure with residual noise. Spectral measures summarize how energy is distributed across frequency. MFCC values are also used by the model but are compact signal descriptors, not directly interpretable clinical findings.", styles["body"]), _p("Privacy note: raw microphone audio is deliberately discarded after feature extraction. This report can show aggregate pitch and stability measurements, but it cannot recreate a waveform, spectrogram, or time-by-time pitch track.", styles["callout"])]

    story += [PageBreak(), _p("MOTOR ANALYSIS", styles["kicker"]), _p("Dominant-hand spiral measurements", styles["title"])]
    motor_rows = [["Trial", "Duration", "Points", "Rate", "Turns", "Extent", "Pressure", "Status"]]
    for trial in sorted(session["trials"], key=lambda x: x["mode"], reverse=True):
        q = trial["quality"]
        motor_rows.append([trial["mode"].title(), f"{q['duration_s']:.2f} s", _fmt(q["point_count"], 0), f"{q['sample_rate_hz']:.1f} Hz", f"{q['turns']:.3f}", f"{q['extent_px']:.1f} px", f"{q['pressure_range'][0]:.2f}-{q['pressure_range'][1]:.2f}", "Pass" if q["valid"] else "Review"])
    story += [_table(motor_rows, [23 * mm, 25 * mm, 20 * mm, 24 * mm, 21 * mm, 24 * mm, 27 * mm, 20 * mm], styles), Spacer(1, 4 * mm)]
    motor_features = _motor_feature_rows(session)
    if motor_features:
        profile = motor_features[0]
        feature_rows = [["Movement metric", "Static", "Dynamic", "Dynamic - static"]]
        for base in ["path_length_norm", "speed_mean_norm", "speed_cv", "accel_rms_norm", "stationary_fraction", "pressure_cv", "archimedean_rmse_norm", "radial_monotonicity", "angle_backtrack_fraction", "tremor_power_4_7_ratio"]:
            static = profile.get(f"static__{base}")
            dynamic = profile.get(f"dynamic__{base}")
            feature_rows.append([_friendly(base), _fmt(static), _fmt(dynamic), _fmt(None if static is None or dynamic is None else dynamic - static)])
        story += [_p("Derived trajectory metrics", styles["h1"]), _table(feature_rows, [66 * mm, 36 * mm, 36 * mm, 36 * mm], styles), Spacer(1, 4 * mm)]
        motor_contrib = _contributions(motor_model_path, motor_features)
        motor_contrib_rows = [["Model input", "Direction", "Relative log-odds contribution"]]
        for name, value in motor_contrib:
            motor_contrib_rows.append([_friendly(name), "Toward PD-labelled pattern" if value > 0 else "Toward control-labelled pattern", f"{value:+.3f}"])
        story += [_p("Strongest motor-model contributions", styles["h1"])]
        if motor_contrib:
            story.append(_table(motor_contrib_rows, [65 * mm, 69 * mm, 40 * mm], styles))
        else:
            story.append(_p("Contribution detail is unavailable for this model artifact.", styles["body"]))

    motor_metrics = _load_metrics(motor_model_path)
    voice_metrics = _load_metrics(voice_model_path)
    performance_rows = [["Model", "Participants", "ROC AUC", "Sensitivity at 75", "Specificity at 75", "Evaluation"]]
    for name, metrics in (("Spiral", motor_metrics), ("Voice", voice_metrics)):
        performance_rows.append([
            name,
            _fmt(metrics.get("participants"), 0),
            _fmt(metrics.get("roc_auc")),
            _fmt(metrics.get("sensitivity_at_threshold")),
            _fmt(metrics.get("specificity_at_threshold")),
            metrics.get("evaluation", "Not available"),
        ])
    story += [
        PageBreak(),
        _p("MODEL CARD", styles["kicker"]),
        _p("Methods, provenance, and limitations", styles["title"]),
        _p("How the scores were produced", styles["h1"]),
        _p("Motor model: regularized logistic regression using paired static tracing and dynamic freehand spiral trajectory features. Voice model: robust-scaled, feature-selected logistic regression using sustained-vowel measurements standardized to an 8 kHz telephone-audio passband. The session voice result is the median of three independently scored recordings.", styles["body"]),
        _p("The 75/100 boundary is a conservative research decision rule chosen before this session. It is not 75% diagnostic confidence. Raising a threshold usually reduces false positive signals but increases missed cases.", styles["callout"]),
        Spacer(1, 4 * mm),
        _p("Cross-validation summary", styles["h1"]),
        _table(performance_rows, [22 * mm, 25 * mm, 24 * mm, 28 * mm, 28 * mm, 47 * mm], styles),
        _p("Sensitivity is the fraction of Parkinson's-labelled participants flagged at this boundary; specificity is the fraction of controls not flagged. These estimates come from small research datasets and may not transfer to this participant, tablet, or microphone.", styles["small"]),
        Spacer(1, 4 * mm),
        _p("Training data", styles["h1"]),
        _table([
            ["Modality", "Source", "Cohort", "Identifier"],
            ["Spiral", "UCI HandPD/NewHandPD trajectory dataset", "77 participants: 62 Parkinson's-labelled, 15 controls", "DOI 10.24432/C5Q01S"],
            ["Voice", "Figshare labelled sustained-/a/ raw telephone audio", "81 participants: 40 Parkinson's-labelled, 41 controls", "DOI 10.6084/m9.figshare.23849127.v1"],
        ], [24 * mm, 55 * mm, 61 * mm, 34 * mm], styles),
        Spacer(1, 4 * mm),
        _p("Important limitations", styles["h1"]),
        _p("The motor and voice models use different research cohorts. Dataset size, selection bias, device differences, age, medication, microphone, language, fatigue, other neurological or voice conditions, and recording technique can affect results. Cross-validation measures internal discrimination, not real-world diagnostic safety or calibration.", styles["body"]),
        _p("This report is for research and software evaluation only. It cannot diagnose or exclude Parkinson's disease. If there are symptoms such as tremor, slowness, stiffness, balance problems, or persistent voice change, seek assessment from a qualified clinician regardless of these scores.", styles["callout"]),
    ]

    doc.build(story)
    return stream.getvalue()
