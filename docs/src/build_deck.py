"""Build the 8-slide presentation (.pptx + .pdf) and the one-page architecture PDF.

    python docs/src/build_deck.py          # from the repository root; needs python-pptx and LibreOffice (soffice)

Every number on the slides is read from ParkiScanML/reports/*.json, so the deck cannot drift from the reports.
Plain shapes and tables only (no pictures except the dashboard screenshot), so everything is editable in PowerPoint.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
IMG = DOCS / "src" / "img"
sys.path.insert(0, str(DOCS / "src"))
from fill import TEAM  # noqa: E402  (team line lives in one place)

NAME = "ParkiScan Neuroverse"
FULL = "ParkiScan Neuroverse · Game + Digital Twin"
BASE = "ParkiScan_Neuroverse_Game_DigitalTwin"

INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x5F, 0x6B, 0x7A)
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
LINE = RGBColor(0xC9, 0xD0, 0xD8)
FILL = RGBColor(0xF3, 0xF5, 0xF7)
GAME = RGBColor(0xE3, 0xEB, 0xF4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"


# --------------------------------------------------------------------------- numbers from the reports
def numbers():
    ml = ROOT / "ParkiScanML" / "reports"
    fog = json.loads((ml / "fog_forecast_report.json").read_text())
    tw = json.loads((ml / "twin_risk_report.json").read_text())
    h3 = fog["results"]["H3"]["ankle_imu"]
    cue = fog["deployed"]["cue_rule"]
    cons = next(o for o in fog["deployed"]["operating_curve"] if abs(o["window_specificity"] - 0.95) < 1e-6)
    r = tw["results"]
    td = tw["top_decile"]
    return {
        "a_auc": f"{h3['auc']:.2f} ({h3['auc_ci95_subject_bootstrap'][0]:.2f}–{h3['auc_ci95_subject_bootstrap'][1]:.2f})",
        "a_base": f"{fog['baselines']['H3']['freeze_index_auc']:.2f}",
        "a_warn": f"{100 * cue['early_warning_rate']:.0f}% ({cue['warned_before_onset']} of {cue['episodes']})",
        "a_lead": f"{cue['median_lead_s']:.1f} s",
        "a_fa": f"{cue['false_cues_per_hour']:.0f}",
        "a_cons": f"{100 * cons['early_warning_rate']:.0f}% warned at {cons['false_cues_per_hour']:.0f} false cues / h",
        "b_ehr": f"{r['ehr_only']['auc']:.3f}", "b_wear": f"{r['wearable_only']['auc']:.3f}",
        "b_both": f"{r['ehr+wearable']['auc']:.3f}", "b_twin": f"{r['digital_twin (ehr+wearable+pk)']['auc']:.3f}",
        "b_cap": f"{100 * td['capture']:.0f}%", "b_lift": f"{td['lift']:.1f}×",
    }


# --------------------------------------------------------------------------- drawing helpers
def new_prs():
    p = Presentation()
    p.slide_width, p.slide_height = Inches(13.333), Inches(7.5)
    return p


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _run(par, text, size, bold=False, color=INK, italic=False):
    r = par.add_run()
    r.text = text
    f = r.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color
    return r


def text(slide, x, y, w, h, paras, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=0, line=1.0):
    """paras: str | list of (str | list of (text, {bold,color,size}) runs | ('bullet', same))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = line
        if i > 0:
            par.space_before = Pt(spacing)
        bullet = False
        if isinstance(p, tuple) and p and p[0] == "bullet":
            bullet, p = True, p[1]
        runs = [(p, {})] if isinstance(p, str) else p
        if bullet:
            pPr = par._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.22))))
            pPr.set("indent", str(-int(Inches(0.22))))
            for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
                for e in pPr.findall(qn(tag)):
                    pPr.remove(e)
            clr = etree.SubElement(pPr, qn("a:buClr"))
            etree.SubElement(clr, qn("a:srgbClr")).set("val", "1F3A5F")
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
        for t, o in runs:
            _run(par, t, o.get("size", size), o.get("bold", bold), o.get("color", color), o.get("italic", False))
    return tb


def _nostyle(shape):
    for e in shape._element.findall(qn('p:style')):
        shape._element.remove(e)


def rect(slide, x, y, w, h, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    _nostyle(s)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.text_frame.text = ""
    return s


def hline(slide, x1, y1, x2, y2, color=LINE, lw=1.0, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    _nostyle(c)
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    if arrow:
        ln = c.line._get_or_add_ln()
        te = etree.SubElement(ln, qn("a:tailEnd"))
        te.set("type", "triangle")
        te.set("w", "med")
        te.set("len", "med")
    return c


def frame(slide, title, n, total=8):
    text(slide, 0.6, 0.42, 12.1, 0.7, title, size=28, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    hline(slide, 0.6, 1.2, 12.73, 1.2, LINE, 1.0)
    text(slide, 0.6, 7.07, 8, 0.25, FULL, size=10, color=MUTED)
    text(slide, 11.73, 7.07, 1.0, 0.25, f"{n} / {total}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def label(slide, x, y, w, s):
    text(slide, x, y, w, 0.25, s.upper(), size=11, bold=True, color=NAVY)


def table(slide, x, y, w, col_w, rows, size=12, head=True, row_h=0.4, head_color=NAVY, bold_first=False):
    nrow, ncol = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nrow, ncol, Inches(x), Inches(y), Inches(w), Inches(row_h * nrow))
    tbl = gs.table
    tblPr = gs._element.graphic.graphicData.tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    for e in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(e)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left, cell.margin_right = Inches(0.06), Inches(0.1)
            cell.margin_top, cell.margin_bottom = Inches(0.04), Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            par = tf.paragraphs[0]
            is_head = head and i == 0
            runs = [(val, {})] if isinstance(val, str) else val
            for t, o in runs:
                _run(par, t, o.get("size", size - (1 if is_head else 0)),
                     o.get("bold", is_head or (bold_first and j == 0)),
                     o.get("color", head_color if is_head else INK))
            cell.fill.background()
            tcPr = cell._tc.get_or_add_tcPr()
            for tag, show in (("a:lnB", True), ("a:lnT", False), ("a:lnR", False), ("a:lnL", False)):
                ln = etree.Element(qn(tag))
                if show:
                    ln.set("w", str(int(Pt(1.25 if is_head else 0.5))))
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr")).set("val", "1F3A5F" if is_head else "C9D0D8")
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
                tcPr.insert(0, ln)
    return gs


# --------------------------------------------------------------------------- the architecture diagram
def node(slide, x, y, w, h, title, sub, fill=WHITE, tsize=15, ssize=12.5):
    rect(slide, x, y, w, h, fill=fill, line=NAVY if fill == GAME else MUTED, lw=1.0)
    text(slide, x + 0.15, y + 0.12, w - 0.3, 0.3, title, size=tsize, bold=True)
    text(slide, x + 0.15, y + 0.46, w - 0.3, h - 0.5, sub, size=ssize, color=MUTED, line=0.95)


def architecture(slide, x0, y0, scale=1.0):
    """Three columns: data in → twin service → clinician out. Shaded = part of the game."""
    sw, tw_, vw = 3.7, 3.5, 3.13
    gx1, gx2 = 1.0, 0.8
    bh, gap = 1.25, 0.25
    top = y0 + 0.4
    xs, xt, xv = x0, x0 + sw + gx1, x0 + sw + gx1 + tw_ + gx2
    label(slide, xs, y0, sw, "Data in")
    label(slide, xt, y0, tw_, "Digital twin service · Python · local")
    label(slide, xv, y0, vw, "Clinician and player out")

    srcs = [("Synthetic health record", "FHIR R4 · SNOMED CT · LOINC · every resource tagged SYNTHETIC", WHITE),
            ("Wearable streams", "Daphnet ankle, thigh, trunk IMU (real, anonymised) · simulated steps, heart rate, sleep, HRV", WHITE),
            ("ParkiScan Neuroverse game", "3 levels · webcam, voice, drawing, memory, reaction · seated mode · 5 languages", GAME)]
    views = [("Dashboard inside the game", "F2 anywhere · live, or an offline snapshot when the service is off", GAME),
             ("Dashboard in a browser", "/dashboard · static copy for GitHub Pages", WHITE),
             ("Twin report after Level 3", "six charts for the player · session synced to the twin", GAME)]
    mids = []
    for i, (t, s, f) in enumerate(srcs):
        y = top + i * (bh + gap)
        node(slide, xs, y, sw, bh, t, s, f)
        mids.append(y + bh / 2)
        node(slide, xv, y, vw, bh, *views[i][:2], views[i][2])
    ch = 3 * bh + 2 * gap
    rect(slide, xt, top, tw_, ch, fill=FILL, line=MUTED, lw=1.0)
    inner = [("One twin per patient", "record + wearables + game sessions on a 15-minute grid"),
             ("Levodopa PK/PD model", "dopamine effect, ON / OFF state, personal parameters"),
             ("Risk models and what-if", "A: freeze within 3 s · B: within 2 h · alert drivers · re-simulation")]
    ih, ig = 0.97, 0.25
    pad = (ch - 3 * ih - 2 * ig) / 2
    for i, (t, s) in enumerate(inner):
        y = top + pad + i * (ih + ig)
        node(slide, xt + 0.2, y, tw_ - 0.4, ih, t, s, WHITE, tsize=14, ssize=12)
        if i < 2:
            hline(slide, xt + tw_ / 2, y + ih, xt + tw_ / 2, y + ih + ig, MUTED, 1.0, arrow=True)
    for m in mids:
        hline(slide, xs + sw, m, xt, m, NAVY, 1.5, arrow=True)
        hline(slide, xt + tw_, m, xv, m, NAVY, 1.5, arrow=True)
    ly = top + ch + 0.22
    rect(slide, xs, ly + 0.03, 0.22, 0.16, fill=GAME, line=NAVY, lw=0.75)
    text(slide, xs + 0.32, ly, 3.6, 0.25, "Part of the ParkiScan Neuroverse game", size=11, color=MUTED)
    rect(slide, xs + 4.2, ly + 0.03, 0.22, 0.16, fill=WHITE, line=MUTED, lw=0.75)
    text(slide, xs + 4.52, ly, 4.0, 0.25, "Data source or output outside the game", size=11, color=MUTED)
    text(slide, xs + 8.4, ly, 3.73, 0.25, "All data stays on the computer", size=11, color=MUTED, align=PP_ALIGN.RIGHT)


# --------------------------------------------------------------------------- slides
def build():
    N = numbers()
    prs = new_prs()

    # 1 ------------------------------------------------------------ title
    s = blank(prs)
    text(s, 0.9, 1.9, 11.5, 0.9, NAME, size=48, bold=True, color=NAVY)
    text(s, 0.9, 2.9, 11.5, 1.1, "Game + Digital Twin for freezing of gait in Parkinson's disease", size=28, color=INK)
    hline(s, 0.9, 4.15, 3.4, 4.15, NAVY, 2.0)
    text(s, 0.9, 4.4, 11.5, 0.9, ["Happiest Health · Reimagining and Reforming Healthcare in India Summit 2026, Bengaluru",
                                   "Digital Twin Challenge · Phase 1 submission"], size=16, color=MUTED, spacing=4)
    text(s, 0.9, 5.55, 11.5, 0.9, [TEAM["TEAM_LINE"].replace("&lt;", "<").replace("&gt;", ">"),
                                    "[Team name]  ·  [Member names]"], size=14, color=INK, spacing=4)
    text(s, 0.9, 6.75, 11.5, 0.3, "Research prototype · not a medical device · synthetic or anonymised open data only",
         size=11, color=MUTED)

    # 2 ------------------------------------------------------------ problem + objective
    s = blank(prs)
    frame(s, "Problem statement and objective", 2)
    label(s, 0.6, 1.5, 5.8, "Problem")
    text(s, 0.6, 1.85, 5.9, 3.5, [
        ("bullet", "Parkinson's disease: about 5.8 lakh people in India, roughly 10% of the global burden."),
        ("bullet", "Freezing of gait, a sudden inability to step, affects about half of patients in earlier stages and up to 80% as the disease progresses. It is a leading cause of falls."),
        ("bullet", "Care is reactive. A neurologist sees the patient every few months and relies on recall, while freezing, wearing-off and falls happen at home in between."),
    ], size=17, spacing=12)
    label(s, 6.9, 1.5, 5.8, "Objective")
    text(s, 6.9, 1.85, 5.8, 3.5, [
        ("bullet", "Build a digital twin of one patient that fuses a synthetic health record, wearable streams, a levodopa model and a home game."),
        ("bullet", "Predict the adverse event, freezing of gait: seconds ahead as a warning, and hours ahead as a risk window."),
        ("bullet", "Give the clinician a dashboard to see risk and its causes, and to test what-if changes before the next visit."),
    ], size=17, spacing=12)
    rect(s, 0.6, 5.35, 12.13, 1.1, fill=FILL)
    text(s, 0.85, 5.5, 11.6, 0.9, [[("Use case.  ", {"bold": True, "color": NAVY}),
                                    ("A person with Parkinson's plays a short game on a laptop and may wear a low-cost ankle sensor. "
                                     "After each session the twin updates, forecasts freezing risk for the next hours and lets the clinician try changes on the twin first.", {})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.6, 6.62, 12.1, 0.4, "Sources: Khurana & Gourie-Devi, Cureus 2025 (India prevalence); Conde et al., Front Neurol 2023 (freezing and falls).",
         size=10, color=MUTED)

    # 3 ------------------------------------------------------------ requirements
    s = blank(prs)
    frame(s, "Challenge requirements and how we meet them", 3)
    rows = [["Requirement", "What we built", "Where"],
            ["Digital twin of a patient with a localised adverse event", "One virtual Parkinson's patient; the adverse event is freezing of gait", "Twin service"],
            ["Static data: synthetic EHR", "FHIR R4 record with SNOMED CT and LOINC codes, tagged SYNTHETIC; real clinical header for the 10 Daphnet participants", "ParkiScanML/parkiscan/twin"],
            ["Dynamic data: wearables / IoT", "Real Daphnet ankle, thigh and trunk accelerometers; simulated steps, heart rate, sleep, HRV; ESP32 sensor path", "Daphnet, simulator, firmware"],
            ["Fuse both streams and predict the event", f"Model A: freeze within 3 s, AUC {N['a_auc'].split(' ')[0]} on real data. Model B: freeze within 2 h, AUC {N['b_twin']} on a synthetic cohort", "Reports in ParkiScanML/reports"],
            ["Doctor dashboard", "Risk timeline, alert drivers, ON / OFF curve, what-if, sensor replay, record, model card. Opens inside the game (F2) and in a browser", "Game, /dashboard"],
            ["Sandbox data rules", "Only synthetic or anonymised open data; everything runs and stays on the computer", "Whole project"],
            ["Submission package", "GitHub README, Apache-2.0 licence, demo video of 20 minutes or more, architecture PDF, presentation PDF", "Repository, docs/"]]
    table(s, 0.6, 1.5, 12.13, [3.5, 6.33, 2.3], rows, size=13, row_h=0.64, bold_first=True)

    # 4 ------------------------------------------------------------ architecture
    s = blank(prs)
    frame(s, "Architecture", 4)
    architecture(s, 0.6, 1.5)

    # 5 ------------------------------------------------------------ flow
    s = blank(prs)
    frame(s, "End-to-end flow", 5)
    steps = [("Play", "The patient plays three levels of short tasks. Seated mode and a clean skip are always available."),
             ("Measure", "Each task is recorded on the computer: timing, accuracy, smoothness and tracking quality."),
             ("Sync", "At the end of Level 3 the session goes to the local twin service and becomes a patient."),
             ("Fuse", "The twin joins record, wearables, the levodopa curve and game measures on a 15-minute grid."),
             ("Predict", "Model A: freeze within 3 s. Model B: freeze within 2 h, with the drivers behind each alert."),
             ("Review", "The clinician opens the dashboard in the game: risk, alerts, what-if, replay, record, model card.")]
    cw, g = 1.9, 0.146
    hline(s, 0.6 + cw / 2, 1.95, 0.6 + 5 * (cw + g) + cw / 2, 1.95, NAVY, 1.5)
    for i, (t, d) in enumerate(steps):
        x = 0.6 + i * (cw + g)
        c = rect(s, x + cw / 2 - 0.22, 1.73, 0.44, 0.44, fill=NAVY, shape=MSO_SHAPE.OVAL)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        par = tf.paragraphs[0]
        par.alignment = PP_ALIGN.CENTER
        _run(par, str(i + 1), 14, True, WHITE)
        text(s, x, 2.4, cw, 0.35, t, size=18, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        text(s, x + 0.05, 2.9, cw - 0.1, 1.9, d, size=14.5, align=PP_ALIGN.CENTER, line=1.05)
    rect(s, 0.6, 4.95, 5.95, 1.4, fill=FILL)
    text(s, 0.85, 5.08, 5.5, 1.4, [[("If the twin service is off", {"bold": True, "color": NAVY, "size": 14})],
                                    "The in-game dashboard opens a bundled snapshot of 16 twins with 75 what-if scenarios each. The latest session stays listed and can be synced later."],
         size=13, spacing=5)
    rect(s, 6.78, 4.95, 5.95, 1.4, fill=FILL)
    text(s, 7.03, 5.08, 5.5, 1.4, [[("Closing the loop", {"bold": True, "color": NAVY, "size": 14})],
                                    "The clinician tries a change on the twin, such as shifting a dose time, discusses it with the patient, and the next session updates the twin. What-if output is for discussion, not a dosing instruction."],
         size=13, spacing=5)

    # 6 ------------------------------------------------------------ what the game extracts
    s = blank(prs)
    frame(s, "What the game extracts", 6)
    rows = [["Level · task", "Measures extracted", "Domain"],
            ["1 · Follow the memory lantern (webcam gaze)", "pointer error, % on target, re-acquisition latency, flash reaction time, eye-calibration error", "Visual attention"],
            ["1 · Memory Core: sustained “Ahh”", "voiced duration, pitch mean and variability, jitter and shimmer proxies, HNR", "Voice"],
            ["1 · Memory Core: “pa-ta-ka” and sentence", "onset rate and regularity, pauses, speech rate", "Speech motor"],
            ["1 · Memory Core: clap to 90 BPM", "inter-clap interval and variability, tempo error, asynchrony to the beat", "Rhythm timing"],
            ["1 · Recover the lost memory", "hits, omissions, false recognitions, recall time", "Memory"],
            ["2 · Trace the circuit (spiral)", "radial deviation, tremor-band proxy, smoothness, pen lifts", "Fine motor, tremor"],
            ["2 · Connect nodes, repair component", "completion time, wrong connections, placement and rotation error", "Sequencing, visuospatial"],
            ["2 · Reactor signal", "reaction time mean and SD, false and missed responses", "Reaction time"],
            ["3 · Mirror the avatar (webcam pose)", "moves completed, arm elevation, similarity, tracking confidence", "Upper-limb movement"],
            ["3 · Glowing path (steps on the spot)", "steps, cadence, step-interval SD, stop and go latency", "Stepping, stopping"],
            ["3 · Memory vehicle (seated steering)", "lane offset, time in lane, steering steadiness, brake and go latency", "Hand reactions"],
            ["3 · Six check-ins", "answers per domain; kept separate and never scored", "Self-report"]]
    table(s, 0.6, 1.4, 12.13, [3.9, 6.2, 2.03], rows, size=11.5, row_h=0.355)
    text(s, 0.6, 6.2, 12.13, 0.75,
         [[("How it is used.  ", {"bold": True, "color": NAVY}),
           ("Every record carries quality flags (valid, caveats, insufficient, invalid); QA and simulated runs are labelled. On the twin the measures sit beside the levodopa ON / OFF curve. "
            "They are not yet inputs to Models A and B, because no clinician-labelled game sessions exist to train on.", {})]], size=12)

    # 7 ------------------------------------------------------------ results + dashboard
    s = blank(prs)
    frame(s, "Predictions and clinician dashboard", 7)
    label(s, 0.6, 1.45, 5.6, "Model A · real Daphnet data · freeze within 3 s")
    table(s, 0.6, 1.8, 5.6, [2.75, 2.85], [
        ["Freeze-index baseline, AUC", N["a_base"]],
        ["Ankle IMU model, AUC (95% CI)", [(N["a_auc"], {"bold": True})]],
        ["Freezes warned before onset", f"{N['a_warn']}"],
        ["Median warning lead time", N["a_lead"]],
        ["False cues per hour of walking", N["a_fa"]],
        ["Conservative setting", N["a_cons"]]],
        size=12, head=False, row_h=0.38)
    text(s, 0.6, 4.2, 5.6, 0.3, "Leave-one-subject-out on 10 people; the test person is never seen in training.", size=11, color=MUTED)
    label(s, 0.6, 4.62, 5.6, "Model B · synthetic cohort · freeze within 2 h")
    table(s, 0.6, 4.95, 5.6, [3.9, 1.7], [
        ["Health record only, AUC", N["b_ehr"]],
        ["Wearables only, AUC", N["b_wear"]],
        ["Record + wearables, AUC", N["b_both"]],
        ["Digital twin (+ levodopa model), AUC", [(N["b_twin"], {"bold": True})]]],
        size=12, head=False, row_h=0.32)
    text(s, 0.6, 6.33, 5.6, 0.7, f"Top 10% of alerts capture {N['b_cap']} of freeze windows (lift {N['b_lift']}). Labels come from a documented simulator: this shows the fusion works end to end, not clinical accuracy.",
         size=11, color=MUTED)
    from PIL import Image
    crop = IMG / "dash_overview_crop.png"
    Image.open(IMG / "dash_overview.png").convert("RGB").crop((450, 110, 2400, 1130)).save(crop, optimize=True)
    pic_w, px = 6.1, 6.63
    ph = pic_w * 1020 / 1950
    s.shapes.add_picture(str(crop), Inches(px), Inches(1.5), width=Inches(pic_w))
    rect(s, px, 1.5, pic_w, ph, fill=None, line=LINE, lw=0.75)
    text(s, px, 1.5 + ph + 0.12, pic_w, 1.2,
         "Clinician dashboard: freezing risk now and for the next 24 h, alerts with their causes, levodopa ON / OFF curve and what-if simulator. "
         "The same views open inside the game with F2.", size=12, color=MUTED)

    # 8 ------------------------------------------------------------ novelty, limits, next
    s = blank(prs)
    frame(s, "Novelty, limits and next steps", 8)
    label(s, 0.6, 1.5, 5.6, "What is new")
    text(s, 0.6, 1.85, 5.6, 5.0, [
        [("A game that is also the twin's data source. ", {"bold": True}), ("The patient plays; the clinician view opens inside the same program.", {})],
        [("Four sources in one twin. ", {"bold": True}), ("Record, wearables, game and a personal levodopa PK/PD model.", {})],
        [("Seconds-ahead forecast, honestly scored. ", {"bold": True}), ("Leave-one-subject-out, with warned %, lead time and false cues per hour instead of one AUC.", {})],
        [("Alerts explain themselves. ", {"bold": True}), ("Each driver is removed in simulation to show how much risk it adds; what-if re-runs the twin with identical noise.", {})],
        [("Built for Parkinson's. ", {"bold": True}), ("Webcam only, no hardware needed, seated mode, clean skip, no pass or fail, five languages, runs offline.", {})],
    ], size=15, spacing=10)
    label(s, 6.6, 1.5, 3.2, "What we do not claim")
    text(s, 6.6, 1.85, 3.2, 5.0, [
        ("bullet", "No diagnosis, staging or dosing advice."),
        ("bullet", "Model A: 10 people in a lab protocol that provokes freezing; home false-cue rates will differ."),
        ("bullet", "Model B and consumer-wearable streams are synthetic."),
        ("bullet", "Webcam measures are 2-D estimates, not clinical tracking."),
    ], size=14.5, spacing=11)
    label(s, 10.2, 1.5, 2.5, "Next steps")
    text(s, 10.2, 1.85, 2.53, 5.0, [
        ("bullet", "Clinician-labelled game sessions, with ethics approval."),
        ("bullet", "Feed game measures into the models."),
        ("bullet", "Wearable cue pilot with the ESP32 sensor."),
        ("bullet", "Native-speaker review of translations."),
    ], size=14.5, spacing=11)
    rect(s, 0.6, 6.2, 12.13, 0.7, fill=FILL)
    text(s, 0.6, 6.2, 12.13, 0.7, "Freezing of gait: from surprise to forecast, from one home game and one digital twin.",
         size=18, bold=True, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    out = DOCS / f"{BASE}_Presentation.pptx"
    prs.save(out)

    # architecture one-pager
    a = new_prs()
    s = blank(a)
    text(s, 0.6, 0.42, 12.1, 0.7, f"{NAME} · Game + Digital Twin · system architecture", size=26, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    hline(s, 0.6, 1.2, 12.73, 1.2, LINE, 1.0)
    architecture(s, 0.6, 1.5)
    text(s, 0.6, 6.78, 12.13, 0.45, "Models: A predicts a freeze within 3 s from the ankle accelerometer (real data, leave-one-subject-out). "
         "B predicts a freeze within 2 h from record, wearables and the levodopa model (synthetic cohort). Research prototype, not a medical device.",
         size=10.5, color=MUTED)
    arch = DOCS / "src" / "architecture.pptx"
    a.save(arch)
    return out, arch


def to_pdf(pptx: Path, out_pdf: Path):
    out_dir = pptx.parent
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(pptx)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    produced = out_dir / (pptx.stem + ".pdf")
    if produced != out_pdf:
        produced.replace(out_pdf)


if __name__ == "__main__":
    deck, arch = build()
    to_pdf(deck, DOCS / f"{BASE}_Presentation.pdf")
    to_pdf(arch, DOCS / f"{BASE}_Architecture.pdf")
    print("wrote", deck.name, "and both PDFs")
