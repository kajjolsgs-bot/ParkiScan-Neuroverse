"""Fill README.md from the model reports, so every number in it traces back to a file (the deck: build_deck.py).

    python docs/src/fill.py            # from the repository root
writes README.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ParkiScanML" / "reports"
TEAM = {  # edit here, then re-run
    "TEAM_LINE": "Team leader: Kajjol · Integrated M.Tech CSE with Business Analytics · VIT Chennai",
    "REPO": "github.com/&lt;your-account&gt;/ParkiScan-Neuroverse",
    "VIDEO": "unlisted YouTube link in README",
}


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def main():
    fog = json.loads((ML / "fog_forecast_report.json").read_text())
    tw = json.loads((ML / "twin_risk_report.json").read_text())
    dep = fog["deployed"]
    ev = dep["event_level"]
    cue = dep.get("cue_rule", ev)
    curve = dep.get("operating_curve", [])
    names = ["ankle_imu", "ankle_imu+ehr", "all_imu", "all_imu+ehr"]

    def auc_cell(v):
        return f"{v['auc']:.2f} ({v['auc_ci95_subject_bootstrap'][0]:.2f}–{v['auc_ci95_subject_bootstrap'][1]:.2f})" if v else "–"

    rows = ["| freeze onset within | freeze-index baseline | ankle IMU | ankle IMU + clinical header | 3 IMUs | 3 IMUs + header |",
            "|---|---|---|---|---|---|"]
    for h, res in fog["results"].items():
        rows.append(f"| {h[1:]} s | {fog['baselines'][h]['freeze_index_auc']:.2f} | " + " | ".join(auc_cell(res.get(n)) for n in names) + " |")
    fog_table = ("ROC AUC, pooled out-of-fold over 10 held-out people, 95% CI by subject bootstrap "
                 f"({fog['n_windows']:,} windows).\n\n" + "\n".join(rows))
    fog_events = (f"**{cue['warned_before_onset']} of {cue['episodes']} freezes ({pct(cue['early_warning_rate'])}) were preceded by an alarm "
                  f"in the 5 s before onset**, median lead time **{cue['median_lead_s']:.1f} s**. Window level: sensitivity "
                  f"{dep['window_level']['sensitivity']:.2f}, specificity {dep['window_level']['specificity']:.2f}. "
                  f"As a wearable cue (10-s refractory period) this setting produces {cue['false_cues_per_hour']:.0f} false cues per hour of "
                  f"walking, in a lab protocol that provoked ≈ {cue['episodes'] / cue['non_fog_hours']:.0f} freezes per hour. "
                  "The clinician can trade sensitivity for fewer cues:\n\n"
                  "| setting (window specificity) | freezes warned | false cues / h | median lead |\n|---|---|---|---|\n"
                  + "\n".join(f"| {o['window_specificity']:.2f} | {pct(o['early_warning_rate'])} ({o['warned_before_onset']}/{o['episodes']}) | "
                              f"{o['false_cues_per_hour']:.0f} | {o['median_lead_s']:.1f} s |" for o in curve)
                  + "\n\nFull per-participant table: `ParkiScanML/reports/fog_forecast_report.md`.")
    tr = tw["results"]
    twin_table = (f"{tw['data'][0].upper() + tw['data'][1:]}. Target: {tw['target']}. Validation: {tw['validation']}. "
                  f"Prevalence {pct(tw['prevalence'], 1)}.\n\n| inputs | ROC AUC | PR AUC | Brier |\n|---|---|---|---|\n"
                  + "\n".join(f"| {k} | {v['auc']:.3f} | {v['pr_auc']:.3f} | {v['brier']:.4f} |" for k, v in tr.items()))
    td = tw["top_decile"]
    twin_top = (f"Alerting on the top 10% of risk captures **{pct(td['capture'])}** of the 2-hour windows that contain a freeze "
                f"(precision {pct(td['precision'])}, lift **{td['lift']:.1f}×**). Calibration by risk decile is in `twin_risk_report.json`.")

    tpl = (ROOT / "README.template.md").read_text()
    out = (tpl.replace("{{FOG_TABLE}}", fog_table).replace("{{FOG_EVENTS}}", fog_events)
              .replace("{{TWIN_TABLE}}", twin_table).replace("{{TWIN_TOPDEC}}", twin_top))
    assert "{{" not in out, re.findall(r"\{\{\w+\}\}", out)
    (ROOT / "README.md").write_text(out)

    print("README.md written; the deck is built by docs/src/build_deck.py")


if __name__ == "__main__":
    main()
