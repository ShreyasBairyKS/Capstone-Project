"""Single source of truth for every number in the EPM report.

Baseline durations and all hours are retrospective estimates (commit history
plus team recall). Actual finish dates come from git where a commit exists;
actual starts before the first relevant commit are team estimates.
"""
from datetime import date

PROJECT_START = date(2026, 2, 2)
PLANNED_FINISH = date(2026, 9, 30)
STATUS_DATE = date(2026, 10, 26)
INTERIM_DATE = date(2026, 7, 31)

SU, UL, SH = "Sushanth", "Ullas", "Shreyas"
MEMBERS = [SU, UL, SH]
RATE = {SU: 250, UL: 200, SH: 200}  # Sushanth is PM, at the team-lead rate
MENTOR_RATE = 1000
MENTOR_BASELINE_HOURS = 32
MENTOR_ACTUAL_HOURS = 35
CONTINGENCY_PCT = 10
ACTUAL_HOURS = {SU: 250, UL: 240, SH: 220}  # team-confirmed totals

# Calendar months used as reporting periods.
PERIODS = [(2026, m) for m in range(2, 11)]

# id, wbs, name, owner, predecessors, baseline duration (working days),
# baseline hours, actual hours, actual start, actual finish, estimator, evidence
ACTIVITIES = [
    dict(id="A", wbs="1.1", name="Literature survey", owner=SU, pred=[], dur=10,
         base={SU: 16, UL: 16, SH: 17}, act={SU: 16, UL: 16, SH: 17},
         start=date(2026, 2, 2), finish=date(2026, 2, 16),
         evidence="Team record; references in the thesis and IEEE paper"),
    dict(id="B", wbs="1.2", name="Requirements and architecture", owner=SU, pred=["A"], dur=15,
         base={SU: 20, UL: 8, SH: 9}, act={SU: 18, UL: 8, SH: 9},
         start=date(2026, 2, 17), finish=date(2026, 3, 6),
         evidence="Team estimate; precedes the first commit"),
    dict(id="C", wbs="2.1", name="Platform scaffold and core", owner=SH, pred=["B"], dur=10,
         base={SH: 21, SU: 8}, act={SU: 8, SH: 21},
         start=date(2026, 3, 9), finish=date(2026, 3, 21),
         evidence="defc059, 1f79bba (21 Mar)"),
    dict(id="D", wbs="2.2", name="Backend API and database", owner=SU, pred=["C"], dur=20,
         base={SU: 33, SH: 4}, act={SU: 29, SH: 8},
         start=date(2026, 3, 23), finish=date(2026, 4, 17),
         evidence="7761ad1 (17 Apr)"),
    dict(id="E", wbs="2.3", name="Dashboard v1 and redesign", owner=SH, pred=["C"], dur=30,
         base={SH: 38, SU: 4, UL: 4}, act={SU: 4, UL: 4, SH: 46},
         start=date(2026, 3, 23), finish=date(2026, 5, 4),
         evidence="8eda575 (4 May)"),
    dict(id="F", wbs="3.1", name="Cap dataset, detector and classifier iterations", owner=UL, pred=["B"], dur=25,
         base={UL: 44, SH: 8}, act={UL: 52, SH: 8},
         start=date(2026, 3, 23), finish=date(2026, 5, 3),
         evidence="9c7c355 (21 Apr) to 452ad11 (3 May)"),
    dict(id="G", wbs="3.2", name="Fill dataset and fill-level models", owner=UL, pred=["F"], dur=10,
         base={UL: 20, SH: 8}, act={UL: 22, SH: 11},
         start=date(2026, 4, 20), finish=date(2026, 5, 15),
         evidence="c4639ff (3 May) to 47263b8 (15 May)"),
    dict(id="H", wbs="3.3", name="First integrated demo", owner=UL, pred=["D", "E", "G"], dur=5,
         base={UL: 8, SU: 4, SH: 4}, act={SU: 4, UL: 8, SH: 4},
         start=date(2026, 5, 4), finish=date(2026, 5, 15),
         evidence="8c54630 (15 May)"),
    dict(id="I", wbs="6.1", name="Thesis, IEEE paper, presentation", owner=SU, pred=["H"], dur=10,
         base={SU: 20, UL: 20, SH: 8}, act={SU: 18, UL: 18, SH: 8},
         start=date(2026, 5, 1), finish=date(2026, 5, 22),
         evidence="76a9451 (1 May) to a07dfde (22 May)"),
    dict(id="J", wbs="4.1", name="Label: PatchCore baseline", owner=UL, pred=["I"], dur=25,
         base={UL: 28, SH: 4}, act={UL: 36, SH: 4},
         start=date(2026, 6, 15), finish=date(2026, 7, 11),
         evidence="473f027 (20 Jun) to 188e0e7 (11 Jul)"),
    dict(id="K", wbs="4.2", name="Label: annotation, YOLO iteration, template matching", owner=SH, pred=["J"], dur=15,
         base={SH: 25, UL: 20}, act={UL: 24, SH: 34},
         start=date(2026, 7, 13), finish=date(2026, 8, 4),
         evidence="70f0b5e (21 Jul) to adc3f16 (4 Aug)"),
    dict(id="L", wbs="4.3", name="Label: EfficientAD, grayscale, calibration", owner=UL, pred=["K"], dur=15,
         base={UL: 36, SU: 4}, act={SU: 4, UL: 36},
         start=date(2026, 7, 20), finish=date(2026, 8, 7),
         evidence="38fe0d5 (26 Jul) to c046291 (7 Aug)"),
    dict(id="M", wbs="4.4", name="Label dashboard", owner=SH, pred=["L"], dur=15,
         base={SH: 25}, act={SH: 34},
         start=date(2026, 8, 10), finish=date(2026, 9, 18),
         evidence="Team record; last branch commit 2b4e1d8 (18 Sep)"),
    dict(id="N", wbs="5.1", name="Cap teacher-student distillation", owner=SU, pred=["L"], dur=10,
         base={SU: 25, UL: 4}, act={SU: 27, UL: 4},
         start=date(2026, 8, 24), finish=date(2026, 9, 7),
         evidence="1455412 (4 Sep) to 4b40def (7 Sep)"),
    dict(id="O", wbs="5.2", name="Station 1 production pipeline", owner=SU, pred=["N"], dur=10,
         base={SU: 37, SH: 4}, act={SU: 37, SH: 4},
         start=date(2026, 9, 8), finish=date(2026, 9, 14),
         evidence="8cf38d4 (14 Sep)"),
    dict(id="P", wbs="5.3", name="Docker and TensorRT deployment", owner=SU, pred=["O"], dur=8,
         base={SU: 29}, act={SU: 35},
         start=date(2026, 9, 15), finish=date(2026, 10, 22),
         evidence="483f566 (15 Sep), a177fdf (16 Sep), 72f0722 (22 Oct)"),
    dict(id="Q", wbs="5.4", name="Audit, Replay, Active Learning workstations", owner=SU, pred=["O"], dur=10,
         base={SU: 29}, act={SU: 25, SH: 8},
         start=date(2026, 9, 21), finish=date(2026, 10, 22),
         evidence="c8f0524 (22 Oct)"),
    dict(id="R", wbs="5.5", name="Station 2 integration, env-driven paths", owner=SU, pred=["P", "Q"], dur=5,
         base={SU: 21}, act={SU: 21},
         start=date(2026, 10, 5), finish=date(2026, 10, 26),
         evidence="8135fed (26 Oct)"),
    dict(id="S", wbs="5.6", name="GPU verification and handover", owner=UL, pred=["M", "R"], dur=3,
         base={UL: 8, SU: 4, SH: 4}, act={SU: 4, UL: 12, SH: 4},
         start=date(2026, 10, 19), finish=date(2026, 10, 26),
         evidence="Team demonstration; no latency figures retained"),
]

WBS_GROUPS = [
    ("1", "Initiation and research"),
    ("2", "Inspection platform"),
    ("3", "Cap and fill models (first system)"),
    ("4", "Label anomaly pipeline"),
    ("5", "Final system and deployment"),
    ("6", "Reporting"),
    ("7", "Guide reviews"),
]
MENTOR_WBS = "7.1"
