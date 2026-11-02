"""Generate every calculated table and diagram for the EPM report.

Standard library only. Reads data/project_data.py, computes the CPM schedule and
earned value, asserts the template's consistency checks, writes generated/*.tex.
"""
import calendar
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import project_data as d  # noqa: E402

OUT = ROOT / "generated"
OUT.mkdir(exist_ok=True)

AC = d.ACTIVITIES
BY = {a["id"]: a for a in AC}


# ---------- calendar ----------
def is_wd(x):
    return x.weekday() < 5


def wd_date(i):
    """Date of working day index i, where index 0 is the project start."""
    x = d.PROJECT_START
    n = 0
    while True:
        if is_wd(x):
            if n == i:
                return x
            n += 1
        x += timedelta(days=1)


def wd_between(a, b):
    """Working days in the inclusive range a..b."""
    n, x = 0, a
    while x <= b:
        n += is_wd(x)
        x += timedelta(days=1)
    return n


def slip(planned, actual):
    """Working days by which actual is later than planned (negative if earlier)."""
    if actual > planned:
        return wd_between(planned + timedelta(days=1), actual)
    return -wd_between(actual + timedelta(days=1), planned)


def fmt_date(x):
    return f"{x.day} {x:%b}"


def fmt_full(x):
    return f"{x.day} {x:%B %Y}"


# ---------- formatting ----------
def rs(v):
    """Indian digit grouping, with the rupee sign."""
    neg = v < 0
    s = str(abs(int(round(v))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if neg else "") + "₹" + s


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


TEXT_W, COL_PAD = 17.5, 0.1  # cm: text width and \tabcolsep, from main.tex


def fit(col_spec):
    """Scale p-column widths so the table spans the text width exactly."""
    widths = [float(w) for w in re.findall(r"p\{([\d.]+)cm\}", col_spec)]
    target = TEXT_W - 2 * COL_PAD * len(widths)
    new = [round(w * target / sum(widths), 2) for w in widths]
    new[-1] = round(new[-1] + target - sum(new), 2)
    it = iter(new)
    return re.sub(r"p\{[\d.]+cm\}", lambda m: f"p{{{next(it)}cm}}", col_spec)


def table(col_spec, header, rows, caption, long=True, bold_last=0):
    """Navy-headed table; long tables repeat their header across pages."""
    col_spec = re.sub(r"(?<!\\arraybackslash\})p\{", r">{\\raggedright\\arraybackslash}p{", fit(col_spec))
    head = " & ".join(rf"\thead{{{h}}}" for h in header) + r" \\"
    body = []
    for i, r in enumerate(rows):
        cells = [str(c) for c in r]
        if bold_last and i >= len(rows) - bold_last:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        body.append(" & ".join(cells) + r" \\ \hline")
    return "\n".join([
        rf"\begin{{longtable}}{{{col_spec}}}",
        r"\hline\rowcolor{navy}" + head + r" \hline\endfirsthead",
        r"\hline\rowcolor{navy}" + head + r" \hline\endhead",
        *body,
        r"\end{longtable}\nopagebreak",
        rf"\vspace{{-1.2em}}\nopagebreak\tcap{{{caption}}}",
    ]) + "\n"


def write(name, text):
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")


# ---------- CPM ----------
for a in AC:
    a["succ"] = [b["id"] for b in AC if a["id"] in b["pred"]]
for a in AC:  # data file is in topological order
    a["ES"] = max((BY[p]["EF"] for p in a["pred"]), default=0)
    a["EF"] = a["ES"] + a["dur"]
PROJECT_DUR = max(a["EF"] for a in AC)
for a in reversed(AC):
    a["LF"] = min((BY[s]["LS"] for s in a["succ"]), default=PROJECT_DUR)
    a["LS"] = a["LF"] - a["dur"]
for a in AC:
    a["TS"] = a["LS"] - a["ES"]
    a["FS"] = min((BY[s]["ES"] for s in a["succ"]), default=PROJECT_DUR) - a["EF"]
    a["crit"] = a["TS"] == 0
    a["pstart"] = wd_date(a["ES"])
    a["pfinish"] = wd_date(a["EF"] - 1)
    a["lfinish"] = wd_date(a["LF"] - 1)
    a["slip"] = slip(a["pfinish"], a["finish"])
    a["bh"] = sum(a["base"].values())
    a["ah"] = sum(a["act"].values())
    a["tbc"] = sum(h * d.RATE[m] for m, h in a["base"].items())
    a["cac"] = sum(h * d.RATE[m] for m, h in a["act"].items())

CRITICAL = [a["id"] for a in AC if a["crit"]]
assert wd_date(PROJECT_DUR - 1) == d.PLANNED_FINISH, "planned finish mismatch"

# ---------- earned value ----------
MONTH_ENDS = [date(y, m, calendar.monthrange(y, m)[1]) for y, m in d.PERIODS]
MENTOR_TBC = d.MENTOR_BASELINE_HOURS * d.MENTOR_RATE
MENTOR_CAC = d.MENTOR_ACTUAL_HOURS * d.MENTOR_RATE
TBC = sum(a["tbc"] for a in AC) + MENTOR_TBC
RESERVE = round(TBC * d.CONTINGENCY_PCT / 100)


def linear(total, start, end, at):
    if at < start:
        return 0
    if at >= end:
        return total
    return round(total * wd_between(start, at) / wd_between(start, end))


def pct_at(a, at):
    """Physical percent complete: elapsed share of actual window, to nearest 5%."""
    if at >= a["finish"]:
        return 100
    if at < a["start"]:
        return 0
    return min(95, 5 * round(20 * wd_between(a["start"], at) / wd_between(a["start"], a["finish"])))


def cbc_at(at):
    return sum(linear(a["tbc"], a["pstart"], a["pfinish"], at) for a in AC) + linear(
        MENTOR_TBC, d.PROJECT_START, d.PLANNED_FINISH, at)


def cac_at(at):
    return sum(linear(a["cac"], a["start"], a["finish"], at) for a in AC) + linear(
        MENTOR_CAC, d.PROJECT_START, d.STATUS_DATE, at)


def cev_at(at):
    # Guide reviews are level of effort: earned in step with elapsed project time.
    return sum(a["tbc"] * pct_at(a, at) / 100 for a in AC) + linear(
        MENTOR_TBC, d.PROJECT_START, d.STATUS_DATE, at)


def measures(at):
    cbc, cac, cev = cbc_at(at), cac_at(at), round(cev_at(at))
    m = dict(cbc=cbc, cac=cac, cev=cev, cpi=cev / cac, cv=cev - cac, sv=cev - cbc)
    m["f1"] = TBC / m["cpi"]
    m["f2"] = cac + (TBC - cev)
    m["tcpi"] = (TBC - cev) / (TBC - cac) if TBC != cac else 0
    return m


ST = measures(d.STATUS_DATE)
IN = measures(d.INTERIM_DATE)
SERIES = [dict(end=e, **measures(min(e, d.STATUS_DATE))) for e in MONTH_ENDS]

# ---------- consistency assertions ----------
assert sum(a["tbc"] for a in AC) + MENTOR_TBC == TBC
assert ST["cbc"] == TBC, "all work is scheduled before the Status Date"
assert ST["cev"] == TBC, "project is 100 percent complete"
assert ST["cac"] == sum(a["cac"] for a in AC) + MENTOR_CAC
assert slip(d.PLANNED_FINISH, d.STATUS_DATE) == 18
assert PROJECT_DUR == 173
for a in AC:
    assert a["owner"] in a["base"], f"{a['id']} owner has no hours"
    assert a["TS"] >= 0 and a["FS"] >= 0
    assert a["FS"] <= a["TS"]
    assert a["start"] <= a["finish"]
for x, y in zip(SERIES, SERIES[1:]):
    assert x["cbc"] <= y["cbc"] and x["cac"] <= y["cac"] and x["cev"] <= y["cev"]
crit_len = 0
cur = AC[0]
while True:
    crit_len += cur["dur"]
    nxt = [BY[s] for s in cur["succ"] if BY[s]["crit"]]
    if not nxt:
        break
    cur = nxt[0]
assert crit_len == PROJECT_DUR, "critical path length must equal the planned duration"

for f in (ROOT / "sections").glob("*.tex"):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith(r"\simpletable{"):
            w = [float(x) for x in re.findall(r"(?:L|p)\{([\d.]+)cm\}", line)]
            assert abs(sum(w) + 2 * COL_PAD * len(w) - TEXT_W) < 0.005, f"table width in {f.name}: {line[:60]}"

# ---------- macros ----------
HOURS = {m: sum(a["act"].get(m, 0) for a in AC) for m in d.MEMBERS}
BHOURS = {m: sum(a["base"].get(m, 0) for a in AC) for m in d.MEMBERS}
assert HOURS == d.ACTUAL_HOURS, "member hours must match the team-confirmed totals"
assert ST["cac"] - TBC <= RESERVE, "overrun must stay within the reserve"


def over(*ids):
    return sum(BY[i]["cac"] - BY[i]["tbc"] for i in ids)


# Contingency per risk: fixed shares of the reserve, rounded to ₹10, remainder on R5.
CONT_SHARE = {"One": 14000, "Two": 8000, "Three": 0, "Four": 5000, "Six": 2000, "Seven": 2000, "Eight": 2000}
CONT = {k: round(RESERVE * v / 38100, -1) for k, v in CONT_SHARE.items()}
CONT["Five"] = RESERVE - sum(CONT.values())
assert sum(CONT.values()) == RESERVE and CONT["Five"] > 0
most_slack = max(AC, key=lambda a: a["TS"])
longest_crit = max((a for a in AC if a["crit"]), key=lambda a: a["dur"])
FCAC80 = TBC / 0.80
macros = {
    "TBC": rs(TBC), "RESERVE": rs(RESERVE), "FUNDING": rs(TBC + RESERVE),
    "MentorTBC": rs(MENTOR_TBC), "MentorCAC": rs(MENTOR_CAC),
    "CBC": rs(ST["cbc"]), "CAC": rs(ST["cac"]), "CEV": rs(ST["cev"]),
    "CPI": f"{ST['cpi']:.3f}", "CV": rs(ST["cv"]),
    "FCACone": rs(ST["f1"]), "FCACtwo": rs(ST["f2"]),
    "Overrun": rs(TBC - ST["cac"]), "OverrunAbs": rs(ST["cac"] - TBC),
    "ReserveLeft": rs(RESERVE - (ST["cac"] - TBC)),
    "ReserveUsedPct": f"{100 * (ST['cac'] - TBC) / RESERVE:.0f}",
    "iCBC": rs(IN["cbc"]), "iCAC": rs(IN["cac"]), "iCEV": rs(IN["cev"]),
    "iCPI": f"{IN['cpi']:.3f}", "iCV": rs(IN["cv"]), "iSV": rs(IN["sv"]),
    "iFCACone": rs(IN["f1"]), "iFCACtwo": rs(IN["f2"]), "iTCPI": f"{IN['tcpi']:.3f}",
    "iPct": f"{100 * IN['cev'] / TBC:.1f}",
    "iOverrun": rs(IN["f1"] - TBC),
    "FCACeighty": rs(FCAC80), "OverrunEighty": rs(FCAC80 - TBC),
    "StudentHoursPlan": f"{sum(BHOURS.values()):,}", "StudentHoursActual": f"{sum(HOURS.values()):,}",
    "HoursSu": str(HOURS[d.SU]), "HoursUl": str(HOURS[d.UL]), "HoursSh": str(HOURS[d.SH]),
    "ProjectDur": str(PROJECT_DUR), "CriticalPath": " -- ".join(CRITICAL),
    "SlackMostId": most_slack["id"], "SlackMostName": esc(most_slack["name"]), "SlackMost": str(most_slack["TS"]),
    "LongestId": longest_crit["id"], "LongestName": esc(longest_crit["name"]), "LongestDur": str(longest_crit["dur"]),
    "OverJ": rs(over("J")), "OverK": rs(over("K")), "OverN": rs(over("N")), "OverP": rs(over("P")),
    "RiskActOne": rs(over("F", "G", "J", "K")),
    **{f"Cont{k}": rs(v) for k, v in CONT.items()},
}
write("macros", "\n".join(rf"\newcommand{{\v{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

# ---------- WBS and RAM ----------
rows = []
for num, title in d.WBS_GROUPS:
    members = [a for a in AC if a["wbs"].startswith(num + ".")]
    sub = sum(a["tbc"] for a in members) + (MENTOR_TBC if num == "7" else 0)
    rows.append([rf"\textbf{{{num}}}", rf"\textbf{{{esc(title)}}}", "", rf"\textbf{{{rs(sub)}}}"])
    for a in sorted(members, key=lambda a: a["wbs"]):
        rows.append([a["wbs"], f"{esc(a['name'])} ({a['id']})", a["owner"], rs(a["tbc"])])
    if num == "7":
        rows.append([d.MENTOR_WBS, "Weekly guide reviews (level of effort)", d.SU, rs(MENTOR_TBC)])
rows.append(["", "Project total budgeted cost (TBC)", "", rs(TBC)])
write("wbs_table", table(r"p{1.5cm}p{8.6cm}p{2.5cm}>{\raggedleft\arraybackslash}p{2.2cm}",
                         ["WBS number", "Deliverable / work package", "Owner (one person)", "TBC (₹)"], rows,
                         "Table 3.3 - Work breakdown structure with work package budgets", bold_last=1))

GUIDE_S = {"A", "B", "F", "I", "J", "L", "N", "S"}
CLIENT_S = {"B", "I", "S"}
rows = []
for a in sorted(AC, key=lambda a: a["wbs"]):
    cells = [f"{a['wbs']} {esc(a['name'])}"]
    for m in d.MEMBERS:
        cells.append("P" if a["owner"] == m else ("S" if m in a["act"] else ""))
    assert cells[1:].count("P") == 1, a["id"]
    cells += ["S" if a["id"] in GUIDE_S else "", "S" if a["id"] in CLIENT_S else ""]
    rows.append(cells)
rows.append([f"{d.MENTOR_WBS} Weekly guide reviews", "P", "S", "S", "S", ""])
center = r">{\centering\arraybackslash}p"
write("ram_table", table(rf"p{{7.3cm}}{center}{{1.5cm}}{center}{{1.1cm}}{center}{{1.3cm}}{center}{{1.2cm}}{center}{{1.1cm}}",
                         ["WBS work package", "Sushanth", "Ullas", "Shreyas", "Mentor", "Client"], rows,
                         "Table 3.4 - Responsibility assignment matrix (P = primary, S = support)"))

# ---------- schedule ----------
def res_text(hours):
    return ", ".join(f"{m[:2]} {h}" for m, h in hours.items())


rows = [[a["id"], esc(a["name"]), ", ".join(a["pred"]) or "-", res_text(a["base"]), a["dur"], a["owner"]] for a in AC]
write("activity_table", table(r"p{0.6cm}p{6cm}p{1.9cm}p{3.3cm}>{\centering\arraybackslash}p{1.3cm}p{1.9cm}",
                              ["\\#", "Activity", "Predecessors", "Resources (hours)", "Duration (days)", "Estimated by"],
                              rows, "Table 4.1 - Activity list with resource and duration estimates"))

rows = [[a["id"], esc(a["name"]), a["dur"], a["ES"], a["EF"], a["LS"], a["LF"], a["TS"], a["FS"],
         r"\textbf{Yes}" if a["crit"] else "No"] for a in AC]
c = r">{\centering\arraybackslash}p{0.75cm}"
write("cpm_table", table(rf"p{{0.5cm}}p{{5.7cm}}{c}{c}{c}{c}{c}{c}{c}>{{\centering\arraybackslash}}p{{1.3cm}}",
                         ["\\#", "Activity", "Dur.", "ES", "EF", "LS", "LF", "Total slack", "Free slack", "Critical?"],
                         rows, "Table 4.2 - Forward and backward pass, slack and critical path"))

CAUSE = {
    "A": ("Survey ran one day past its two weeks", "-1 on the critical path; recovered within B"),
    "E": ("Redesign landed on 4 May", "-1 on the critical path"),
    "F": ("Bottle cap models needed repeated iteration: small caps, class mapping, white-only colour bias",
          "Used all 5 days of slack; path at -10"),
    "G": ("Inherited F's delay; water datasets had to be merged", "Path stayed at -10"),
    "H": ("Waited on G", "-5 on the critical path; recovered by overlapping I"),
    "J": ("Paper revision and a pause after the thesis delayed the start to mid-June", "-10 on the critical path"),
    "K": ("YOLO ROI detector tried, then replaced by template matching", "-12 on the critical path"),
    "M": ("Label dashboard finished after the models", "Used 15 of its 20 days of slack"),
    "N": ("CSITSS rejection (22 Aug) took priority, delaying the start", "-11 on the critical path"),
    "O": ("Inherited N's delay", "-6 on the critical path"),
    "P": ("TensorRT static batch mismatch; dynamic batching and versioning reworked", "Used its 2 days of slack; path at -24"),
    "Q": ("Started after the rework on P began; delivered with P on 22 Oct", "-24 on the critical path"),
    "R": ("Waited on P and Q; Station 2 merge into one UI", "-21 on the critical path"),
    "S": ("Waited on R", "Project finished 18 days late"),
}
rows = []
for a in AC:
    if a["slip"] != 0:
        cause, effect = CAUSE[a["id"]]
        rows.append([f"{a['id']} {esc(a['name'])}", fmt_date(a["pfinish"]), fmt_date(a["finish"]),
                     f"{a['slip']:+d}", cause, effect])
write("variance_table", table(r"p{3.3cm}p{1.2cm}p{1.2cm}>{\centering\arraybackslash}p{1.35cm}p{4.6cm}p{3.4cm}",
                              ["Activity", "Planned finish", "Actual finish", "Variance (days)", "Cause", "Effect on total slack"],
                              rows, "Table 4.4 - Schedule variance analysis"))

MILESTONES = [("Literature survey complete", "A"), ("Requirements and architecture complete", "B"),
              ("Platform scaffold pushed", "C"), ("Cap models accepted", "F"), ("First integrated demo", "H"),
              ("Thesis complete", "I"), ("Label models calibrated", "L"),
              ("Student cap model accepted", "N"), ("Station 1 production pipeline", "O"),
              ("Station 2 integrated", "R"), ("GPU verification and handover", "S")]
rows = []
for name, i in MILESTONES:
    a = BY[i]
    s = "Met" if a["slip"] <= 0 else f"Late by {a['slip']} day" + ("s" if a["slip"] > 1 else "")
    rows.append([name, fmt_date(a["pfinish"]), fmt_date(a["finish"]), s])
write("milestone_table", table("p{7.6cm}p{2.2cm}p{2.8cm}p{2.5cm}",
                               ["Milestone", "Planned date", "Actual date", "Status"], rows,
                               "Table 4.3 - Milestones, all in 2026 (a milestone has zero duration)"))

# ---------- cost tables ----------
BASIS = {
    "A": "3 members for 2 weeks at about 8 h/week", "B": "Mostly PM; sized against the 15-day duration",
    "C": "Size of the Phase 0-4 scaffold", "D": "API routers and DB schema count",
    "E": "Page count of the dashboard", "F": "Training runs expected, plus analysis",
    "G": "Two detectors and fill-ratio logic", "H": "Integration week",
    "I": "Thesis, paper and slides, two weeks", "J": "One anomaly baseline",
    "K": "528 crops plus localisation trials", "L": "4 ROI models with calibration",
    "M": "Single-purpose dashboard", "N": "Teacher and student training watch",
    "O": "Six pipeline layers", "P": "Compose, engine compile, stress test",
    "Q": "Three operator pages", "R": "Mirror of Station 1 layout",
    "S": "Checks on target PC and handover",
}
rows = [[a["wbs"], esc(a["name"]), rs(a["tbc"]), "₹0", BASIS[a["id"]], rs(a["tbc"])]
        for a in sorted(AC, key=lambda a: a["wbs"])]
rows.append([d.MENTOR_WBS, "Weekly guide reviews", rs(MENTOR_TBC), "₹0",
             f"{d.MENTOR_BASELINE_HOURS} h planned at ₹1,000/h", rs(MENTOR_TBC)])
rows.append(["", "Total budgeted cost (TBC)", rs(TBC), "₹0", "Sum of work packages", rs(TBC)])
rows.append(["", "Contingency reserve (held separately)", "", "", f"{d.CONTINGENCY_PCT}\\% of TBC", rs(RESERVE)])
rows.append(["", "Funding limit", "", "", "TBC + reserve", rs(TBC + RESERVE)])
r = r">{\raggedleft\arraybackslash}p"
write("cost_table", table(rf"p{{0.9cm}}p{{5cm}}{r}{{1.5cm}}{r}{{1.45cm}}p{{4.2cm}}{r}{{1.5cm}}",
                          ["WBS number", "Work package", "Labour (₹)", "Materials / other (₹)", "Basis of estimate", "TBC (₹)"],
                          rows, "Table 5.1 - Activity cost estimates and total budgeted cost (all amounts in ₹)", bold_last=3))

month_labels = [calendar.month_abbr[m] for _, m in d.PERIODS]
items = [(a["id"], a["tbc"], a["pstart"], a["pfinish"]) for a in AC] + [
    ("7.1", MENTOR_TBC, d.PROJECT_START, d.PLANNED_FINISH)]
rows, totals = [], [0] * len(MONTH_ENDS)
prev_ends = [date(2026, 1, 31)] + MONTH_ENDS[:-1]
for name, tbc, s, f in items:
    cells, row_sum = [name], 0
    for i, (p, e) in enumerate(zip(prev_ends, MONTH_ENDS)):
        v = linear(tbc, s, f, e) - linear(tbc, s, f, p)
        totals[i] += v
        row_sum += v
        cells.append(f"{v:,}" if v else "")
    assert row_sum == tbc
    rows.append(cells + [f"{tbc:,}"])
assert sum(totals) == TBC
rows.append(["Period"] + [f"{t:,}" for t in totals] + [f"{TBC:,}"])
cum, cum_cells = 0, []
for i, t in enumerate(totals):
    cum += t
    assert cum == SERIES[i]["cbc"]
    cum_cells.append(f"{cum:,}")
rows.append(["CBC"] + cum_cells + [""])
c = r">{\raggedleft\arraybackslash}p{1.15cm}"
write("phased_table", "{\\footnotesize\n" + table(
    "p{0.85cm}" + c * 9 + r">{\raggedleft\arraybackslash}p{1.25cm}",
    ["WP"] + [f"P{i + 1} {l}" for i, l in enumerate(month_labels)] + ["Total (₹)"], rows,
    "Table 5.2 - Time-phased budget; the last row is the cumulative budgeted cost (CBC)", bold_last=2) + "}\n")

rows = [[a["wbs"], esc(a["name"]), rs(a["tbc"]), rs(a["cac"]), "₹0", rs(a["cac"])]
        for a in sorted(AC, key=lambda a: a["wbs"])]
rows.append([d.MENTOR_WBS, "Weekly guide reviews", rs(MENTOR_TBC), rs(MENTOR_CAC), "₹0", rs(MENTOR_CAC)])
rows.append(["", "Total", rs(ST["cbc"]), rs(ST["cac"]), "₹0", rs(ST["cac"])])
write("actual_table", table(rf"p{{0.9cm}}p{{7cm}}{r}{{1.7cm}}{r}{{1.7cm}}{r}{{1.5cm}}{r}{{1.7cm}}",
                            ["WBS", "Work package", "CBC to date (₹)", "Spent (₹, notional)", "Committed (₹)", "CAC (₹)"],
                            rows, "Table 5.3 - Actual and committed cost as of the Status Date", bold_last=1))

HOW = {
    "A": "Reference list in the thesis", "B": "Planning documents exist", "C": "Code in repo",
    "D": "API and schema commits", "E": "Dashboard commits", "F": "Models in repo, metrics logged",
    "G": "Fill models run in pipeline", "H": "Commit 8c54630", "I": "Thesis and plagiarism report",
    "J": "PatchCore scripts, membank", "K": "Crops, localisation committed", "L": "Thresholds, crop test records",
    "M": "Separate dashboard runs", "N": "results.csv, 126 epochs", "O": r"test\_step1-6 scripts",
    "P": "Compose files, stress test", "Q": "Commit c8f0524", "R": r"test\_system\_e2e.py", "S": "Team demonstration",
}
rows, i_rows = [], []
for a in sorted(AC, key=lambda a: a["wbs"]):
    p = pct_at(a, d.INTERIM_DATE)
    rows.append([a["wbs"], esc(a["name"]), rs(a["tbc"]), "100\\%", HOW[a["id"]], rs(a["tbc"])])
    if p or a["start"] <= d.INTERIM_DATE:
        i_rows.append([a["wbs"], esc(a["name"]), rs(a["tbc"]), f"{p}\\%", rs(a["tbc"] * p / 100)])
rows.append([d.MENTOR_WBS, "Weekly guide reviews", rs(MENTOR_TBC), "100\\%", "Level of effort", rs(MENTOR_TBC)])
rows.append(["", "Total", rs(TBC), "100\\%", "", rs(ST["cev"])])
write("ev_table", table(rf"p{{0.9cm}}p{{5.6cm}}{r}{{1.5cm}}>{{\centering\arraybackslash}}p{{1.3cm}}p{{3.9cm}}{r}{{1.5cm}}",
                        ["WBS", "Work package", "TBC (₹)", "Percent complete", "How judged", "CEV (₹)"],
                        rows, "Table 5.4 - Cumulative earned value", bold_last=1))
rows = [[f"{m:%b}", rs(s["cbc"]), rs(s["cac"]), rs(s["cev"]), f"{s['cpi']:.3f}", rs(s["cv"]), rs(s["sv"])]
        for m, s in zip(MONTH_ENDS, SERIES)]
write("monthly_table", table(f"p{{1.2cm}}{r}{{2.1cm}}{r}{{2.1cm}}{r}{{2.1cm}}{r}{{1.3cm}}{r}{{1.9cm}}{r}{{1.9cm}}",
                             ["Month end", "CBC", "CAC", "CEV", "CPI", "CV", "CEV - CBC"], rows,
                             "Table 5.8 - Monthly control data behind Figure 5.1 (October is the Status Date)"))

# ---------- contribution and consistency ----------
own = {m: ", ".join(a["wbs"] for a in sorted(AC, key=lambda a: a["wbs"]) if a["owner"] == m) for m in d.MEMBERS}
own[d.SU] += ", " + d.MENTOR_WBS
MAIN = {
    d.SU: "PM; backend, distillation, production pipeline, Docker and TensorRT, final UI, Station 2 merge",
    d.UL: "Cap and fill model iterations, label anomaly models, GPU verification and handover",
    d.SH: "Platform scaffold, dashboards, template matching for label localisation",
}
FULL = {d.SU: "Sushanth S", d.UL: "Ullas N", d.SH: "Shreyas Bairy K S"}
rows = [[FULL[m], own[m], HOURS[m], MAIN[m], "Yes (below)"] for m in d.MEMBERS]
write("contribution_table", table(r"p{2.2cm}p{4.1cm}>{\centering\arraybackslash}p{1.1cm}p{5.4cm}p{1.9cm}",
                                  ["Member", "WBS packages owned (P)", "Hours worked", "Main contribution", "Reflection attached?"],
                                  rows, "Table 9.1 - Individual contribution"))

rows = [
    ["Work package budgets add up to the project TBC", "Table 3.3 = Table 5.1", rs(TBC), "Yes"],
    ["CBC at the Status Date", "Table 5.2 = Table 5.5", rs(ST["cbc"]), "Yes"],
    ["CAC includes committed amounts", "Table 5.3 = Table 5.5", f"{rs(ST['cac'])} (committed ₹0)", "Yes"],
    ["CEV uses the same percent complete as the status report", "Table 5.4 = Section 1", f"{rs(ST['cev'])} (100\\%)", "Yes"],
    ["Critical path duration equals the planned project duration", "Table 4.2 = Figure 4.1", f"{PROJECT_DUR} days", "Yes"],
    ["Every activity's total slack is consistent with the critical path", "Table 4.2",
     f"{len(CRITICAL)} activities at 0; none negative", "Yes"],
    ["Contingency reserve appears in the budget", "Table 5.1", rs(RESERVE), "Yes"],
    ["Each RAM row has exactly one P", "Table 3.4", f"{len(AC) + 1} of {len(AC) + 1} rows", "Yes"],
]
write("consistency_table", table("p{6.6cm}p{3cm}p{3.9cm}p{1.2cm}", ["Check", "Where it comes from", "Figure", "Agrees?"], rows,
                                 "Table 10.1 - Consistency checks. The build script fails if any of these does not hold."))

# ---------- Gantt ----------
BLUE, ORANGE, AQUA = "2a78d6", "eb6834", "1baf7a"
CHART_START, CHART_END = date(2026, 2, 1), date(2026, 10, 31)
GANTT_W = 11.2


def xpos(x):
    return GANTT_W * (x - CHART_START).days / (CHART_END - CHART_START).days


def gantt(actual):
    row_h = 0.62 if actual else 0.46
    n = len(AC)
    h = n * row_h
    out = [r"\begin{tikzpicture}[font=\scriptsize]"]
    for i, ((y, m), e) in enumerate(zip(d.PERIODS, MONTH_ENDS)):
        x0, x1 = xpos(date(y, m, 1)), xpos(e)
        out.append(rf"\draw[gridline] ({x0:.2f},0.25) -- ({x0:.2f},{-h:.2f});")
        out.append(rf"\node[anchor=south] at ({(x0 + x1) / 2:.2f},0.2) {{{calendar.month_abbr[m]}}};")
    out.append(rf"\draw[gridline] ({GANTT_W:.2f},0.25) -- ({GANTT_W:.2f},{-h:.2f});")
    for i, a in enumerate(AC):
        y = -row_h * (i + 0.5)
        label = esc(a["name"])
        out.append(rf"\node[anchor=east] at (-0.1,{y:.2f}) {{{a['id']}\ \ {label}}};")
        x0, x1 = xpos(a["pstart"]), xpos(a["pfinish"] + timedelta(days=1))
        style = "critbar" if a["crit"] else "plainbar"
        top = y + (0.22 if actual else 0.13)
        bot = y + (0.02 if actual else -0.13)
        out.append(rf"\fill[{style}] ({x0:.2f},{bot:.2f}) rectangle ({x1:.2f},{top:.2f});")
        if a["TS"] > 0:
            xl = xpos(a["lfinish"] + timedelta(days=1))
            mid = (top + bot) / 2
            out.append(rf"\draw[slackline] ({x1:.2f},{mid:.2f}) -- ({xl:.2f},{mid:.2f});")
            out.append(rf"\draw[slackline,solid] ({xl:.2f},{mid - 0.07:.2f}) -- ({xl:.2f},{mid + 0.07:.2f});")
        if actual:
            ax0, ax1 = xpos(a["start"]), xpos(a["finish"] + timedelta(days=1))
            out.append(rf"\fill[actualbar] ({ax0:.2f},{y - 0.22:.2f}) rectangle ({ax1:.2f},{y - 0.04:.2f});")
    xp = xpos(d.PLANNED_FINISH + timedelta(days=1))
    out.append(rf"\draw[marker] ({xp:.2f},0) -- ({xp:.2f},{-h:.2f});")
    out.append(rf"\node[anchor=north,align=center] at ({xp:.2f},{-h:.2f}) {{Planned finish\\30 Sep}};")
    if actual:
        xs = xpos(d.STATUS_DATE + timedelta(days=1))
        out.append(rf"\draw[marker] ({xs:.2f},0) -- ({xs:.2f},{-h:.2f});")
        out.append(rf"\node[anchor=north west,align=left] at ({xs:.2f},{-h:.2f}) {{Actual\\26 Oct}};")
    ly = -h - 0.95
    lx = -5.4
    out.append(rf"\fill[critbar] ({lx:.2f},{ly:.2f}) rectangle ({lx + 0.5:.2f},{ly + 0.18:.2f});")
    out.append(rf"\node[anchor=west] at ({lx + 0.55:.2f},{ly + 0.09:.2f}) {{Critical}};")
    out.append(rf"\fill[plainbar] ({lx + 2:.2f},{ly:.2f}) rectangle ({lx + 2.5:.2f},{ly + 0.18:.2f});")
    out.append(rf"\node[anchor=west] at ({lx + 2.55:.2f},{ly + 0.09:.2f}) {{Non-critical}};")
    out.append(rf"\draw[slackline] ({lx + 4.8:.2f},{ly + 0.09:.2f}) -- ({lx + 5.3:.2f},{ly + 0.09:.2f});")
    out.append(rf"\node[anchor=west] at ({lx + 5.35:.2f},{ly + 0.09:.2f}) {{Total slack}};")
    if actual:
        out.append(rf"\fill[actualbar] ({lx + 7.3:.2f},{ly:.2f}) rectangle ({lx + 7.8:.2f},{ly + 0.18:.2f});")
        out.append(rf"\node[anchor=west] at ({lx + 7.85:.2f},{ly + 0.09:.2f}) {{Actual}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


write("gantt_baseline", gantt(False))
write("gantt_actual", gantt(True))

# ---------- network ----------
POS = {
    "A": (0, 1), "B": (1, 1), "C": (2, 2), "F": (2, 0), "D": (3, 3), "E": (3, 2), "G": (3, 0), "H": (4, 1), "I": (5, 1),
    "J": (0, -2), "K": (1, -2), "L": (2, -2), "N": (3, -2), "M": (3, -3.3), "O": (4, -2), "Q": (5, -2), "P": (5, -0.9),
    "R": (6, -2), "S": (7, -2),
}
out = [r"\resizebox{\textwidth}{!}{\begin{tikzpicture}[x=2.75cm,y=1.55cm,font=\footnotesize]"]
for a in AC:
    x, y = POS[a["id"]]
    style = "netcrit" if a["crit"] else "netplain"
    out.append(
        rf"\node[{style}] ({a['id']}) at ({x},{y}) {{\begin{{tabular}}{{@{{}}c@{{\hspace{{5pt}}}}c@{{\hspace{{5pt}}}}c@{{}}}}"
        rf"{a['ES']} & \textbf{{{a['id']}}} ({a['dur']}) & {a['EF']}\\ {a['LS']} & TS {a['TS']} & {a['LF']}\end{{tabular}}}};")
for a in AC:
    for s in a["succ"]:
        crit = a["crit"] and BY[s]["crit"] and a["EF"] == BY[s]["ES"]
        style = "netedgecrit" if crit else "netedge"
        pair = (a["id"], s)
        if pair == ("I", "J"):
            out.append(rf"\draw[{style}] (I.south) |- (0,-0.45) -- (J.north);")
        elif pair == ("M", "S"):
            out.append(rf"\draw[{style}] (M.east) -| (S.south);")
        elif pair == ("P", "R"):
            out.append(rf"\draw[{style}] (P.east) -| (R.north);")
        elif pair == ("O", "P"):
            out.append(rf"\draw[{style}] (O.north) |- (P.west);")
        elif pair == ("L", "M"):
            out.append(rf"\draw[{style}] (L.south) |- (M.west);")
        else:
            out.append(rf"\draw[{style}] ({a['id']}.east) -- ({s}.west);")
out.append(r"\end{tikzpicture}}")
write("network", "\n".join(out) + "\n")

# ---------- WBS diagram ----------
out = [r"\resizebox{\textwidth}{!}{\begin{tikzpicture}[font=\footnotesize]",
       rf"\node[wbstop] (top) at ({(len(d.WBS_GROUPS) - 1) * 1.45:.2f},1.6) {{VisionQAI\\TBC {rs(TBC)}}};"]
for i, (num, title) in enumerate(d.WBS_GROUPS):
    x = i * 2.9
    out.append(rf"\node[wbsgroup] (g{num}) at ({x:.2f},0) {{{num} {esc(title)}}};")
    out.append(rf"\draw (top.south) -- ++(0,-0.3) -| (g{num}.north);")
    prev = f"g{num}"
    wps = sorted((a for a in AC if a["wbs"].startswith(num + ".")), key=lambda a: a["wbs"])
    labels = [f"{a['wbs']} {esc(a['name'])}" for a in wps] or [f"{d.MENTOR_WBS} Weekly guide reviews"]
    for j, text in enumerate(labels):
        out.append(rf"\node[wbsleaf,below=3mm of {prev}] (n{num}{j}) {{{text}}};")
        prev = f"n{num}{j}"
out.append(r"\end{tikzpicture}}")
write("wbs_diagram", "\n".join(out) + "\n")

# ---------- S-curves ----------
W, H = 13.0, 7.0
ymax_lakh = 2.0
tick_lakh = 0.5
step = W / len(SERIES)


def sx(i):
    return step * (i + 1)


def sy(v):
    return H * (v / 100000) / ymax_lakh


out = [r"\begin{tikzpicture}[font=\footnotesize]"]
for k in range(int(ymax_lakh / tick_lakh) + 1):
    v = k * tick_lakh
    out.append(rf"\draw[gridline] (0,{H * v / ymax_lakh:.2f}) -- ({W + 0.2:.2f},{H * v / ymax_lakh:.2f});")
    out.append(rf"\node[anchor=east,text=inksecondary] at (-0.05,{H * v / ymax_lakh:.2f}) {{{v:.1f}}};")
out.append(rf"\node[rotate=90,inner sep=0pt,text=inksecondary] at (-0.95,{H / 2:.2f}) {{₹ lakh, cumulative}};")
for i, m in enumerate(month_labels):
    out.append(rf"\node[anchor=north,text=inksecondary] at ({sx(i):.2f},-0.05) {{{m}}};")
out.append(rf"\draw[inksecondary] (0,0) -- ({W + 0.2:.2f},0);")
i_idx = [e.month for e in MONTH_ENDS].index(d.INTERIM_DATE.month)
out.append(rf"\draw[marker] ({sx(i_idx):.2f},0) -- ({sx(i_idx):.2f},{H - 0.7:.2f});")
out.append(rf"\node[anchor=west,text=inksecondary] at ({sx(i_idx):.2f},0.45) {{31 Jul checkpoint}};")
for key, style, mark in [("cbc", "curveCBC", "*"), ("cac", "curveCAC", "square*"), ("cev", "curveCEV", "triangle*")]:
    pts = " -- ".join(f"({sx(i):.2f},{sy(s[key]):.2f})" for i, s in enumerate(SERIES))
    out.append(rf"\draw[{style}] {pts};")
    for i, s in enumerate(SERIES):
        out.append(rf"\node[{style}mark] at ({sx(i):.2f},{sy(s[key]):.2f}) {{}};")
lx = 0.25
for label, style in [("CBC (budgeted)", "curveCBC"), ("CAC (actual)", "curveCAC"), ("CEV (earned)", "curveCEV")]:
    out.append(rf"\draw[{style}] ({lx:.2f},{H - 0.35:.2f}) -- ({lx + 0.7:.2f},{H - 0.35:.2f});")
    out.append(rf"\node[{style}mark] at ({lx + 0.35:.2f},{H - 0.35:.2f}) {{}};")
    out.append(rf"\node[anchor=west] at ({lx + 0.75:.2f},{H - 0.35:.2f}) {{{label}}};")
    lx += 3.5
end = SERIES[-1]
out.append(rf"\node[anchor=west,inner sep=0pt] at ({sx(8) + 0.15:.2f},{sy(end['cac']) + 0.15:.2f}) {{CAC}};")
out.append(rf"\node[anchor=west,inner sep=0pt] at ({sx(8) + 0.15:.2f},{sy(end['cev']) - 0.15:.2f}) {{CEV, CBC}};")
out.append(r"\end{tikzpicture}")
write("scurve", "\n".join(out) + "\n")

# ---------- summary for review ----------
if __name__ == "__main__":
    print(f"Duration {PROJECT_DUR} days; critical path: {' '.join(CRITICAL)}")
    print(f"{'id':3}{'dur':>4}{'ES':>4}{'EF':>4}{'LS':>4}{'LF':>4}{'TS':>4}{'FS':>4}  {'plan':>13}  {'actual':>13} {'slip':>4} {'bh':>4} {'ah':>4} {'TBC':>7} {'CAC':>7}")
    for a in AC:
        print(f"{a['id']:3}{a['dur']:4}{a['ES']:4}{a['EF']:4}{a['LS']:4}{a['LF']:4}{a['TS']:4}{a['FS']:4}  "
              f"{a['pstart']:%d%b}-{a['pfinish']:%d%b}  {a['start']:%d%b}-{a['finish']:%d%b} {a['slip']:4} "
              f"{a['bh']:4} {a['ah']:4} {a['tbc']:7} {a['cac']:7}")
    print("hours planned", BHOURS, sum(BHOURS.values()), "actual", HOURS, sum(HOURS.values()))
    print(f"TBC {TBC} reserve {RESERVE} funding {TBC + RESERVE}")
    print("month      CBC     CAC     CEV    CPI      CV      SV")
    for e, s in zip(MONTH_ENDS, SERIES):
        print(f"{e:%b} {s['cbc']:8} {s['cac']:7} {s['cev']:7} {s['cpi']:.3f} {s['cv']:7} {s['sv']:7}")
    print("status", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ST.items()})
    print("interim", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in IN.items()})
    print("All consistency assertions passed")
