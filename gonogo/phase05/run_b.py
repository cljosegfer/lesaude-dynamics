"""
Phase 0.5 on b: GPU Runs of 0.5a (exploratory intra), 3d and 3e
================================================================
Sequential driver of gonogo/part1/gain.py for go_no_go_plan.md revision 9 once b is the chosen representation.
Every stage is resumable: gain.py skips results already on disk with the same hparams, so rerunning a stage after an
allocation ends continues where it stopped.

Stages (in order; `all` runs them in this order):
  intra    exploratory ψ̂² inputs for intra and [intra; b]: state [b, intra, m], targets Δh (= Δintra) and Δb, Z = z,
           positive controls, clean and adjusted subsets   -> outputs/phase05/7_intra_gain/gain_{clean,adjusted}
  screen   b, state [b, m], target Δb, Z = z, clean subset: positive controls, C1 controls, ambiguous and venous codes;
           codes significant after z on b (CRT, BH within the GO family, >= 250 onsets) -> 8_3d_b/codes_3d.json
  tasks    3d on b: Task 1 (z, z+R, z+A) and Task 2 (dt, dt+A, z-setting_t1, z-setting_t1+A, z) for the Phase 0 codes
           significant after z plus the codes significant after z on b; Task 1 also for the positive controls as
           references. Frozen sets of 0.5d (A = C0 without the validation subset, R = frozen R)  -> 8_3d_b/gain
  tested   3e tested codes on b under z+A with residuals: positive controls, ambiguous codes, C1 controls -> 8_3d_b/gain
  rotation 3e withheld codes: each rotation code of group g under z + rot<g> (the other four groups) -> 9_3e_b/gain
3 seeds and 1,000 CRT draws throughout (gain.py defaults). Residuals are saved for every run (b has 39 dimensions).

Example
-------
python -I gonogo/phase05/run_b.py all
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P05 = ROOT / "outputs" / "phase05"
FROZEN = P05 / "freeze" / "label_sets.json"
GAIN = [sys.executable, "-I", str(ROOT / "gonogo" / "part1" / "gain.py")]
D3 = P05 / "8_3d_b"
D3E = P05 / "9_3e_b"
# Codes significant after z in Phase 0 (plan Tables 1 and 4, either target) and the rows of Table 5 (Task 2)
PHASE0_CODES = ["I10", "I25", "I50", "I85", "I95", "K59", "K80", "K22", "J06", "I82", "K21", "G43"]
TASK1 = ["z", "z+R", "z+A"]
TASK2 = ["dt", "dt+A", "z-setting_t1", "z-setting_t1+A", "z"]


def gain(*args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = GAIN + [str(a) for a in args]
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    print(f"  exit {r.returncode}: {' '.join(str(a) for a in args[:8])} ... (log {log.relative_to(ROOT)})", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"gain.py failed, see {log}")


def common(out):
    return ["--state", "bm", "--targets", "b", "--subset", "clean", "--save-residuals", "--keep-going",
            "--label-sets", FROZEN, "--out-dir", out]


def stage_intra():
    for subset in ("clean", "adjusted"):
        out = P05 / "7_intra_gain" / f"gain_{subset}"
        gain("--h-tag", "intra_last", "--state", "bhm", "--targets", "h,b", "--conds", "z", "--codes", "positive",
             "--subset", subset, "--save-residuals", "--out-dir", out, log=out / "run.log")


def screen_codes():
    sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
    from gain import bh, group_of
    rows = []
    for f in sorted((D3 / "gain").glob("clean_bm_b_z_*.json")):
        r = json.loads(f.read_text())
        rows.append({"code": r["code"], "group": group_of(r["code"]), "onsets": r["n_onsets"],
                     "p_crt": r["rlearner"]["crt"]["p"],
                     "E_x1e5": round((r["rlearner"]["G"] - r["rlearner"]["crt"]["mean"]) * 1e5, 3)})
    go = [x for x in rows if x["group"] != "positive" and x["onsets"] >= 250]
    q = bh([x["p_crt"] for x in go]) if go else []
    for x, qq in zip(go, q):
        x["q_bh_go"] = float(qq)
    sig_b = sorted(x["code"] for x in go if x["q_bh_go"] < 0.05)
    codes = list(dict.fromkeys(PHASE0_CODES + sig_b))
    out = {"phase0_codes": PHASE0_CODES, "significant_after_z_on_b": sig_b, "codes_3d": codes, "screen": rows,
           "rule": "CRT p of the R-learner gain, BH (q < 0.05) within the GO family (C1, venous, ambiguous) of codes "
                   "with >= 250 clean onsets"}
    (D3 / "codes_3d.json").write_text(json.dumps(out, indent=1))
    print(f"  significant after z on b: {sig_b}; 3d codes: {codes}", flush=True)
    return codes


def stage_screen():
    gain(*common(D3 / "gain"), "--conds", "z", "--codes", "positive,c1,ambiguous", log=D3 / "gain" / "run.log")
    return screen_codes()


def stage_tasks():
    codes = json.loads((D3 / "codes_3d.json").read_text())["codes_3d"] if (D3 / "codes_3d.json").exists() \
        else stage_screen()
    conds = list(dict.fromkeys(TASK1 + TASK2))
    gain(*common(D3 / "gain"), "--conds", ",".join(conds), "--codes", ",".join(codes), log=D3 / "gain" / "run.log")
    gain(*common(D3 / "gain"), "--conds", ",".join(TASK1), "--codes", "positive", log=D3 / "gain" / "run.log")


def stage_tested():
    gain(*common(D3 / "gain"), "--conds", "z+A", "--codes", "positive,ambiguous,c1", log=D3 / "gain" / "run.log")


def stage_rotation():
    fz = json.loads(FROZEN.read_text())
    for g, codes in fz["rotation_groups"].items():
        gain(*common(D3E / "gain"), "--conds", f"z+rot{g}", "--codes", ",".join(codes), log=D3E / "gain" / "run.log")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = {"intra": stage_intra, "screen": stage_screen, "tasks": stage_tasks, "tested": stage_tested,
             "rotation": stage_rotation}
    for name in (list(order) if stage == "all" else [stage]):
        print(f"== stage {name}", flush=True)
        order[name]()
    print("== done", flush=True)


if __name__ == "__main__":
    main()
