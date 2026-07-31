"""Exercise the scoring math against cases with known answers."""
import sys, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vlm.eval.metrics import (score_classification, paired_bootstrap_delta,
                              score_extraction, match_items, percentile,
                              latency_summary, bootstrap_ci)

C = ["Bill Summary","Bill Detail","Pharmacy Bill","Lab Bill","Other"]
fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <- {extra}"))
    if not cond: fails.append(name)

# --- perfect classifier
g = ["Bill Detail"]*10 + ["Lab Bill"]*10
r = score_classification(g, list(g), C, resamples=2000)
check("perfect acc==1.0", r.accuracy == 1.0)
check("perfect CI is degenerate [1,1]", r.accuracy_ci95 == (1.0,1.0), r.accuracy_ci95)

# --- known 75% accuracy
g = ["Bill Detail"]*4
p = ["Bill Detail"]*3 + ["Lab Bill"]
r = score_classification(g, p, C, resamples=2000)
check("acc==0.75", r.accuracy == 0.75, r.accuracy)
check("CI brackets point estimate", r.accuracy_ci95[0] <= 0.75 <= r.accuracy_ci95[1], r.accuracy_ci95)

# --- macro-F1 ignores classes absent from gold AND pred
g = ["Bill Detail"]*5; p = ["Bill Detail"]*5
r = score_classification(g, p, C, resamples=500)
check("macro-F1 skips unused classes (==1.0 not 0.2)", r.macro_f1 == 1.0, r.macro_f1)

# --- small-n CI really is wide (the core methodological claim)
random.seed(1)
g = [random.choice(C[:3]) for _ in range(15)]
p = list(g)
for i in range(2): p[i] = "Other"          # 13/15 = 0.867
r = score_classification(g, p, C, resamples=5000)
width = r.accuracy_ci95[1] - r.accuracy_ci95[0]
check("n=15 CI width > 0.20", width > 0.20, f"width={width:.3f} ci={r.accuracy_ci95}")
print(f"      n=15, acc={r.accuracy:.3f}, CI={r.accuracy_ci95}, width={width:.3f}")

# --- paired bootstrap: identical systems -> delta 0, CI contains 0, p high
g = [random.choice(C) for _ in range(40)]
d = paired_bootstrap_delta(g, list(g), list(g), resamples=3000)
check("identical systems delta==0", d["delta"] == 0.0, d)
check("identical systems p==1.0", d["p_value"] >= 0.99, d)

# --- paired bootstrap: A strictly better than B
a = list(g)
b = ["Other"]*40
d = paired_bootstrap_delta(g, a, b, resamples=3000)
check("A>>B delta>0", d["delta"] > 0.5, d)
check("A>>B CI excludes 0", d["ci95"][0] > 0, d)
check("A>>B p small", d["p_value"] < 0.05, d)

# --- item matching
gold = [{"item_name":"BED CHARGE GENERAL WARD","item_amount":1500.0},
        {"item_name":"CONSULTATION","item_amount":500.0},
        {"item_name":"CBC TEST","item_amount":250.0}]
pred = [{"item_name":"Bed Charge General Ward","item_amount":1500.0},   # name variant, same amt
        {"item_name":"CONSULTATION","item_amount":999.0},               # right name, WRONG amt
        {"item_name":"XRAY CHEST","item_amount":800.0}]                 # hallucinated
m, missed, spur = match_items(gold, pred)
check("case/space-insensitive name match", len(m) == 1, [ (a['item_name'],b['item_name']) for a,b in m])
check("wrong amount is NOT a match", all(x["item_name"]!="CONSULTATION" for _,x in m))
check("2 missed", len(missed) == 2, missed)
check("2 spurious", len(spur) == 2, spur)

e = score_extraction([gold],[pred], resamples=500)
check("extraction P==1/3", abs(e["precision"] - 1/3) < 1e-3, e["precision"])
check("extraction R==1/3", abs(e["recall"] - 1/3) < 1e-3, e["recall"])
check("grand total pct error computed", e["grand_total_abs_pct_error"] is not None, e)
print(f"      extraction: {e['precision']:.3f}P {e['recall']:.3f}R f1={e['micro_f1']:.3f} totalErr={e['grand_total_abs_pct_error']}%")

# --- percentile against known values
v = [1,2,3,4,5,6,7,8,9,10]
check("p50 of 1..10 == 5.5", percentile(v,50) == 5.5, percentile(v,50))
check("p90 of 1..10 == 9.1", abs(percentile(v,90)-9.1) < 1e-9, percentile(v,90))
check("p100 == max", percentile(v,100) == 10)
ls = latency_summary([0.5,1.0,1.5])
check("latency mean==1.0", ls["mean_s"] == 1.0, ls)

print("\n" + ("ALL METRIC TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
