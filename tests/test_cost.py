import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vlm.eval import cost_model as cm

fails=[]
def check(n,c,e=""):
    print(("PASS " if c else "FAIL ")+n+("" if c else f"  <- {e}")); 
    if not c: fails.append(n)

# Hand-computable: 1M input tokens @ $0.15, 1M output @ $1.25
a = cm.api_cost(pages=1000, input_tokens=1_000_000, output_tokens=1_000_000)
check("api input $0.15", a.input_usd == 0.15, a.input_usd)
check("api output $1.25", a.output_usd == 1.25, a.output_usd)
check("api total $1.40", abs(a.total_usd-1.40)<1e-9, a.total_usd)
check("per-1k-pages == total (1000 pages)", abs(a.usd_per_1000_pages-1.40)<1e-9, a.usd_per_1000_pages)

# Local: T4 @$0.35/hr, 2s/page -> 1000 pages = 2000s = 0.5556hr = $0.1944
l = cm.local_cost(pages=1000, seconds_per_page=2.0, deployment="T4", training_gpu_hours=3.0)
expect = 0.35*(2.0*1000)/3600
check("T4 marginal/1k correct", abs(l.marginal_usd_per_1000_pages-expect)<1e-3, (l.marginal_usd_per_1000_pages, expect))
check("training cost 3h@0.35=$1.05", abs(l.training_usd-1.05)<1e-9, l.training_usd)

# on-device -> zero marginal
od = cm.local_cost(pages=1000, seconds_per_page=2.0, deployment="on-device", training_gpu_hours=3.0)
check("on-device marginal == 0", od.marginal_usd_per_1000_pages == 0.0, od)

# breakeven: api $0.0014/pg, local $0.000194/pg, training $1.05
be = cm.breakeven_pages(0.0014, 0.000194, 1.05)
check("breakeven positive & finite", be is not None and be > 0, be)
print(f"      breakeven = {be:,.0f} pages")

# breakeven when local is MORE expensive -> None (API always wins)
be2 = cm.breakeven_pages(0.0001, 0.002, 1.05)
check("no breakeven when local costlier", be2 is None, be2)

# full compare, realistic-ish shape
r = cm.compare(pages=15, api_input_tokens=30_000, api_output_tokens=10_500,
               local_seconds_per_page=3.0, deployment="T4", training_gpu_hours=3.0)
check("compare has projections", len(r["projections"])==4)
check("compare emits caveats", len(r["caveats"])>=4)
print(f"      api ${r['api_usd_per_page']:.6f}/pg | local ${r['local_marginal_usd_per_page']:.6f}/pg | breakeven {r['breakeven_pages']}")
for v,p in r["projections"].items():
    print(f"        {v:>16}: api ${p['api_usd']:>10,.2f}  local ${p['local_usd']:>10,.2f}  -> {p['cheaper']}")

print("\n"+("ALL COST TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
