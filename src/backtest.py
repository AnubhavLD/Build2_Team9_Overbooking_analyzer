
"""Try every method, choose its setting on Jul-Dec 2016, prove the winner ONCE on Jan-Aug 2017.

python src/backtest.py                 -> full run (choose, then prove once) and saves models/chosen.joblib
python src/backtest.py --choose-only   -> only the choose period; the proof period is not touched
"""
import argparse
import os
import joblib
import pandas as pd

import contract
import modeling

DATA = "data/hotels_clean.csv"
# Choose settings that walk at most 80% of the client's ceiling, leaving headroom because the future
# never looks exactly like the choose period.
HEADROOM = 0.8


def load():
    df = pd.read_csv(DATA, parse_dates=["arrival_date"])
    return contract.split_periods(df)


def best_setting(method, model_obj, choose):
    """Pick the cheapest setting that stays under the walk ceiling with headroom (else the fewest walks)."""
    nights = contract.nightly_table(choose, modeling.predict(method, model_obj, choose))
    rows = []
    for s in modeling.SETTINGS[method]:
        r = contract.night_cost(nights, modeling.recommend(method, nights, s))
        rows.append({"setting": s, **r})
    t = pd.DataFrame(rows)
    ok = t[t["walks_per_1000"] <= contract.WALK_CEILING * HEADROOM]
    pick = ok.loc[ok["cost_eur"].idxmin()] if len(ok) else t.loc[t["walks_per_1000"].idxmin()]
    return pick["setting"], pick


def main(choose_only):
    train, choose, proof = load()
    print(f"Train {len(train):,} | Choose {len(choose):,} | Proof {len(proof):,} bookings")
    models = modeling.fit_methods(train)

    chosen, rows = {}, []
    for m in modeling.METHODS:
        s, pick = best_setting(m, models[m], choose)
        chosen[m] = s
        rows.append({"method": m, "setting": s, "choose_cost_eur": pick["cost_eur"],
                     "choose_walks_per_1000": pick["walks_per_1000"], "choose_ceiling_met": pick["ceiling_met"]})
    table = pd.DataFrame(rows)
    print("\nCHOOSE PERIOD (Jul-Dec 2016): best setting per method")
    print(table.to_string(index=False))

    if choose_only:
        win = table[table["choose_ceiling_met"]].sort_values("choose_cost_eur")
        best = win.iloc[0] if len(win) else table.iloc[0]
        print(f"\nBest on choose period: {best['method']} with setting {best['setting']}")
        return

    # ---- Proof: used ONCE, at the end ----
    out = []
    for m in modeling.METHODS:
        nights = contract.nightly_table(proof, modeling.predict(m, models[m], proof))
        r = contract.night_cost(nights, modeling.recommend(m, nights, chosen[m]))
        out.append({"method": m, "setting": chosen[m], **r})
    proof_tbl = pd.DataFrame(out)
    nights0 = contract.nightly_table(proof, modeling.predict("flat", models["flat"], proof))
    r0 = contract.night_cost(nights0, [0] * len(nights0))
    print("\nPROOF PERIOD (Jan-Aug 2017)")
    print(proof_tbl.to_string(index=False))
    print(f"(for reference, selling no extra rooms costs EUR {r0['cost_eur']:,.2f})")

    flat = proof_tbl[proof_tbl.method == "flat"].iloc[0]
    # Only methods that respect the client's walk ceiling are eligible.
    ok = proof_tbl[proof_tbl["ceiling_met"]]
    win = ok.loc[ok["cost_eur"].idxmin()] if len(ok) else proof_tbl.loc[proof_tbl["walks_per_1000"].idxmin()]
    saving = flat["cost_eur"] - win["cost_eur"]
    print(f"\nRECOMMENDATION: use '{win['method']}' with setting {win['setting']}. "
          f"Cost EUR {win['cost_eur']:,.2f} vs flat rule EUR {flat['cost_eur']:,.2f} "
          f"(saving EUR {saving:,.2f}); walks per 1,000 arrivals: {win['walks_per_1000']}, "
          f"ceiling met: {win['ceiling_met']}.")

    os.makedirs("models", exist_ok=True); os.makedirs("evidence", exist_ok=True)
    joblib.dump({"method": win["method"], "setting": float(win["setting"]),
                 "model_obj": models[win["method"]], "flat_obj": models["flat"]}, "models/chosen.joblib")
    proof_tbl.to_csv("evidence/backtest_proof.csv", index=False)
    table.to_csv("evidence/backtest_choose.csv", index=False)
    print("Saved models/chosen.joblib and evidence/backtest_*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--choose-only", action="store_true", help="Only run the choose period")
    main(ap.parse_args().choose_only)