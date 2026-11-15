"""Select on Train/Choose, or evaluate the frozen bundle on Proof."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, roc_auc_score

if __package__:
    from . import clean_data, contract, data, features, modeling
else:
    import clean_data
    import contract
    import data
    import features
    import modeling

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "selected.joblib"
EVIDENCE = ROOT / "evidence"
INPUT = clean_data.CLEAN_FILE
BIN1_INPUT = ROOT / "data" / "cleaned_hotels_bin1.csv"
BIN1_MODEL = ROOT / "models" / "bin1_provisional.joblib"
BIN1_RESULTS = ROOT / "results"
PROOF_END = pd.Timestamp("2017-08-31")
DATE_PARTS = ("arrival_date_year", "arrival_date_month", "arrival_date_day_of_month")


def read_bookings(path, usecols=None):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=usecols, float_precision="round_trip", low_memory=False)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, usecols=usecols)
    raise ValueError("Input must be a CSV, XLSX or XLS file")


def prepare_dates(rows):
    if "arrival_date" not in rows:
        if not set(DATE_PARTS).issubset(rows.columns):
            raise ValueError("Provide arrival_date or its three original components")
        rows = rows.assign(arrival_date=data.arrival_date(rows))
    else:
        rows = rows.assign(arrival_date=pd.to_datetime(rows["arrival_date"], format="%Y-%m-%d"))
        if set(DATE_PARTS).issubset(rows.columns):
            if not rows["arrival_date"].eq(data.arrival_date(rows)).all():
                raise ValueError("Arrival dates disagree with original components")
    dates = rows["arrival_date"]
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Arrival dates must be valid calendar days")
    return rows


def load_scoring(path, provisional_bin1=False):
    raw = read_bookings(path)
    rows = prepare_dates(raw)
    components = {"stays_in_weekend_nights", "stays_in_week_nights"}
    if components.issubset(rows.columns):
        nights = rows["stays_in_weekend_nights"] + rows["stays_in_week_nights"]
        if "nights" in rows and not rows["nights"].eq(nights).all():
            raise ValueError("Stay lengths disagree with original components")
        rows = rows.assign(nights=nights)
    clean_data.validate_scoring(rows)
    features.predictors(rows)
    # Verify complete history and each scoring value before a model can be selected.
    source = clean_data.read_source(clean_data.RAW_FILE)
    expected = clean_data.build_master(source)
    if provisional_bin1:
        # This experiment accepts only the identified nonpositive-ADR exclusion.
        # Verify row order and all fields the models and contract actually read.
        expected = expected.loc[expected["adr"].gt(0)].reset_index(drop=True)
    if "source_row_id" in rows:
        if rows["source_row_id"].duplicated().any():
            raise ValueError("Duplicate source row IDs")
        rows = rows.sort_values("source_row_id")
        if not np.array_equal(rows["source_row_id"], expected["source_row_id"]):
            raise ValueError("Cleaned input has missing or unexpected source rows")
    checked = list(dict.fromkeys([*contract.REQUIRED, *features.FEATURES]))
    try:
        pd.testing.assert_frame_equal(
            rows[checked].reset_index(drop=True), expected[checked],
            check_dtype=False, check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError("Scoring and feature values must reconcile to the original course rows. "
                         "Without source_row_id, the original order is required.") from exc
    return rows


def diagnostics(bookings, probabilities, nights):
    target = bookings["is_canceled"].astype(int)
    return {
        "auc": roc_auc_score(target, probabilities) if target.nunique() == 2 else np.nan,
        "brier": brier_score_loss(target, probabilities),
        "nightly_mae": float((nights["expected"] - nights["cancelled"]).abs().mean()),
    }


def score_buffer(nights, buffer):
    """Turn estimated cancellations into rooms, then score those rooms."""
    expected_cancellations = nights["expected"]
    oversell_rooms = modeling.recommend(expected_cancellations, buffer)
    result = contract.night_cost(nights, oversell_rooms)
    return {"buffer": buffer, **result}


def buffer_search(nights):
    """Find the cheapest buffer that meets the walking ceiling."""
    largest_estimate = nights["expected"].max()
    # Include a buffer large enough to allow zero oversells on every night.
    largest_buffer = max(30, int(np.ceil(largest_estimate)))
    results = []
    for buffer in range(largest_buffer + 1):
        results.append(score_buffer(nights, buffer))
    grid = pd.DataFrame(results)
    feasible = grid.loc[grid["ceiling_met"]]
    if feasible.empty:
        raise ValueError("No safety buffer satisfies the walking ceiling")
    best = feasible.sort_values(["cost_eur", "walked_guests", "buffer"]).iloc[0].to_dict()
    return grid, best


def comparison(bookings, methods, buffers=None):
    records, grids = [], []
    for method in modeling.METHODS:
        probabilities = modeling.predict(method, methods[method], bookings)
        nights = contract.nightly_table(bookings, probabilities)
        if buffers is None:
            grid, score = buffer_search(nights)
            grids.append(grid.assign(method=method))
        else:
            # Proof receives the frozen buffer and skips the search.
            score = score_buffer(nights, buffers[method])
        prediction_metrics = diagnostics(bookings, probabilities, nights)
        records.append({"method": method, **prediction_metrics, **score})
    table = pd.DataFrame(records)
    flat = table.loc[table.method.eq("flat"), "cost_eur"].iloc[0]
    table["saving_vs_flat_eur"] = (flat - table["cost_eur"]).round(2)
    return table, pd.concat(grids, ignore_index=True) if grids else None


def versions():
    return {"scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__}


def load_bundle(path):
    """Load only a trusted locally produced joblib bundle."""
    bundle = joblib.load(path)
    if tuple(bundle["features"]) != features.FEATURES:
        raise ValueError("Frozen feature allowlist differs from the implementation")
    if bundle["versions"] != versions():
        raise ValueError(f"Use the frozen package versions {bundle['versions']}")
    return bundle


def save_bundle(bundle, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def choose(train, selection, source_hash, model_path, output, provisional_bin1=False):
    methods = modeling.fit_methods(train)
    table, grid = comparison(selection, methods)
    eligible = table[table["ceiling_met"]]
    if eligible.empty:
        raise ValueError("No method meets the walking ceiling")
    # Stable model order breaks exact economic ties in favour of simpler methods.
    selected = eligible.sort_values(["cost_eur", "walked_guests"], kind="stable").iloc[0]["method"]
    buffers = {row.method: int(row.buffer) for row in table.itertuples()}
    bundle = {
        "selected": selected, "buffers": buffers, "features": list(features.FEATURES),
        "data_sha256": source_hash, "seed": modeling.SEED, "versions": versions(),
        "train_rows": len(train), "choose_rows": len(selection), "models": methods,
        "train_end_exclusive": contract.TRAIN_END, "choose_end_exclusive": contract.PROOF_START,
        "contract_sha256": clean_data.fingerprint(Path(contract.__file__)),
        "implementation_sha256": implementation_hashes(),
        "lookup_prior_bookings": modeling.POOL,
        "policy": "max(0, floor(expected - buffer))",
        "choose_results": table.to_dict(orient="records"),
    }
    if provisional_bin1:
        bundle["provisional"] = True
        bundle["data_file"] = BIN1_INPUT.name
        bundle["exclusion_rule"] = "Source ADR <= 0 excluded by Data Lead export"
        bundle["selection_rule"] = "Lowest Choose euro loss under the pooled walk ceiling"
        bundle["proof_rows_used_for_selection"] = 0
    save_bundle(bundle, model_path)
    output.mkdir(parents=True, exist_ok=True)
    table["selected"] = table["method"].eq(selected)
    comparison_name = "choose_model_comparison.csv" if provisional_bin1 else "choose_comparison.csv"
    table.to_csv(output / comparison_name, index=False)
    if provisional_bin1:
        nightly_frames = []
        for method in modeling.METHODS:
            probabilities = modeling.predict(method, methods[method], selection)
            nights = contract.nightly_table(selection, probabilities)
            nights["oversell"] = modeling.recommend(nights["expected"], buffers[method])
            nightly_frames.append(nights.reset_index().assign(method=method))
        pd.concat(nightly_frames, ignore_index=True).to_csv(
            output / "choose_nightly_predictions.csv", index=False,
        )
    grid.to_csv(output / "choose_buffer_grid.csv", index=False)
    metadata = {key: value for key, value in bundle.items() if key != "models"}
    metadata["artifact_sha256"] = clean_data.fingerprint(model_path)
    (output / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Frozen selection: {selected}, buffer {buffers[selected]}")
    return table


def implementation_hashes():
    return {module.__name__.split(".")[-1]: clean_data.fingerprint(Path(module.__file__))
            for module in (features, modeling)}


def final_proof(proof, source_hash, model_path, output):
    bundle = load_bundle(model_path)
    if bundle.get("provisional"):
        raise ValueError("Provisional Bin 1 model is ineligible for final Proof")
    if source_hash != bundle["data_sha256"]:
        raise ValueError("Proof input differs from the frozen selection input")
    if clean_data.fingerprint(Path(contract.__file__)) != bundle["contract_sha256"]:
        raise ValueError("Scoring contract differs from the frozen contract")
    if implementation_hashes() != bundle["implementation_sha256"]:
        raise ValueError("Predictor or decision code differs from the frozen implementation")
    if proof.empty or proof["arrival_date"].max() > PROOF_END:
        raise ValueError("Proof data is missing or extends beyond August 2017")
    artifact_hash = clean_data.fingerprint(model_path)
    report = output / "proof_comparison.csv"
    if report.exists():
        raise FileExistsError(f"Proof already recorded at {report}")
    table, _ = comparison(proof, bundle["models"], bundle["buffers"])
    table["frozen_selection"] = table.method.eq(bundle["selected"])
    table["artifact_sha256"] = artifact_hash
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement or repeat scoring in this output directory.
    with report.open("x", encoding="utf-8", newline="") as file:
        table.to_csv(file, index=False)
    print(table.to_string(index=False))
    selected = table.loc[table["frozen_selection"]].iloc[0]
    status = "met" if selected.ceiling_met else "FAILED"
    print(f"Frozen selection: {bundle['selected']}. Proof walking ceiling {status}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--choose-only", action="store_true", help="Fit, compare and freeze (default)")
    mode.add_argument("--final-proof", action="store_true", help="Score the frozen artifact once")
    parser.add_argument("--provisional-bin1", action="store_true", help="Run the Data Lead experiment")
    parser.add_argument("--cleaned", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.provisional_bin1 and args.final_proof:
        parser.error("The provisional Bin 1 dataset is ineligible for final Proof")
    args.cleaned = args.cleaned or (BIN1_INPUT if args.provisional_bin1 else INPUT)
    args.model = args.model or (BIN1_MODEL if args.provisional_bin1 else MODEL)
    args.evidence = args.evidence or (BIN1_RESULTS if args.provisional_bin1 else EVIDENCE)
    if args.provisional_bin1 and (
        args.model.resolve() == MODEL.resolve() or args.evidence.resolve() == EVIDENCE.resolve()
    ):
        parser.error("Keep provisional results separate from the earlier model and evidence")
    proof_report = args.evidence / "proof_comparison.csv"
    if proof_report.exists():
        parser.error(f"Proof output already exists at {proof_report}. Preserve the original run.")
    protected = (args.cleaned, clean_data.RAW_FILE, clean_data.LEGACY_FILE, INPUT)
    if any(args.model.resolve() == path.resolve() or (
        args.model.exists() and path.exists() and args.model.samefile(path)
    ) for path in protected):
        parser.error("The model artifact must not replace a booking input")
    try:
        source_hash = clean_data.fingerprint(args.cleaned)
        rows = load_scoring(args.cleaned, provisional_bin1=args.provisional_bin1)
        if source_hash != clean_data.fingerprint(args.cleaned):
            raise ValueError("Cleaned input changed while it was being read")
        train, selection, proof = contract.split_periods(rows)
        if args.final_proof:
            final_proof(proof, source_hash, args.model, args.evidence)
        else:
            choose(train, selection, source_hash, args.model, args.evidence, args.provisional_bin1)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

