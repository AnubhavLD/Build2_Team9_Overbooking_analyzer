"""Write target-free hotel-night recommendations for a supplied bookings snapshot."""
import argparse
from pathlib import Path

import pandas as pd

if __package__:
    from . import backtest, features, modeling
else:
    import backtest
    import features
    import modeling

ROUTING = {"arrival_date", *backtest.DATE_PARTS}
INPUT_COLUMNS = ROUTING | set(features.FEATURES)


def nightly_recommendations(rows, start, number_of_nights, bundle):
    if number_of_nights < 1:
        raise ValueError("Number of nights must be positive")
    days = pd.date_range(start, periods=number_of_nights)
    if days.tz is not None or days[0] != days[0].normalize():
        raise ValueError("Start must be a calendar date")
    # Strip outcome fields even when passed an in-memory scoring frame.
    rows = backtest.prepare_dates(rows.loc[:, rows.columns.isin(INPUT_COLUMNS)])
    window = rows.loc[rows.arrival_date.between(days[0], days[-1])]
    method = bundle["selected"]
    buffer = bundle["buffers"][method]
    probabilities = modeling.predict(method, bundle["models"][method], window)
    totals = (
        window.assign(probability=probabilities)
        .groupby(["hotel", "arrival_date"])
        .agg(bookings=("probability", "size"), expected_cancellations=("probability", "sum"))
    )
    full_window = pd.MultiIndex.from_product(
        [features.HOTELS, days], names=["hotel", "arrival_date"],
    )
    totals = totals.reindex(full_window, fill_value=0)
    totals["safety_buffer"] = buffer
    totals["recommended_oversell"] = modeling.recommend(totals["expected_cancellations"], buffer)
    return totals.reset_index().sort_values(["arrival_date", "hotel"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--nights", type=int, default=14)
    parser.add_argument("--bookings", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model is None:
        args.model = backtest.BIN1_MODEL if backtest.BIN1_MODEL.exists() else backtest.MODEL
    try:
        date = pd.to_datetime(args.start, format="%Y-%m-%d")
        bundle = backtest.load_bundle(args.model)
        provisional = bundle.get("provisional", False)
        default_input = backtest.BIN1_INPUT if provisional else backtest.INPUT
        default_output = backtest.BIN1_RESULTS if provisional else backtest.EVIDENCE
        args.bookings = args.bookings or default_input
        args.output = args.output or (default_output / "nightly_recommendations.csv")
        if provisional:
            print("Provisional Bin 1 demo using the supplied booking list")
        if backtest.implementation_hashes() != bundle["implementation_sha256"]:
            raise ValueError("Predictor or decision code differs from the frozen implementation")
        bookings = backtest.read_bookings(args.bookings, usecols=INPUT_COLUMNS.__contains__)
        table = nightly_recommendations(bookings, date, args.nights, bundle)
        if args.output.resolve() in {args.bookings.resolve(), args.model.resolve()}:
            raise ValueError("Output must differ from the bookings and model files")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False, date_format="%Y-%m-%d", float_format="%.10f")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(table)} hotel-night recommendations to {args.output}")


if __name__ == "__main__":
    main()

