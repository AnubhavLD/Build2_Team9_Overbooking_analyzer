"""Check hand-calculated results and a complete small Train/Choose run."""
import numpy as np
import pandas as pd
import pytest

from src import backtest, clean_data, contract, features, modeling, nightly


@pytest.mark.parametrize("oversell, expected_pass", [(1, True), (2, False)])
def test_walking_ceiling_accepts_exactly_ten_per_thousand(oversell, expected_pass):
    nights = pd.DataFrame({"book": [100], "cancelled": [0], "stay_value": [100.0]})
    score = contract.night_cost(nights, [oversell])
    assert score["walked_guests"] == oversell
    assert score["walks_per_1000"] == 10 * oversell
    assert score["cost_eur"] == 300 * oversell
    assert score["ceiling_met"] is expected_pass


@pytest.mark.parametrize("probability", [-0.01, float("nan"), float("inf")])
def test_invalid_probabilities_are_rejected(probability):
    row = pd.DataFrame({
        "hotel": ["City Hotel"], "arrival_date": pd.to_datetime(["2016-08-01"]),
        "is_canceled": [1], "adr": [100], "nights": [1],
    })
    with pytest.raises(ValueError, match="between 0 and 1"):
        contract.nightly_table(row, [probability])


@pytest.fixture
def raw_bookings():
    rows = []
    for year, month, city_outcomes in [
        (2016, "June", [1, 0, 1, 1]),
        (2016, "July", [0, 0, 0, 1]),
        (2017, "January", [1, 1, 1, 1]),
    ]:
        for hotel in features.HOTELS:
            outcomes = city_outcomes if hotel == "City Hotel" else [0, 0, 0, 1]
            for i, outcome in enumerate(outcomes):
                rows.append({
                    "hotel": hotel, "is_canceled": outcome, "arrival_date_year": year,
                    "arrival_date_month": month, "arrival_date_day_of_month": 1,
                    "stays_in_weekend_nights": 1, "stays_in_week_nights": 2,
                    "adults": 2, "children": 0, "babies": 0, "adr": 100,
                    "distribution_channel": ["Direct", "Direct", "TA/TO", None][i]
                    if month == "June" else "Future channel",
                    "is_repeated_guest": 0,
                    "previous_cancellations": [0, 2, None, 4][i] if month == "June" else 100,
                    "previous_bookings_not_canceled": i,
                    "reservation_status": "Canceled" if outcome else "Check-Out",
                })
    return pd.DataFrame(rows)


def test_flat_and_lookup_rates_follow_input_order(raw_bookings):
    train, choose, _ = contract.split_periods(clean_data.build_master(raw_bookings))
    models = modeling.fit_methods(train)
    # City Direct is 1/2, pooled toward the City rate of 3/4.
    probe = choose.iloc[[4, 0, 1]].assign(distribution_channel=["Unknown", "Direct", None])
    np.testing.assert_allclose(modeling.predict("flat", models["flat"], probe), [.25, .75, .75])
    np.testing.assert_allclose(
        modeling.predict("lookup", models["lookup"], probe),
        [.25, (1 + 50 * .75) / 52, .75],
    )


def test_clean_to_frozen_recommendation_uses_only_train_and_choose(raw_bookings, tmp_path):
    master = clean_data.build_master(raw_bookings)
    clean_data.verify_master(raw_bookings, master)
    train, choose, proof = contract.split_periods(master)
    assert [len(part) for part in (train, choose, proof)] == [8, 8, 8]
    path = tmp_path / "selected.joblib"
    table = backtest.choose(train, choose, "synthetic", path, tmp_path)
    bundle = backtest.load_bundle(path)
    chosen = table.loc[table["selected"]].iloc[0]
    assert chosen.ceiling_met
    assert chosen.cost_eur == table.loc[table.ceiling_met, "cost_eur"].min()
    assert bundle["selected"] == chosen.method
    assert bundle["train_rows"] == bundle["choose_rows"] == 8
    assert bundle["models"]["flat"].loc["City Hotel"] == .75
    assert not (tmp_path / "proof_comparison.csv").exists()

    for method in ("logistic", "forest"):
        model = bundle["models"][method]
        preparation = model.steps[0][1]
        numeric = preparation.named_transformers_["numeric"]
        category = preparation.named_transformers_["category"]
        # Missing Train counts are imputed at 2. Choose counts of 100 never affect the fit.
        assert numeric.named_steps["simpleimputer"].statistics_[1] == 2
        assert numeric.named_steps["standardscaler"].mean_[1] == 2
        channels = category.named_steps["onehotencoder"].categories_[1]
        assert set(channels) == {"Direct", "TA/TO", "__MISSING__"}
        inputs = features.predictors(choose)
        probabilities = modeling.predict(method, model, inputs)
        np.testing.assert_array_equal(probabilities, model.predict_proba(inputs)[:, 1])
        assert np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all()
        shuffled = inputs.iloc[::-1]
        np.testing.assert_array_equal(modeling.predict(method, model, shuffled), probabilities[::-1])

    method = bundle["selected"]
    probabilities = modeling.predict(method, bundle["models"][method], choose)
    scored_nights = contract.nightly_table(choose, probabilities)
    forecast = nightly.nightly_recommendations(
        choose.drop(columns="is_canceled"), "2016-07-01", 1, bundle,
    ).set_index(["hotel", "arrival_date"])
    np.testing.assert_allclose(forecast["expected_cancellations"], scored_nights["expected"])
    np.testing.assert_array_equal(
        forecast["recommended_oversell"], modeling.recommend(scored_nights["expected"], chosen.buffer),
    )

