"""Exercise source preservation, derived columns and the CSV round trip."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clean_data


@pytest.fixture
def raw():
    return pd.DataFrame({
        "hotel": ["City Hotel", "Resort Hotel", "Resort Hotel", "City Hotel"],
        "is_canceled": [0, 1, 1, 0],
        "arrival_date_year": [2016, 2016, 2016, 2017],
        "arrival_date_month": ["June", "July", "July", "January"],
        "arrival_date_day_of_month": [30, 1, 1, 1],
        "stays_in_weekend_nights": [0, 1, 1, 0],
        "stays_in_week_nights": [0, 2, 2, 1],
        "adults": [0, 2, 2, 0],
        "children": [0.0, 0.0, 0.0, float("nan")],
        "babies": [0, 0, 0, 0],
        "adr": [0.0, 100.5, 100.5, -6.38],
        "country": [None, "PRT", "PRT", None],
        "agent": [None, 12, 12, None],
        "company": [None] * 4,
        "reservation_status": ["Canceled", "Check-Out", "Check-Out", "Check-Out"],
    })


def test_preserves_columns_missingness_duplicates_and_outcomes(raw):
    master = clean_data.build_master(raw)
    pd.testing.assert_frame_equal(raw, master[raw.columns])
    assert master.source_row_id.tolist() == [1, 2, 3, 4]
    assert master[raw.columns].duplicated().sum() == 1
    assert master.groupby("is_canceled").size().to_dict() == {0: 2, 1: 2}


def test_dates_nights_flags_and_periods(raw):
    master = clean_data.build_master(raw)
    assert master.arrival_date.dt.strftime("%Y-%m-%d").tolist() == [
        "2016-06-30", "2016-07-01", "2016-07-01", "2017-01-01",
    ]
    assert master.nights.tolist() == [0, 3, 3, 1]
    assert master.flag_zero_nights.tolist() == [True, False, False, False]
    assert master.flag_zero_adr.tolist() == [True, False, False, False]
    assert master.flag_zero_guests.tolist() == [True, False, False, pd.NA]
    assert master.flag_adr_review.tolist() == [False, False, False, True]
    counts = clean_data.reconciliation(master)
    assert counts.bookings.tolist() == [1, 2, 1]
    assert counts.canceled.tolist() == [0, 2, 0]


def test_rejects_loss_and_changed_outcomes(raw):
    master = clean_data.build_master(raw)
    with pytest.raises(ValueError, match="Unexplained deletions: 1"):
        clean_data.verify_master(raw, master.iloc[1:])
    master.loc[0, "is_canceled"] = 1
    with pytest.raises(AssertionError, match="is_canceled"):
        clean_data.verify_master(raw, master)
    master.loc[0, "is_canceled"] = 2
    with pytest.raises(ValueError, match="binary"):
        clean_data.verify_master(raw, master)


def test_rejects_invalid_and_changed_dates(raw):
    master = clean_data.build_master(raw)
    master.loc[0, "arrival_date"] += pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="arrival_date"):
        clean_data.verify_master(raw, master)
    raw.loc[0, "arrival_date_day_of_month"] = 31
    with pytest.raises(ValueError):
        clean_data.build_master(raw)


def test_output_is_reproducible_and_preserves_input(tmp_path, monkeypatch, raw):
    source, first, second = [tmp_path / name for name in ("raw.csv", "one.csv", "two.csv")]
    raw.fillna({"country": ""}).to_csv(source, index=False)
    monkeypatch.setattr(clean_data.data, "EXPECTED_ROWS", len(raw))
    monkeypatch.setattr(clean_data.data, "EXPECTED_COLUMNS", len(raw.columns))
    before = source.read_bytes()
    clean_data.write_master(source, first)
    clean_data.write_master(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        clean_data.write_master(source, source)

