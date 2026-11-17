"""Features = ONLY columns the hotel knows on the day of booking (bin 1 of docs/COLUMNS.md).

Left out on purpose:
  bin 3 (leaks the answer): is_canceled, reservation_status, reservation_status_date,
                            assigned_room_type, booking_changes, required_car_parking_spaces
  bin 2 (can change before arrival): stays_*, adults, children, babies, meal, country, adr,
                            total_of_special_requests
  arrival_date_year: the proof period (2017) never appears in training, so "year" teaches the model
                     nothing useful about it. Month / week / weekday are used instead.
"""
import pandas as pd

CATEGORICAL = (
    "hotel",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "arrival_date_month",
)

NUMERIC = (
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "arrival_weekday",              # engineered below
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "has_agent",                    # blank agent = none
    "has_company",                  # blank company = none
)

FEATURES = CATEGORICAL + NUMERIC

HOTELS = ["Resort Hotel", "City Hotel"]


def predictors(rows):
    """Return a table holding exactly the FEATURES, built only from booking-day information.
    Works on the cleaned file and also on the raw export (rebuilds missing helper columns)."""
    df = rows.copy()

    if "arrival_date" not in df.columns:
        df["arrival_date"] = pd.to_datetime(
            df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"] + "-"
            + df["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["arrival_weekday"] = df["arrival_date"].dt.dayofweek

    if "has_agent" not in df.columns:
        df["has_agent"] = df["agent"].notna().astype(int)
    if "has_company" not in df.columns:
        df["has_company"] = df["company"].notna().astype(int)

    return df[list(FEATURES)]


def encode(rows, columns=None):
    """One-hot encode the categorical features. Pass the training columns so other data lines up."""
    X = pd.get_dummies(predictors(rows), columns=list(CATEGORICAL), dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X