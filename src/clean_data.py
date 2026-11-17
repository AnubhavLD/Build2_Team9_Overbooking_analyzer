"""Clean data/hotels.csv -> data/hotels_clean.csv, removing as few rows as possible."""
import os
import pandas as pd

MAX_ADR = 1000.0   # one booking has adr = 5,400 (data-entry error); every other value is <= 510


def _arrival_date(df):
    return pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"] + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )


def clean_data(input_path="data/hotels.csv", output_path="data/hotels_clean.csv"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Upload hotels.csv to the data/ folder.")

    df = pd.read_csv(input_path, low_memory=False)
    n0 = len(df)
    log = []

    # 1. Blanks are filled, never dropped (0 rows removed)
    df["children"] = df["children"].fillna(0)                 # 4 blanks -> no children
    df["country"] = df["country"].fillna("Unknown")           # 488 blanks
    df["has_agent"] = df["agent"].notna().astype(int)         # blank agent = booked without one
    df["has_company"] = df["company"].notna().astype(int)     # blank company = not a corporate booking
    df["agent"] = df["agent"].fillna(0).astype(int)
    df["company"] = df["company"].fillna(0).astype(int)

    # 2. `babies` contains one "??" cell; make the column numeric, bad cell -> 0
    df["babies"] = pd.to_numeric(df["babies"], errors="coerce")
    log.append(f"babies unreadable ('??') set to 0: {int(df['babies'].isna().sum())} cell(s)")
    df["babies"] = df["babies"].fillna(0)
    for c in ["children", "babies"]:
        df[c] = df[c].astype(int)

    # 3. Columns the scoring contract (contract.py) needs
    df["arrival_date"] = _arrival_date(df)
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    # 4. Remove ONLY impossible rows (a money value that cannot be real)
    bad_adr = (df["adr"] < 0) | (df["adr"] > MAX_ADR)
    log.append(f"rows removed for impossible adr (<0 or >{MAX_ADR:.0f}): {int(bad_adr.sum())}")
    df = df[~bad_adr].copy()

    # Kept on purpose: adr = 0 stays, 0 adults, 0 nights, 'Undefined' labels. These bookings still
    # arrive or cancel on a night, so they count in the contract's scoring.
    df.to_csv(output_path, index=False)

    print(f"Rows in: {n0:,} | rows out: {len(df):,} | removed: {n0 - len(df):,} ({(n0 - len(df)) / n0:.3%})")
    for line in log:
        print(" -", line)
    print(f"Cleaned data saved to {output_path}")
    return df


if __name__ == "__main__":
    clean_data()