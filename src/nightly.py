"""Morning table: rooms to oversell for each hotel on each of the next N nights.
python src/nightly.py --from 2017-08-18 --nights 14"""
import argparse
import joblib
import pandas as pd

import contract
import modeling

DATA = "data/hotels_clean.csv"


def generate_nightly_report(start_date, nights):
    saved = joblib.load("models/chosen.joblib")
    method, setting, model_obj = saved["method"], saved["setting"], saved["model_obj"]

    df = pd.read_csv(DATA, parse_dates=["arrival_date"])
    dates = pd.date_range(start=start_date, periods=nights)
    book = df[df["arrival_date"].isin(dates)]
    if book.empty:
        raise SystemExit(f"No bookings found for {dates[0].date()} to {dates[-1].date()} in {DATA}.")

    t = contract.nightly_table(book, modeling.predict(method, model_obj, book))
    t["oversell_rooms"] = modeling.recommend(method, t, setting)
    report = t.reset_index().rename(columns={
        "arrival_date": "Date", "hotel": "Hotel", "book": "Booked_Rooms", "expected": "Predicted_Cancellations"})
    report["Predicted_Cancellations"] = report["Predicted_Cancellations"].round(1)
    report = report[["Date", "Hotel", "Booked_Rooms", "Predicted_Cancellations", "oversell_rooms"]]
    report = report.rename(columns={"oversell_rooms": "Recommended_Oversell"}).sort_values(["Date", "Hotel"])

    report.to_csv("evidence/nightly_recommendations.csv", index=False)
    print(f"Generated {nights}-night table from {start_date} using '{method}' (setting {setting}).")
    print("Deliverable saved to evidence/nightly_recommendations.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", "--from-date", dest="start_date", required=True)
    ap.add_argument("--nights", type=int, required=True)
    a = ap.parse_args()
    generate_nightly_report(a.start_date, a.nights)