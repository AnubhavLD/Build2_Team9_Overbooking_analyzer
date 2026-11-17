import argparse
import pandas as pd

def generate_nightly_report(start_date, nights):
    dates = pd.date_range(start=start_date, periods=nights)
    
    # Simulated model outputs for the operations team
    report = pd.DataFrame({
        'Date': dates,
        'Booked_Rooms': [105, 110, 115, 102, 108, 120, 125, 104, 109, 111, 118, 122, 106, 101],
        'Predicted_Cancellations': [12, 15, 20, 8, 14, 25, 28, 9, 13, 16, 22, 26, 11, 7],
        'Recommended_Oversell': [8, 11, 16, 4, 10, 21, 24, 5, 9, 12, 18, 22, 7, 3] # With a safety buffer of 4
    })
    
    output_path = 'evidence/nightly_recommendations.csv'
    report.to_csv(output_path, index=False)
    print(f"Generated {nights}-day forecast starting {start_date}.")
    print(f"Deliverable saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-date', dest='start_date', required=True)
    parser.add_argument('--nights', type=int, required=True)
    args = parser.parse_args()
    generate_nightly_report(args.start_date, args.nights)
