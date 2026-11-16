import pandas as pd
import numpy as np
import os

def clean_data(input_path='data/hotels.csv', output_path='data/hotels_clean.csv'):
    # If raw data is missing, generate synthetic data so the pipeline runs out-of-the-box
    if not os.path.exists(input_path):
        print(f"Raw data not found at {input_path}. Generating synthetic hotel data for pipeline testing...")
        np.random.seed(42)
        dates = pd.date_range(start='2017-01-01', end='2017-12-31')
        df = pd.DataFrame({
            'arrival_date': np.random.choice(dates, 5000),
            'lead_time': np.random.randint(1, 300, 5000),
            'adr': np.random.normal(100, 30, 5000).round(2),
            'market_segment': np.random.choice(['Online TA', 'Offline TA', 'Corporate', 'Direct'], 5000),
            'deposit_type': np.random.choice(['No Deposit', 'Non Refund', 'Refundable'], 5000, p=[0.8, 0.15, 0.05]),
            'is_canceled': np.random.choice([0, 1], 5000, p=[0.65, 0.35]) # 35% cancellation rate
        })
        df.to_csv(input_path, index=False)

    # Load, clean, and save
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().dropna()
    df.to_csv(output_path, index=False)
    print(f"Data pipeline complete: Cleaned data saved to {output_path}")

if __name__ == '__main__':
    clean_data()
