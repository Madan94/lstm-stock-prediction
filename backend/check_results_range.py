import pickle
import os

results_path = "backend/data/results/SP500_results.pkl"

if os.path.exists(results_path):
    print(f"Loading {results_path}...")
    with open(results_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'equity_curve' in data and 'dates' in data['equity_curve']:
        dates = data['equity_curve']['dates']
        print(f"Equity Curve Range: {dates[0]} to {dates[-1]}")
        print(f"Total points: {len(dates)}")
    else:
        print("No equity curve data found.")
else:
    print("Results file not found.")
