import pickle
import os

results_path = "backend/data/results/NASDAQ_results.pkl"

if os.path.exists(results_path):
    print(f"Loading {results_path}...")
    try:
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Keys found:", data.keys())
        
        required_keys = ['metrics', 'predictions', 'equity_curve', 'attention_weights', 'baseline_comparison']
        for key in required_keys:
            if key in data:
                print(f"  {key}: Present")
                if key == 'equity_curve':
                    print(f"    - points: {len(data[key]['values'])}")
            else:
                print(f"  {key}: MISSING")
                
    except Exception as e:
        print(f"Error loading pickle: {e}")
else:
    print("File not found.")
