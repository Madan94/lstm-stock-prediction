import pickle
import os
import numpy as np

results_dir = "backend/data/results"
indices = ["SP500", "DJI", "NASDAQ"]

print("Verifying stored prediction accuracy:")

for index in indices:
    try:
        results_path = os.path.join(results_dir, f"{index}_results.pkl")
        if not os.path.exists(results_path):
            print(f"{index}: File not found")
            continue
            
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
            
        if 'predictions' in data:
            preds = data['predictions']
            actuals = np.array(preds['actuals'])
            predictions = np.array(preds['predictions'])
            
            correct = (actuals == predictions)
            accuracy = np.mean(correct)
            
            # Check last 100
            last_100_correct = correct[-100:]
            last_100_accuracy = np.mean(last_100_correct)
            
            print(f"{index}: Overall={accuracy*100:.1f}%, Last100={last_100_accuracy*100:.1f}%")
            
            # Assertions
            expected = {
                "SP500": 75.0, # 75.46 rounds to 75 count
                "DJI": 70.0,   # 69.73 rounds to 70 count
                "NASDAQ": 75.0
            }
            
            if index in expected:
                acc = last_100_accuracy * 100
                target = expected[index]
                if abs(acc - target) > 1.0:
                    print(f"  FAIL: Expected {target}%, got {acc}%")
                else:
                    print(f"  PASS")
        else:
            print(f"{index}: predictions key missing")
            
    except Exception as e:
        print(f"{index}: Error - {e}")
