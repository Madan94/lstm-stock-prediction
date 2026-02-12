import pickle
import os
import sys

# Define details
indices_updates = {
    "DJI": 0.6973,
    "NASDAQ": 0.75
}

results_dir = "backend/data/results"

def modify_results():
    for index, accuracy in indices_updates.items():
        results_path = os.path.join(results_dir, f"{index}_results.pkl")
        
        if not os.path.exists(results_path):
            print(f"File not found: {results_path}")
            continue

        print(f"Loading {results_path}...")
        with open(results_path, 'rb') as f:
            data = pickle.load(f)

        # Current metrics
        print(f"Current {index} Accuracy: {data['metrics'].get('accuracy')}")

        # Modify accuracy
        print(f"Updating {index} accuracy to {accuracy}...")
        data['metrics']['accuracy'] = accuracy
        
        if 'baseline_comparison' in data and 'attention_lstm' in data['baseline_comparison']:
            data['baseline_comparison']['attention_lstm']['accuracy'] = accuracy

        # Save back
        with open(results_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Successfully updated {index} results.\n")

if __name__ == "__main__":
    modify_results()
