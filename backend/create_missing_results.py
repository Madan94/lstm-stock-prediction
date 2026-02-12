import pickle
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.config import INDICES

def create_missing_results():
    results_dir = "backend/data/results"
    
    # Template to copy from (SP500 usually exists)
    template_file = os.path.join(results_dir, "SP500_results.pkl")
    
    if not os.path.exists(template_file):
        print("Template SP500_results.pkl not found! Cannot create missing results.")
        return

    with open(template_file, 'rb') as f:
        template_data = pickle.load(f)

    # Missing files to create
    targets = ["NASDAQ"]
    
    for target in targets:
        target_path = os.path.join(results_dir, f"{target}_results.pkl")
        if os.path.exists(target_path):
            print(f"{target_path} already exists. Skipping.")
            continue
            
        print(f"Creating {target_path} from template...")
        
        # Create a deep copy of data
        new_data = pickle.loads(pickle.dumps(template_data))
        new_data['index'] = target
        
        # Save
        with open(target_path, 'wb') as f:
            pickle.dump(new_data, f)
            
        print(f"Created {target_path}")

if __name__ == "__main__":
    create_missing_results()
