import sys
import os
import argparse

# Setup path to import project modules
project_root_path = os.getcwd()
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.data.load_datasets import load_query

def extract_hard_queries():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="human")
    parser.add_argument("--oracle_translation", action="store_true", default=True)
    args = parser.parse_args([])
    
    print(f"Loading {args.splits} split...")
    query_id_list, query_data = load_query(args)
    
    hard_uids = []
    print("\n--- Identifying queries with >= 6 constraints ---")
    for uid in query_id_list:
        data = query_data[uid]
        logic = data.get("hard_logic_py", [])
        if len(logic) >= 6:
            hard_uids.append(uid)
    
    print(f"Found {len(hard_uids)} such queries.")
    
    output_dir = os.path.join(project_root_path, "chinatravel", "evaluation", "default_splits")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "human_hard.txt")
    
    with open(output_path, "w") as f:
        for uid in hard_uids:
            f.write(f"{uid}\n")
    
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    extract_hard_queries()
