import sys
import os
import argparse

# Setup path to import project modules
project_root_path = os.getcwd()
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.data.load_datasets import load_query

def find_queries():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="human")
    parser.add_argument("--oracle_translation", action="store_true", default=True)
    args = parser.parse_args([])
    
    query_id_list, query_data = load_query(args)
    
    # Target query with 9 constraints found earlier: h20241029143438341517
    target_uid = "h20241029143438341517"
    
    with open("logic_result.txt", "w") as f:
        found = False
        if target_uid in query_data:
            data = query_data[target_uid]
            f.write(f"UID: {target_uid}\n")
            f.write(f"Target City: {data.get('target_city')}\n")
            logic = data.get("hard_logic_py", [])
            f.write(f"Total logic constraints: {len(logic)}\n")
            for i, l in enumerate(logic):
                f.write(f"--- logic_{i} ---\n")
                f.write(f"{l}\n")
            found = True
        
        if not found:
            for uid in query_id_list:
                data = query_data[uid]
                logic = data.get("hard_logic_py", [])
                if len(logic) == 9:
                    f.write(f"UID: {uid}\n")
                    f.write(f"Target City: {data.get('target_city')}\n")
                    f.write(f"Total logic constraints: {len(logic)}\n")
                    for i, l in enumerate(logic):
                        f.write(f"--- logic_{i} ---\n")
                        f.write(f"{l}\n")
                    found = True
                    break
        
        if found:
            print("Written to logic_result.txt")
        else:
            print("No 9-constraint queries found.")

if __name__ == "__main__":
    find_queries()
