import json
import os
import argparse
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the project's own Dataloader that handles HF datasets correctly
from chinatravel.data.load_datasets import load_query

def main():
    parser = argparse.ArgumentParser(description="Map query IDs from a split file to their natural language queries.")
    parser.add_argument("--splits", type=str, default="human", help="Base split to load query data from (e.g. human)")
    parser.add_argument("--target_split", type=str, default="human_hard", help="Name of the split file to map (e.g. human_hard)")
    parser.add_argument("--oracle_translation", action="store_true", default=True)
    args = parser.parse_args()

    # Load ALL "human" queries into memory using the project's huggingface data loader
    print("Loading base queries from HuggingFace dataset...")
    base_query_ids, query_data = load_query(args)

    split_file_path = os.path.join(project_root, "chinatravel", "evaluation", "default_splits", f"{args.target_split}.txt")
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(project_root, "query_map")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, f"{args.target_split}_mapped.txt")

    if not os.path.exists(split_file_path):
        print(f"Error: Split file not found at {split_file_path}")
        return

    with open(split_file_path, 'r', encoding='utf-8') as f:
        target_query_ids = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(target_query_ids)} queries in {args.target_split}.txt")
    print(f"Mapping queries and saving to {output_file_path}...")

    not_found_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        for idx, qid in enumerate(target_query_ids):
            out_f.write(f"[{idx+1}/{len(target_query_ids)}] UID: {qid}\n")
            if qid in query_data:
                nl_query = query_data[qid].get("nature_language", "No natural language query field found.")
                out_f.write(f"Query: {nl_query}\n")
            else:
                out_f.write("Query: Query not found in base dataset.\n")
                not_found_count += 1
            out_f.write("-" * 50 + "\n")

    print(f"Done! Successfully mapped {len(target_query_ids) - not_found_count} queries.")
    if not_found_count > 0:
        print(f"Warning: {not_found_count} queries were not found in the base dataset.")

if __name__ == "__main__":
    main()
