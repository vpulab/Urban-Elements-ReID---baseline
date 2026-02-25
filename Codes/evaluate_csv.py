import os
import numpy as np
import pandas as pd
import argparse

def read_csv_gt(csv_path):
    """Read Ground Truth (test.csv or query.csv)"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    # Search for the ID column
    col_id = next((c for c in ['objectID', 'Corresponding Indexes', 'object'] if c in df.columns), None)
    if col_id is None:
        raise ValueError(f"ID column not found in {csv_path}")

    # Return a dictionary {imageName: objectID} for easy alignment
    dataset = {}
    for _, row in df.iterrows():
        dataset[row['imageName']] = int(row[col_id])
    return dataset, df['imageName'].tolist()


def read_prediction_csv(csv_path):
    """Read the prediction CSV (Corresponding Indexes format)"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    col_pred = next((c for c in ['Corresponding Indexes', 'predictions'] if c in df.columns), None)

    if col_pred is None:
        raise ValueError(f"Prediction CSV must have a 'Corresponding Indexes' column")

    # Return a dictionary {imageName: [list of indices]}
    predictions = {}
    for _, row in df.iterrows():
        indices = [int(i) for i in str(row[col_pred]).split()]
        predictions[row['imageName']] = indices

    return predictions


# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="ReID Eval CSV Final")
parser.add_argument("--track", default="submission.csv", help="CSV file containing predictions")
parser.add_argument("--path", default="./data/", help="Folder containing query.csv and test.csv")
args = parser.parse_args()

# 1. Load Data
# Gallery: IDs needed in a list ordered by image index
gallery_dict, gallery_names = read_csv_gt(os.path.join(args.path, 'test.csv'))

# Sort gallery IDs so that index '1' corresponds to the image with the lowest numerical name
# (Or according to the order in which the gallery was generated)
sorted_gallery_names = sorted(gallery_names, key=lambda x: int(x.split('.')[0]))
id_gallery = np.array([gallery_dict[name] for name in sorted_gallery_names])

query_dict, query_names = read_csv_gt(os.path.join(args.path, 'query.csv'))
preds_dict = read_prediction_csv(args.track)

# --- EVALUATION ---
AP = 0.0
total_queries = 0
# Determine prediction size (e.g., 100)
sample_key = next(iter(preds_dict))
CMC = np.zeros(len(preds_dict[sample_key]))

print(f"Evaluating {len(query_names)} queries...")

for q_name in query_names:
    if q_name not in preds_dict:
        print(f"Warning: {q_name} is missing from the prediction file. Skipping...")
        continue

    query_id = query_dict[q_name]
    # Get predicted indices (1-based) and convert to 0-based
    pred_indices = np.array(preds_dict[q_name]) - 1

    # Actual IDs from the gallery according to our prediction
    sortID = id_gallery[pred_indices]

    # Find hit positions
    true_positives_in_gallery = np.where(id_gallery == query_id)[0]
    if len(true_positives_in_gallery) == 0:
        continue  # Nothing to find for this query

    rows_good = np.where(sortID == query_id)[0]

    ap = 0
    cmc = np.zeros(len(CMC))
    ngood = len(true_positives_in_gallery)

    if rows_good.size != 0:
        # CMC: From the first hit onwards, everything is marked as 1
        cmc[rows_good[0]:] = 1
        for i, pos in enumerate(rows_good):
            precision = (i + 1) / (pos + 1)
            # Simple interpolation for Average Precision
            if pos != 0:
                old_precision = (i + 1) / (pos + 1)
            else:
                old_precision = 1.0
            ap += (1.0 / ngood) * (old_precision + precision) / 2

    CMC += cmc
    AP += ap
    total_queries += 1

# --- RESULTS ---
mAP = AP / total_queries
mCMC = CMC / total_queries

print("\n" + "=" * 30)
print(f"FINAL RESULTS")
print("=" * 30)
print(f"mAP:      {mAP:.6f}")
print(f"Rank-1:   {mCMC[0]:.6f}")
print(f"Rank-5:   {mCMC[4]:.6f}")
print(f"Rank-10:  {mCMC[9]:.6f}")
print(f"Rank-20:  {mCMC[19]:.6f}")
print("=" * 30)
