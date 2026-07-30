"""
Runs KMeans clustering for n_clusters = 1 … 30 over the full Ausgrid dataset
and saves one ranked CSV per cluster count to clustering_results/.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ── DATA ──────────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "Ausgrid_2010_2013_Orange_Combined.csv")

print(f"Loading {csv_path} …")
raw = pd.read_csv(csv_path, usecols=["Timestamp_UTC", "UserID", "Energy_Consumption"])
raw["Timestamp_UTC"] = pd.to_datetime(raw["Timestamp_UTC"], utc=True).dt.tz_localize(None)

n_users = 300
selected_users = np.sort(raw["UserID"].dropna().unique())[:n_users]
raw = raw[raw["UserID"].isin(selected_users)]

data = (
    raw.pivot_table(index="Timestamp_UTC", columns="UserID",
                    values="Energy_Consumption", aggfunc="mean")
    .sort_index()
    .astype(np.float32)
    .fillna(0.0)
)
data.columns = [f"user_{int(c)}" for c in data.columns]
print(f"Data shape: {data.shape}")

# ── FEATURES ──────────────────────────────────────────────────────────────────
def reshape2threedays(df_in):
    df = df_in.copy()
    df.columns.name = "ts_id"
    df.index.name = "timestamp"
    df = df.unstack().reset_index()
    dayofweek = df.timestamp.dt.dayofweek
    days = dayofweek.where(dayofweek >= 5, 0).rename("dayofweek")
    return (
        pd.pivot_table(df, values=0, index=df.ts_id,
                       columns=[days, df.timestamp.rename("hour").dt.hour])
        .round(3)
    )

def normalize_df(df):
    return df.apply(lambda s: s / df.max(axis=1)).fillna(0)

print("Computing weekly mean profiles …")
X_norm = normalize_df(reshape2threedays(data.resample("1h").mean()))
X = X_norm.to_numpy(dtype=float)
print(f"Feature matrix: {X.shape}")

# ── SWEEP ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(base_dir, "clustering_results")
os.makedirs(out_dir, exist_ok=True)

for n_clusters in range(1, 31):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(X)
    labels = model.labels_
    dist_to_centroid = np.linalg.norm(X - model.cluster_centers_[labels], axis=1)

    ranking_df = pd.DataFrame({
        "user_id":          X_norm.index,
        "cluster":          labels,
        "dist_to_centroid": dist_to_centroid,
    })
    ranking_df = (
        ranking_df
        .sort_values(["cluster", "dist_to_centroid"])
        .reset_index(drop=True)
    )
    ranking_df["rank_in_cluster"] = ranking_df.groupby("cluster").cumcount() + 1

    out_path = os.path.join(out_dir, f"user_ids_sorted_by_cluster_{n_clusters}.csv")
    ranking_df.to_csv(out_path, index=False)
    print(f"  n_clusters={n_clusters:2d} → {os.path.basename(out_path)}")

print("Done. All files saved to clustering_results/")
