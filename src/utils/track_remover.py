import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

# --- CONFIG ---
INPUT_PATH = "../data/pose_tables/conor/conor-dustin.parquet"
OUTPUT_PATH = "../data/pose_tables/conor/conor-dustin.parquet"
TARGET_ID = 1  # the track_id you want to keep

# --- Load table ---
table = pq.read_table(INPUT_PATH)
df = table.to_pandas()

# --- Filter by track_id ---
filtered_df = df[df["track_id"] == TARGET_ID]

# --- Save to new parquet file ---
new_table = pa.Table.from_pandas(filtered_df)
pq.write_table(new_table, OUTPUT_PATH, compression="zstd")

print(f"✅ Saved filtered file with track_id == {TARGET_ID} to {OUTPUT_PATH}")
