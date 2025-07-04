import pyarrow.parquet as pq
import pyarrow as pa

# --- HARDCODED CONFIG ---
INPUT_PATH = "../data/poses/conor1.parquet"
OUTPUT_PATH = "../data/poses/conor1.parquet"
REMAPPING = {
     20 : 1
}

# --- Load table ---
table = pq.read_table(INPUT_PATH)
df = table.to_pandas()

# --- Remap track_id ---
df["track_id"] = df["track_id"].apply(lambda tid: REMAPPING.get(tid, tid))

# --- Save updated file ---
new_table = pa.Table.from_pandas(df)
pq.write_table(new_table, OUTPUT_PATH, compression="zstd")

print(f"✅ Track IDs remapped and saved to {OUTPUT_PATH}")
