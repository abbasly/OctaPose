import pyarrow.parquet as pq
import pyarrow as pa
import glob, os

SRC_DIR   = "../data/poses/clean"
OUT_FILE  = "../data/poses/all_fighters.parquet"

tables = []
for fpath in glob.glob(os.path.join(SRC_DIR, "*.parquet")):
    tbl = pq.read_table(fpath)
    fighter = os.path.basename(fpath).split(".")[0]      # eg  "conor3"
    name    = "".join(filter(str.isalpha, fighter))      # "conor"
    seq_id  = "".join(filter(str.isdigit, fighter)) or 0 # "3"
    tbl = tbl.append_column("fighter", pa.array([name]*tbl.num_rows))
    tbl = tbl.append_column("seq_id",  pa.array([int(seq_id)]*tbl.num_rows))
    tables.append(tbl)

pq.write_table(pa.concat_tables(tables, promote=True),
               OUT_FILE, compression="zstd")