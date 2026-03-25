import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import datetime
import os

master_file = "mandi_master_data.parquet"
print("Reading parquet file...")
table = pq.read_table(master_file)

# The bad records were inserted from 2026-03-21 to 2026-03-25
# They are cloned rows from the live snapshot endpoint.
print(f"Original rows: {table.num_rows}")

# Create a filter expression: keep rows where Arrival_Date < 2026-03-21
target_date = datetime.datetime.strptime("2026-03-21", "%Y-%m-%d").date()
# Or handle if Arrival_Date is timestamp vs date
# The Parquet column Arrival_Date is stored as timestamp[ns] in daily_updater:
target_dt = datetime.datetime(2026, 3, 21)

condition = pc.less(table.column("Arrival_Date"), target_dt)
filtered_table = table.filter(condition)

print(f"Filtered rows: {filtered_table.num_rows}")
deleted = table.num_rows - filtered_table.num_rows
print(f"Deleted {deleted} corrupted clone rows.")

# Overwrite
if deleted > 0:
    print("Saving cleaned parquet file...")
    pq.write_table(filtered_table, master_file + ".clean")
    os.replace(master_file + ".clean", master_file)
    print("Done! Database successfully reverted to 2026-03-20.")
else:
    print("No corrupted rows found.")
