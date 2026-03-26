import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import datetime
import os

master_file = "mandi_master_data.parquet"
print("Reading parquet file...")
table = pq.read_table(master_file)

# Erase anything from the 26th onwards
target_dt = datetime.datetime(2026, 3, 26)
condition = pc.less(table.column("Arrival_Date"), target_dt)
filtered_table = table.filter(condition)

deleted = table.num_rows - filtered_table.num_rows

if deleted > 0:
    print(f"Filtering complete. Detected {deleted} incomplete records from the 26th onwards.")
    print("Saving cleaned parquet file...")
    pq.write_table(filtered_table, master_file + ".clean")
    os.replace(master_file + ".clean", master_file)
    print("Done! Database successfully reverted to the end of March 25th.")
else:
    print("No records found for the 26th or later.")
