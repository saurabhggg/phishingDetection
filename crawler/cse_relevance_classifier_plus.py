import pandas as pd
import glob

# Path where your Excel files are located (adjust as needed)
path = "./"  # same folder as script, or something like "C:/Users/You/Documents/excel_files"

# Get all .xlsx files in the folder
excel_files = glob.glob(path + "*.xlsx")

# List to store DataFrames
dfs = []

# Read each Excel file and append to list
for file in excel_files:
    try:
        df = pd.read_excel(file)
        df["source_file"] = file  # Optional: keep track of which file each row came from
        dfs.append(df)
        print(f"✅ Loaded: {file} ({len(df)} rows)")
    except Exception as e:
        print(f"❌ Could not read {file}: {e}")

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to CSV
    combined_df.to_csv("combined_output.csv", index=False, encoding='utf-8-sig')
    print(f"\n🎉 Combined {len(excel_files)} files → 'combined_output.csv' ({len(combined_df)} rows)")
else:
    print("⚠️ No valid Excel files found.")
