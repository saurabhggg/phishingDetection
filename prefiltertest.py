import pandas as pd

df = pd.read_csv("prefilter_scored.csv")

# Check score distribution
print(df["CSE_Score"].describe(percentiles=[.5, .8, .9, .95, .99]))

# Filter top suspicious domains
threshold = 0.45  # you can tune this later
filtered = df[df["CSE_Score"] >= threshold]

filtered.to_csv("prefilter_for_heavy.csv", index=False)
print(f"✅ Filtered {len(filtered)} domains (CSE_Score >= {threshold})")
