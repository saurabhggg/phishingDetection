#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Prefilter Output for Heavy Classifier
--------------------------------------------
Takes `prefilter_scored.csv`, analyzes score distribution,
and exports only top URLs for the heavy classifier.

✅ Computes descriptive stats & quantiles
✅ Allows threshold by percentile or value
✅ Plots histogram
✅ Exports to CSV (prefilter_for_heavy.csv)
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def main(prefilter_path, output_path, percentile=None, min_score=None):
    print("📂 Reading prefilter file:", prefilter_path)
    df = pd.read_csv(prefilter_path)

    if "CSE_Score" not in df.columns:
        print("❌ Error: Column 'CSE_Score' not found in input file.")
        sys.exit(1)

    print("\n📊 Basic statistics:")
    print(df["CSE_Score"].describe())

    # Plot distribution
    plt.figure(figsize=(8,4))
    plt.hist(df["CSE_Score"], bins=40, color="skyblue", edgecolor="black")
    plt.title("CSE_Score Distribution")
    plt.xlabel("CSE_Score")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("CSE_Score_Distribution.png")
    print("📸 Histogram saved as 'CSE_Score_Distribution.png'")

    # Threshold selection
    if percentile:
        threshold = df["CSE_Score"].quantile(percentile)
        print(f"📈 Using top {int((1 - percentile) * 100)}% (score >= {threshold:.4f})")
    elif min_score:
        threshold = min_score
        print(f"📈 Using score threshold >= {threshold:.4f}")
    else:
        threshold = df["CSE_Score"].quantile(0.95)
        print(f"📈 Defaulting to top 5% (score >= {threshold:.4f})")

    # Filter and save
    filtered_df = df[df["CSE_Score"] >= threshold].copy()
    filtered_df.to_csv(output_path, index=False)
    print(f"\n✅ Exported {len(filtered_df)} records to {output_path}")
    print(f"   (from {len(df)} total records)")

    # Optional: print top few
    print("\n🧾 Sample records:")
    print(filtered_df.head(10)[["Domain", "CSE_Score"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Prefilter Output for Heavy Classifier")
    parser.add_argument("--input", required=True, help="Path to prefilter_scored.csv")
    parser.add_argument("--output", default="prefilter_for_heavy.csv", help="Output CSV for heavy classifier")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile cutoff (e.g. 0.95 for top 5%)")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum CSE_Score threshold (e.g. 0.90)")
    args = parser.parse_args()

    main(args.input, args.output, args.percentile, args.min_score)
