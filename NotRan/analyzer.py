#!/usr/bin/env python3
import csv
from collections import Counter
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ColumnAnalyzer:
    def __init__(self, chunk_size: int = 100000, top_n: int = 20):
        self.chunk_size = chunk_size
        self.total_rows = 0
        self.columns = None
        self.missing_counts = {}
        self.value_counters = {}
        self.top_n = top_n

    def _init_counters(self, columns: List[str]):
        """Initialize counter dictionaries for each column"""
        self.columns = columns
        for col in columns:
            self.missing_counts[col] = 0
            self.value_counters[col] = Counter()

    def _update_counters(self, chunk: pd.DataFrame):
        """Update counters with data from a chunk"""
        self.total_rows += len(chunk)

        for col in self.columns:
            # Count missing values
            missing_mask = (
                chunk[col].isna() | (chunk[col] == "") | (chunk[col].astype(str).str.strip() == "")
            )
            self.missing_counts[col] += missing_mask.sum()

            # Update value frequencies (excluding missing values)
            values = chunk[col][~missing_mask].astype(str)
            self.value_counters[col].update(values)

    def analyze_file(self, filepath: str) -> Dict:
        """Analyze CSV file in chunks and return statistics"""
        print(f"Analyzing {filepath}...")

        # Process file in chunks
        first_chunk = True
        for chunk in tqdm(pd.read_csv(filepath, chunksize=self.chunk_size)):
            if first_chunk:
                self._init_counters(chunk.columns)
                first_chunk = False
            self._update_counters(chunk)

        return self._generate_report()

    def _generate_report(self) -> Dict:
        """Generate analysis report"""
        report = {}

        for col in self.columns:
            # Calculate missing percentage
            missing_pct = (self.missing_counts[col] / self.total_rows) * 100

            # Get top N most frequent values
            top_values = self.value_counters[col].most_common(self.top_n)

            # Calculate percentages for top values
            top_values_with_pct = [
                (value, count, (count / (self.total_rows - self.missing_counts[col])) * 100)
                for value, count in top_values
            ]

            report[col] = {"missing_percentage": missing_pct, "top_values": top_values_with_pct}

        return report


def format_report(report: Dict) -> str:
    """Format the analysis report as readable text"""
    output = []

    for col, stats in report.items():
        output.append(f"\n{'='*80}")
        output.append(f"Column: {col}")
        output.append(f"Missing Data: {stats['missing_percentage']:.2f}%")

        output.append(f"\nTop {len(stats['top_values'])} Most Frequent Values:")
        for value, count, percentage in stats["top_values"]:
            output.append(f"  - {value}: {count:,} occurrences ({percentage:.2f}% of non-missing)")

    return "\n".join(output)


def save_report_as_csv(report: Dict, output_file: str):
    """Save the report in a CSV format suitable for Google Sheets"""
    # Prepare data for two CSV files: missing data summary and value frequencies

    # Missing data summary
    missing_data = pd.DataFrame(
        [
            {"Column": col, "Missing_Percentage": stats["missing_percentage"]}
            for col, stats in report.items()
        ]
    )
    missing_data.to_csv(f"missing_data_{output_file}", index=False)

    # Value frequencies
    # Create a DataFrame with columns for each statistic and rows for each column's values
    freq_data = []
    for col, stats in report.items():
        for value, count, percentage in stats["top_values"]:
            freq_data.append(
                {"Column": col, "Value": value, "Count": count, "Percentage": percentage}
            )

    freq_df = pd.DataFrame(freq_data)
    freq_df.to_csv(f"frequencies_{output_file}", index=False)

    print(f"\nAnalysis exported to:")
    print(f"1. missing_data_{output_file} - Summary of missing data percentages")
    print(
        f"2. frequencies_{output_file} - Top {len(report[list(report.keys())[0]]['top_values'])} values for each column"
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_csv.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    analyzer = ColumnAnalyzer(top_n=20)  # Now getting top 20 values

    try:
        report = analyzer.analyze_file(input_file)
        print(format_report(report))

        # Save as CSV files
        save_report_as_csv(report, output_file)

        # Also save text report
        with open("column_analysis_report.txt", "w") as f:
            f.write(format_report(report))
            print("\nDetailed report saved to column_analysis_report.txt")

    except Exception as e:
        print(f"Error analyzing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
