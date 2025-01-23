#!/usr/bin/env python3
import random
import csv
from pathlib import Path
import sys
from tqdm import tqdm

def count_lines(filename):
    """Count lines in a large file efficiently"""
    print("Counting lines in file...")
    count = 0
    with open(filename, 'rb') as f:
        # Count newline characters
        count = sum(1 for _ in f)
    return count - 1  # Subtract 1 for header

def sample_large_csv(input_file, output_file, sample_percentage=0.005):
    """
    Sample rows from a large CSV file using reservoir sampling.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output sampled CSV file
        sample_percentage (float): Percentage of rows to sample (default: 0.02 for 2%)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} does not exist")
        return

    # Count total lines for sampling and progress bar
    total_lines = count_lines(input_file)
    sample_size = int(total_lines * sample_percentage)
    print(f"Total lines (excluding header): {total_lines}")
    print(f"Will sample {sample_size} lines ({sample_percentage*100}%)")

    # Read header
    with open(input_file, 'r') as infile:
        header = next(csv.reader([infile.readline()]))

    # Initialize reservoir with first k elements
    reservoir = []
    processed = 0
    
    print("Reading initial sample...")
    with open(input_file, 'r') as infile:
        next(infile)  # Skip header
        for _ in range(sample_size):
            try:
                line = next(infile)
                reservoir.append(line)
                processed += 1
            except StopIteration:
                break

    # Continue with reservoir sampling
    print("Processing remaining lines...")
    with open(input_file, 'r') as infile:
        next(infile)  # Skip header
        # Skip lines we've already processed
        for _ in range(processed):
            next(infile)
        
        # Process remaining lines with progress bar
        with tqdm(total=total_lines-processed) as pbar:
            for i, line in enumerate(infile, start=processed):
                pbar.update(1)
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line

    # Write sampled data to output file
    print(f"Writing sampled data to {output_file}...")
    with open(output_file, 'w', newline='') as outfile:
        # Write header
        writer = csv.writer(outfile)
        writer.writerow(header)
        
        # Write sampled rows
        outfile.writelines(reservoir)

    print("Sampling complete!")
    print(f"Sampled data written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_csv.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sample_large_csv(input_file, output_file)
