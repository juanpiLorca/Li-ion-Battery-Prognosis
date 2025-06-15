import os
import glob
import pandas as pd

def add_header_to_csv(file_pattern, headers):
    for filepath in glob.glob(file_pattern):
        print(f"Processing {filepath}...")
        
        # Read existing file
        df = pd.read_csv(filepath, header=None)  # no headers
        
        # Assign new headers
        df.columns = headers
        
        # Overwrite with headers
        df.to_csv(filepath, index=False)
        print(f"Updated: {filepath}")

# --- Add headers to current and voltage files ---
add_header_to_csv('current/current_trace_*.csv', ['time', 'current'])
add_header_to_csv('voltage/voltage_trace_*.csv', ['time', 'voltage'])
