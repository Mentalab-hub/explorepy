import pandas as pd
import sys

def remove_time_decimals(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.replace('.csv', '_no_decimals.csv')
    
    df = pd.read_csv(input_file)
    
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df.to_csv(output_file, index=False)
    print(f"Time decimals removed. Output saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_time_decimals.py input_file.csv [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_time_decimals(input_file, output_file)