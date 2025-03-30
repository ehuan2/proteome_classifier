import numpy as np
import csv

def round_csv_to_4_decimals(csv_file_path, output_file_path):
    # Read the CSV file
    with open(csv_file_path, 'r') as input_file:
        reader = csv.reader(input_file)
        data = list(reader)

    # Round all numeric columns to 4 decimal places
    for row in data:
        for i, value in enumerate(row):
            try:
                row[i] = '{:.4f}'.format(float(value))
            except ValueError:
                pass

    # Write the rounded data to a new CSV file
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(data)

# Example usage:
csv_file_path = 'unrounded_predictions.csv'
output_file_path = 'predictions.csv'
round_csv_to_4_decimals(csv_file_path, output_file_path)
