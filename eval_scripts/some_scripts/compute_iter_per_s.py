import csv
import sys

# Usage: python compute_iter_per_s.py input.csv output.csv

def add_iter_per_second(input_path, output_path):
    with open(input_path, newline='') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        rows = list(reader)
        fieldnames = reader.fieldnames + ['iter_per_second']

    for row in rows:
        try:
            shield_calls = float(row['shield_calls'])
            eval_elapsed_time = float(row['eval_elapsed_time'])
            iter_per_second = shield_calls / eval_elapsed_time if eval_elapsed_time != 0 else ''
        except (KeyError, ValueError):
            iter_per_second = ''
        row['iter_per_second'] = iter_per_second

    with open(output_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python compute_iter_per_s.py input.csv output.csv')
        sys.exit(1)
    add_iter_per_second(sys.argv[1], sys.argv[2])
