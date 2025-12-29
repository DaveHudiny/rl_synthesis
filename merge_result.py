import os
import glob
import pandas as pd
import click

def merge_csv_files(input_folder, output_file):
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not csv_files:
        print(f'No CSV files found in {input_folder}')
        return
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, delimiter=';', on_bad_lines='warn')
            df_list.append(df)
        except Exception as e:
            print(f'Error reading {f}: {e}')
    if not df_list:
        print('No valid CSV files to merge.')
        return
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False, sep=';')
    print(f'Merged {len(df_list)} files into {output_file}')

def average_csv_files(input_folder, average_file):
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not csv_files:
        print(f'No CSV files found in {input_folder}')
        return
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, delimiter=';', on_bad_lines='warn')
            df_list.append(df)
        except Exception as e:
            print(f'Error reading {f}: {e}')
    if not df_list:
        print('No valid CSV files to average.')
        return
    merged_df = pd.concat(df_list, ignore_index=True)
    if merged_df.shape[1] < 6:
        print('Not enough columns to average by first 5 columns.')
        return
    grouped = merged_df.groupby(list(merged_df.columns[:5]), as_index=False).mean(numeric_only=True)
    grouped.to_csv(average_file, index=False, sep=';')
    print(f'Averaged rows saved to {average_file}')

@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False), required=True)
@click.argument('output_file', type=click.Path(), default='results/merged_output.csv', required=False)
@click.option('--average', is_flag=True, help='Average rows with same values in first 5 columns and save to another file.')
@click.option('--average-file', type=click.Path(), default='results/average_output.csv', help='Output file for averaged rows.')
def main(input_folder, output_file, average, average_file):
    """Merge all CSV files in a folder into one CSV file. Optionally average rows with same values in first 5 columns."""
    merge_csv_files(input_folder, output_file)
    if average:
        average_csv_files(input_folder, average_file)

if __name__ == '__main__':
    main()
