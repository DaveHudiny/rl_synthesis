import os
import glob
import pandas as pd
import click

def merge_csv_files(input_folder, output_file):
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not csv_files:
        print(f'No CSV files found in {input_folder}')
        return
    df_list = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f'Merged {len(csv_files)} files into {output_file}')

def average_csv_files(input_folder, average_file):
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not csv_files:
        print(f'No CSV files found in {input_folder}')
        return
    df_list = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    if merged_df.shape[1] < 6:
        print('Not enough columns to average by first 5 columns.')
        return
    grouped = merged_df.groupby(list(merged_df.columns[:5]), as_index=False).mean(numeric_only=True)
    grouped.to_csv(average_file, index=False)
    print(f'Averaged rows saved to {average_file}')

@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False), required=True)
@click.argument('output_file', type=click.Path(), default='merged_output.csv', required=False)
@click.option('--average', is_flag=True, help='Average rows with same values in first 5 columns and save to another file.')
@click.option('--average-file', type=click.Path(), default='average_output.csv', help='Output file for averaged rows.')
def main(input_folder, output_file, average, average_file):
    """Merge all CSV files in a folder into one CSV file. Optionally average rows with same values in first 5 columns."""
    merge_csv_files(input_folder, output_file)
    if average:
        average_csv_files(input_folder, average_file)

if __name__ == '__main__':
    main()
