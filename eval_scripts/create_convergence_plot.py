import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), default=None, help='Output file for the plot (e.g., plot.png). If not set, shows the plot interactively.')
@click.option('--nu', type=float, default=None, required=True, help='Nu value to filter the data.')
@click.option('--model', type=str, default=None, required=True, help='Model name to filter the data.')
@click.option('--agent', type=str, default=None, required=True, help='Agent name to filter the data.')
def plot_convergence(csv_file, output, nu, model, agent):
    """
    Create a line plot from CSV data, with lines for each unique shield_memory value.
    """

    # Load data
    df = pd.read_csv(csv_file, sep=',')

    # Filter data based on model and agent
    if model is not None:
        df = df[df['model'] == model]
    if agent is not None:
        df = df[df['agent'] == agent]
    df = df[df['nu'] == nu]

    if df.empty:
        raise ValueError("No data available after filtering by model and agent.")

    # Extract shield memory from shield_name using regex
    import re
    def extract_memory(shield_name):
        match = re.search(r'-mem_(\d+)', str(shield_name))
        return int(match.group(1)) if match else None

    df['shield_memory'] = df['shield_name'].apply(extract_memory)

    # Compute expected_allowed_actions
    if 'blocked_actions' in df.columns and 'shield_calls' in df.columns:
        df['expected_allowed_actions'] = 1 - (df['blocked_actions'] / df['shield_calls'])
    else:
        raise ValueError("blocked_actions and/or shield_calls columns are missing in the data.")

    plt.figure(figsize=(8, 6))
    unique_memories = sorted(df['shield_memory'].dropna().unique())
    colormap = plt.cm.get_cmap('tab20', len(unique_memories))
    for shield_mem, group in df.groupby('shield_memory'):
        # Sort by iteration and drop duplicate iterations, keeping the last value for each iter
        group_sorted = group.sort_values('iter').drop_duplicates('iter', keep='last')
        x = [0] + list(group_sorted['iter'])
        y = [0] + list(group_sorted['expected_allowed_actions'])
        plt.plot(
            x,
            y,
            label=f'shield_memory={shield_mem}',
            color=colormap(int(shield_mem)),
            linewidth=2.5
        )


    plt.xlabel('Iteration')
    plt.ylabel('Expected allowed actions')
    plt.title(f'Convergence Plot Allowed actions vs Iterations')
    plt.legend(title='Shield Memory')
    plt.tight_layout()

    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        plt.show()

if __name__ == '__main__':
    plot_convergence()