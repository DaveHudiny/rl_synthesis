import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--yaxis', type=click.Choice(['risk', 'reward']), default='risk', help='Column to plot on the y-axis.')
@click.option('--output', type=click.Path(), default=None, help='Output file for the plot (e.g., plot.png). If not set, shows the plot interactively.')
@click.option('--nu', type=float, default=None, help='Upper bound for risk (drawn as a black dotted line if yaxis is risk).')
@click.option('--agent-value', type=float, default=None, help='Reference value for the agent (drawn as a dashed blue line and added to the legend as "Agent value").')
def plot_convergence(csv_file, yaxis, output, nu, agent_value):
    """
    Create a line plot from CSV data, with lines for each unique shield_memory value.
    """

    # Load data
    df = pd.read_csv(csv_file, sep=';')
    # If 'risk' is requested, but only 'safety' column exists, convert it
    if 'risk' in df.columns:
        pass
    elif 'safety' in df.columns:
        df['risk'] = df['safety']
    else:
        raise ValueError("CSV must contain either 'risk' or 'safety' column.")

    # Prepare plot
    plt.figure(figsize=(8, 6))
    colormap = plt.cm.get_cmap('tab10', len(df['shield_memory'].unique()))
    for idx, (shield_mem, group) in enumerate(df.groupby('shield_memory')):
        group_sorted = group.sort_values('iter')
        # Prepend (0,0) to each line
        x = [0] + list(group_sorted['iter'])
        y = [0] + list(group_sorted[yaxis])
        plt.plot(
            x,
            y,
            label=f'shield_memory={shield_mem}',
            color=colormap(idx)
        )



    # If yaxis is risk and nu is provided, plot a black dotted line at y=nu (not in legend)
    if yaxis == 'risk' and nu is not None:
        plt.axhline(y=nu, color='black', linestyle=':', linewidth=2, zorder=1)

    # If agent_value is provided, plot a dashed black line and add to legend
    if agent_value is not None:
        plt.axhline(y=agent_value, color='black', linestyle='--', linewidth=2, label='Agent value', zorder=2)

    plt.xlabel('Iteration')
    plt.ylabel(yaxis.capitalize())
    plt.title(f'Convergence Plot ({yaxis.capitalize()} vs Iteration)')
    plt.legend(title='Shield Memory')
    plt.tight_layout()

    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        plt.show()

if __name__ == '__main__':
    plot_convergence()