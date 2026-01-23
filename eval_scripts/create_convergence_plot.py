import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), default=None, help='Output file for the plot (e.g., plot.png). If not set, shows the plot interactively.')
def plot_convergence(csv_file, output):

    # Load data
    df = pd.read_csv(csv_file, sep=',')


    # Filter for nu == 0.2 and shield == 'constructed' (if needed)
    df = df[df['nu'] == 0.2]
    df = df[df['shield'] == 'constructed']

    if df.empty:
        raise ValueError("No data available after filtering by model and agent.")

    # Extract shield memory from shield_name using regex
    import re
    def extract_memory(shield_name):
        match = re.search(r'-mem_(\d+)', str(shield_name))
        return int(match.group(1)) if match else None

    df['shield_memory'] = df['shield_name'].apply(extract_memory)

    df = df[df['shield_memory'] == 0]

    # Compute expected_allowed_actions
    if 'blocked_actions' in df.columns and 'shield_calls' in df.columns:
        df['expected_allowed_actions'] = 1 - (df['blocked_actions'] / df['shield_calls'])
    else:
        raise ValueError("blocked_actions and/or shield_calls columns are missing in the data.")

    plt.figure(figsize=(10, 7))

    # Color mapping for models and agents
    model_colors = {
        'corridor': {
            'greedy': '#145A32',   # dark green
            'safe':   '#229954',   # normal green
            'timid':  '#229954',   # normal green (alias for safe)
            'random': '#82E0AA',   # light green
        },
        'dpm': {
            'greedy': '#154360',   # dark blue
            'safe':   '#2874A6',   # normal blue
            'timid':  '#2874A6',   # normal blue (alias for safe)
            'random': '#85C1E9',   # light blue
        },
        # Add more models and their color schemes as needed
    }

    def get_agent_color(model, agent):
        model_key = None
        for key in model_colors:
            if key in str(model):
                model_key = key
                break
        if model_key:
            colors = model_colors[model_key]
            if agent in colors:
                return colors[agent]
            if 'safe' in agent:
                return colors['safe']
            if 'greedy' in agent:
                return colors['greedy']
            if 'random' in agent:
                return colors['random']
            return list(colors.values())[0]
        # fallback
        return '#666666'

    # Group by (model, agent)
    for (model, agent), group in df.groupby(['model', 'agent']):
        group_sorted = group.sort_values('iter').drop_duplicates('iter', keep='last')
        x = [0] + list(group_sorted['iter'] * 4000)
        y = [0] + list(group_sorted['expected_allowed_actions'])
        plt.plot(
            x,
            y,
            label=f'{model}, {agent}',
            color=get_agent_color(model, agent),
            linewidth=2.5,
            alpha=0.85 if agent != 'random' else 0.6
        )

    plt.xlabel('Construction steps')
    plt.ylabel('Expected allowed actions')
    plt.title('Convergence Plot: Allowed actions vs Construction Steps')
    # plt.xscale('symlog')
    # Set x-axis ticks and labels for 200k, 400k, 600k, 800k, 1M
    xticks = [200000, 400000, 600000, 800000, 1000000]
    xlabels = ['200k', '400k', '600k', '800k', '1M']
    plt.xticks(xticks, xlabels)
    plt.legend()
    plt.tight_layout()

    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        plt.show()

if __name__ == '__main__':
    plot_convergence()