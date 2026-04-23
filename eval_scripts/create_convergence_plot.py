import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.option('--csv-file', type=click.Path(exists=True), default='22-01-convergence-data.csv', show_default=True, help='CSV data file to load.')
@click.option('--output', type=click.Path(), default='convergence.pdf', show_default=True, help='Output file for the plot. If set to .pgf, uses LaTeX PGF backend.')
@click.option('--xmax', type=float, default=None, help='Maximum value for the x-axis.')
@click.option('--plot-risk', is_flag=True, default=False, help='Plot the risk column on the y-axis instead of expected allowed actions.')
@click.option('--log-scale', is_flag=True, default=False, help='Use log scale for the x-axis with ticks at 10, 1000, 100000.')
@click.option('--dpi', type=int, default=300, help='DPI for raster outputs (PNG, PDF). Ignored for PGF output.')
@click.option('--target-width-in', type=float, default=None, help='Target figure width in inches; useful for small figures in papers.')
@click.option('--textwidth-cm', type=float, default=12.2, help='LaTeX \\textwidth in cm; figure width set to 0.47*textwidth.')
@click.option('--font-size', type=float, default=None, help='Base font size (pt) for axes, ticks, and legend.')
def plot_convergence(csv_file, output, xmax, plot_risk, log_scale, dpi, target_width_in, textwidth_cm, font_size):

    # Load data
    df = pd.read_csv(csv_file, sep=',')


    # Filter for nu == 0.2 and shield == 'constructed' (if needed)
    df = df[df['nu'] == 0.2]
    df = df[df['shield'] == 'constructed']

    # Add one to each value in the 'iter' column
    if 'iter' in df.columns:
        df['iter'] = df['iter'] + 1

    if df.empty:
        raise ValueError("No data available after filtering by model and agent.")

    if not plot_risk:
        y_axis_starts = {('dpm', 'greedy'): 0.771, ('corridor', 'greedy'): 0.02, ('drone', 'greedy'): 0.207, ('drone-b', 'greedy'): 0.16,
                         ('dpm', 'safe'): 0.981, ('corridor', 'safe'): 0.02, ('drone', 'safe'): 0.554, ('drone-b', 'safe'): 0.987,
                         ('dpm', 'random'): 0.798, ('corridor', 'random'): 0.116, ('drone', 'random'): 0.722, ('drone-b', 'random'): 0.762}


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

    import matplotlib
    # Configure PGF backend for LaTeX if requested
    if output and str(output).endswith('.pgf'):
        matplotlib.use('pgf')
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
        })

    # Determine target figure width and height (maintain original 10:7 aspect ratio)
    aspect_w = 10.0
    aspect_h = 5.0
    aspect_ratio = aspect_h / aspect_w

    fig_width_in = None
    if target_width_in is not None:
        fig_width_in = float(target_width_in)
    elif textwidth_cm is not None:
        # Convert cm to inches and apply 0.47*textwidth scaling
        fig_width_in = 0.47 * float(textwidth_cm) / 2.54
    else:
        fig_width_in = aspect_w  # default to original width

    fig_height_in = fig_width_in * aspect_ratio

    # Adjust font sizes for small figures or if explicitly provided
    if font_size is not None:
        base = float(font_size)
        plt.rcParams.update({
            "font.size": base,
            "axes.labelsize": base,
            "xtick.labelsize": max(base - 1, 6),
            "ytick.labelsize": max(base - 1, 6),
            "legend.fontsize": max(base - 1, 6),
            "axes.titlesize": base,
        })
    else:
        # Heuristic: if very small figure width, use compact fonts
        if fig_width_in <= 3.0:
            plt.rcParams.update({
                "font.size": 4,
                "axes.labelsize": 5,
                "xtick.labelsize": 4,
                "ytick.labelsize": 4,
                "legend.fontsize": 4,
                "axes.titlesize": 5,
            })

    # Create figure with computed size; DPI applies to raster outputs
    if output and str(output).endswith('.pgf'):
        plt.figure(figsize=(fig_width_in, fig_height_in))
    else:
        plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)

    # Make axes compact: reduce label/tick padding and margins
    plt.rcParams.update({
        "axes.labelpad": 2,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.3,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.2,
        "legend.borderpad": 0.2,
        "legend.columnspacing": 0.6,
    })
    plt.margins(x=0.01, y=0.02)

    

    # Colorblind-friendly color palette (Okabe-Ito)
    colorblind_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442']
    model_list = sorted(df['model'].unique())
    model_color_map = {model: colorblind_palette[i % len(colorblind_palette)] for i, model in enumerate(model_list)}

    def get_model_color(model):
        return model_color_map.get(model, '#666666')

    def get_agent_linestyle(agent):
        if 'greedy' in agent:
            return '-'
        elif 'safe' in agent or 'timid' in agent:
            return '--'
        elif 'random' in agent:
            return ':'
        else:
            return '-.'

    # Store handles for custom legend
    model_handles = {}
    agent_handles = {}
    from matplotlib.lines import Line2D

    for (model, agent), group in df.groupby(['model', 'agent']):
        group_sorted = group.sort_values('iter').drop_duplicates('iter', keep='last')
        if log_scale:
            x = [1] + list(group_sorted['iter'] * 4000)
        else:
            x = [0] + list(group_sorted['iter'] * 4000)
        if plot_risk:
            y = [0] + list(group_sorted['risk'])
        else:
            y = [y_axis_starts[(model, agent)]] + list(group_sorted['expected_allowed_actions'])
        line, = plt.plot(
            x,
            y,
            color=get_model_color(model),
            linestyle=get_agent_linestyle(agent),
            linewidth=1.2,
            alpha=0.85 if agent != 'random' else 0.6
        )
        # Shorten dash patterns for compact stripes
        style = get_agent_linestyle(agent)
        if style == '--':
            line.set_dashes([2, 1])
        elif style == ':':
            line.set_dashes([0.6, 1.2])
        elif style == '-.':
            line.set_dashes([2, 1, 0.6, 1])
        # Only add one handle per model for color legend
        if model not in model_handles:
            model_handles[model] = Line2D([0], [0], color=get_model_color(model), lw=1.2)
        # Only add one handle per agent type for style legend (all in black)
        style = get_agent_linestyle(agent)
        agent_key = None
        if 'greedy' in agent:
            agent_key = 'greedy'
        elif 'safe' in agent or 'timid' in agent:
            agent_key = 'safe'
        elif 'random' in agent:
            agent_key = 'random'
        if agent_key and agent_key not in agent_handles:
            handle = Line2D([0], [0], color='black', linestyle=style, lw=1.2)
            # Compact dash patterns in legend to avoid long stripes
            if style == '--':
                handle.set_dashes([2, 1])
            elif style == ':':
                handle.set_dashes([0.6, 1.2])
            elif style == '-.':
                handle.set_dashes([2, 1, 0.6, 1])
            agent_handles[agent_key] = handle


    plt.xlabel('Construction steps')
    if plot_risk:
        plt.ylabel('Risk')
    else:
        plt.ylabel('Allowed actions percentage')
    # plt.xscale('symlog')

    if log_scale:
        plt.xscale('log')
        if xmax is not None:
            import numpy as np
            min_tick = 1
            max_tick = int(xmax)
            log_min = int(np.ceil(np.log10(min_tick)))
            log_max = int(np.floor(np.log10(max_tick)))
            xticks = [int(10**i) for i in range(log_min, log_max+1) if 10**i <= xmax]
            if not xticks or xticks[0] > min_tick:
                xticks = [min_tick] + xticks
            xlabels = [str(t) if t < 1000 else f"{int(t/1000)}k" for t in xticks]
            plt.xticks(xticks, xlabels)
            plt.xlim(left=min_tick, right=xmax)
        else:
            xticks = [10, 1000, 100000]
            xlabels = ['10', '1k', '100k']
            plt.xticks(xticks, xlabels)
    else:
        if xmax is not None:
            import numpy as np
            xticks = list(np.linspace(0, xmax, 5+1)[1:])  # skip 0 if not in data
            xlabels = [f"{int(x/1000)}k" if x >= 1000 else str(int(x)) for x in xticks]
            plt.xticks(xticks, xlabels)
            plt.xlim(left=0, right=xmax)
        else:
            xticks = [4000, 200000, 400000, 600000, 800000, 1000000]
            xlabels = ['4k','200k', '400k', '600k', '800k', '1M']
            plt.xticks(xticks, xlabels)

    # Custom legend: 4 model colors, 3 agent styles
    model_labels = [model for model in model_handles]
    model_lines = [model_handles[model] for model in model_labels]
    agent_labels = ['greedy', 'timid', 'random']
    agent_keys = ['greedy', 'safe', 'random']
    agent_lines = [agent_handles[k] for k in agent_keys if k in agent_handles]
    # Reserve space on the right for legends, so they don't overlap data
    plt.tight_layout(rect=[0, 0, 0.78, 1])

    # Place legends outside the axes to the right
    legend1 = plt.legend(model_lines, model_labels, title='Model', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    legend2 = plt.legend(agent_lines, agent_labels[:len(agent_lines)], title='Agent', loc='lower left', bbox_to_anchor=(1.02, 0.0))
    plt.gca().add_artist(legend1)

    if output:
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        plt.show()

if __name__ == '__main__':
    plot_convergence()