import click
import pandas as pd

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
def create_latex_table_data(csv_file):
    """
    Create LaTeX table data from CSV evaluation results.
    """

    # Load data
    df = pd.read_csv(csv_file)

    # Define shield columns and their mapping to (shield, shield_name)
    shield_columns = [
        ("identity", "identity"),
        ("standard", "standard"),
        ("delta", "delta"),
        ("optimistic", "optimistic"),
        ("pessimistic", "pessimistic"),
        ("online", "online"),
        ("offline", "greedy"),
        ("offline", "safe"),
        ("offline", "random"),
    ]

    # For LaTeX output
    def latex_cell(val, color=None, bold=False):
        if val == '-' or val is None:
            return '-'
        s = f"{val}"
        if bold:
            s = f"\\textbf{{{s}}}"
        if color:
            s = f"\\cellcolor{{{color}!20}}{s}"
        return s

    # Group by model, agent, nu
    models = df['model'].unique()
    output_lines = []
    for m_idx, model in enumerate(models):
        model_df = df[df['model'] == model]
        agents = model_df['agent'].unique()
        model_block_lines = []
        for a_idx, agent in enumerate(agents):
            agent_df = model_df[model_df['agent'] == agent]
            nu_values = sorted(agent_df['nu'].unique(), key=lambda x: float(x))
            for n_idx, nu in enumerate(nu_values):
                row_df = agent_df[agent_df['nu'] == nu]
                # For each shield column, get (risk, reward)
                shield_data = []
                for shield, shield_name in shield_columns:
                    match = row_df[(row_df['shield'] == shield) & (row_df['shield_name'] == shield_name)]
                    if not match.empty:
                        risk = match.iloc[0]['risk']
                        reward = match.iloc[0]['reward']
                        shield_data.append((risk, reward))
                    else:
                        shield_data.append(('-', '-'))
                # Determine coloring and bolding
                nu_val = float(nu)
                green_mask = [(r != '-' and float(r) < nu_val) for r, _ in shield_data]
                # Find max reward among green cells
                green_rewards = [float(reward) if is_green and reward != '-' else float('-inf') for (is_green, (risk, reward)) in zip(green_mask, shield_data)]
                max_green_reward = max(green_rewards) if green_rewards else float('-inf')
                # Build latex row
                row = []
                # Model column
                if a_idx == 0 and n_idx == 0:
                    row.append(f"\\multirow{{{len(agents)*len(nu_values)}}}{{*}}{{{model}}}")
                elif n_idx == 0:
                    row.append('')
                else:
                    row.append('')
                # Agent column
                if n_idx == 0:
                    row.append(f"\\multirow{{{len(nu_values)}}}{{*}}{{{agent}}}")
                else:
                    row.append('')
                # nu column
                row.append(str(nu))
                # For each shield, add risk and reward (rounded)
                for idx, (risk, reward) in enumerate(shield_data):
                    is_green = green_mask[idx]
                    color = 'green' if is_green else ('red' if risk != '-' else None)
                    # Bold if green and reward is max among green
                    bold = is_green and reward != '-' and float(reward) == max_green_reward and max_green_reward != float('-inf')
                    risk_str = f"{float(risk):.2f}" if risk != '-' else '-'
                    reward_str = f"{float(reward):.1f}" if reward != '-' else '-'
                    row.append(latex_cell(risk_str, color=color, bold=bold))
                    row.append(latex_cell(reward_str, color=color, bold=bold))
                model_block_lines.append(' & '.join(row) + r" \\")
            # Add cmidrule after each agent except last
            if a_idx < len(agents) - 1:
                model_block_lines.append(r" \cmidrule(lr){2-21}")
        # Add model block to output
        output_lines.extend(model_block_lines)
        # Add cmidrule between models except after last
        if m_idx < len(models) - 1:
            output_lines.append(r" \cmidrule(lr){1-21}")

    # Print output
    for line in output_lines:
        print(line)

if __name__ == '__main__':
    create_latex_table_data()