import click
import pandas as pd

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--split-offline', default=False, type=bool, is_flag=True, help='Split offline shields into separate columns (default) or merge into one using shield_name==agent.')
def create_latex_table_data(csv_file, split_offline):
    """
    Create LaTeX table data from CSV evaluation results.
    """

    # Load data
    df = pd.read_csv(csv_file)

    # Define shield columns and their mapping to (shield, shield_name)
    if split_offline:
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
    else:
        shield_columns = [
            ("identity", "identity"),
            ("standard", "standard"),
            ("delta", "delta"),
            ("optimistic", "optimistic"),
            ("pessimistic", "pessimistic"),
            ("online", "online"),
            ("offline", None),  # merged offline column
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

    # Indices of shields to exclude from bolding (identity and optimistic)
    identity_index = 0  # index of identity shield
    optimistic_index = 3  # index of optimistic shield in both split_offline and merged

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
                # For each shield column, get (risk, allowed_pct)
                shield_data = []
                for shield, shield_name in shield_columns:
                    if shield == "offline" and not split_offline:
                        # merged offline: use shield_name == agent
                        match = row_df[(row_df['shield'] == 'offline') & (row_df['shield_name'] == agent)]
                    else:
                        match = row_df[(row_df['shield'] == shield) & (row_df['shield_name'] == shield_name)]
                    if not match.empty:
                        risk = match.iloc[0]['risk']
                        try:
                            risk = round(float(risk), 3)
                        except Exception:
                            pass
                        try:
                            shield_calls = float(match.iloc[0]['shield_calls'])
                            blocked_actions = float(match.iloc[0]['blocked_actions'])
                            if shield_calls > 0:
                                allowed_pct = round((1 - blocked_actions / shield_calls), 3)
                            else:
                                allowed_pct = '-'
                        except Exception:
                            allowed_pct = '-'
                        shield_data.append((risk, allowed_pct))
                    else:
                        shield_data.append(('-', '-'))
                # Find max allowed_pct (excluding identity and optimistic shields), for shields with risk <= nu
                allowed_pcts = [allowed_pct if idx != identity_index and idx != optimistic_index and allowed_pct != '-' and risk != '-' and float(risk) <= float(nu) else float('-inf')
                                for idx, (risk, allowed_pct) in enumerate(shield_data)]
                allowed_pcts = [float(x) if x != '-' and x != float('-inf') else float('-inf') for x in allowed_pcts]
                max_allowed_pct = max(allowed_pcts) if allowed_pcts else float('-inf')
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
                # For each shield, add risk and allowed_pct (rounded)
                nu_val = float(nu)
                for idx, (risk, allowed_pct) in enumerate(shield_data):
                    # Determine color for risk cell
                    if risk != '-' and float(risk) <= nu_val:
                        risk_color = None
                    elif risk != '-' and float(risk) > nu_val:
                        risk_color = 'red'
                    else:
                        risk_color = None
                    # Determine color and bold for allowed_pct cell
                    allowed_color = risk_color
                    bold = False
                    # Only consider non-identity and non-optimistic shields for bolding
                    if (
                        idx != identity_index and idx != optimistic_index and
                        allowed_pct != '-' and risk != '-' and float(risk) <= nu_val and float(allowed_pct) >= float(max_allowed_pct) and max_allowed_pct != float('-inf')
                    ):
                        bold = True
                    def format_val(val):
                        if val == '-':
                            return '-'
                        val_str = f"{float(val):.3f}"
                        if val_str.startswith('0.'):
                            return val_str[1:]
                        elif val_str[0] != '0':
                            return 'is 1'
                        else:
                            return val_str
                    risk_str = format_val(risk)
                    allowed_str = format_val(allowed_pct)
                    row.append(latex_cell(risk_str, color=risk_color, bold=bold))
                    row.append(latex_cell(allowed_str, color=allowed_color, bold=bold))
                model_block_lines.append(' & '.join(row) + r" \\")
            # Add cmidrule after each agent except last
            if a_idx < len(agents) - 1:
                model_block_lines.append(r" \cmidrule(lr){2-17}")
        # Add model block to output
        output_lines.extend(model_block_lines)
        # Add cmidrule between models except after last
        if m_idx < len(models) - 1:
            output_lines.append(r" \cmidrule(lr){1-17}")

    # Print output
    for line in output_lines:
        print(line)

if __name__ == '__main__':
    create_latex_table_data()