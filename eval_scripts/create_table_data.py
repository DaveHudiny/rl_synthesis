import click
import pandas as pd

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--split-offline', default=False, type=bool, is_flag=True, help='Split offline shields into separate columns (default) or merge into one using shield_name==agent.')
@click.option('--show-identity', default=False, type=bool, is_flag=True, help='Show identity shield column in the table.')
def create_latex_table_data(csv_file, split_offline, show_identity):
    """
    Create LaTeX table data from CSV evaluation results.
    """

    # Load data
    df = pd.read_csv(csv_file)

    # Remap nu values: 0.005 -> 0.05, 0.001 -> 0.01
    nu_remap = {0.005: 0.05, 0.001: 0.01, '0.005': 0.05, '0.001': 0.01}
    df['nu'] = df['nu'].apply(lambda x: nu_remap[x] if x in nu_remap else x)

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
            # ("pessimistic", "pessimistic"),
            ("online", "online"),
            ("offline", None),  # merged offline column
            ("mem1", None),     # new mem1 column
        ]
    # Remove identity shield column unless show_identity is True
    if not show_identity:
        shield_columns = [col for col in shield_columns if col[0] != "identity"]
    # Calculate number of columns for cmidrule
    # Columns: model, agent, nu, then 2 columns per shield (risk, allowed_pct)
    num_shields = len(shield_columns)
    num_columns = 3 + 2 * num_shields

    

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
    
    model_order = ['corridor', 'dpm', 'drone', 'drone-b']
    model_rename = {'corridor': 'corridor', 'drone': 'drone', 'dpm': 'dpm', 'drone-b': 'drone-b'}

    agent_order = ['greedy', 'safe', 'random']
    agent_rename = {'greedy': 'greedy', 'safe': 'timid', 'random': 'random'}

    # Indices of shields to exclude from bolding (identity and optimistic)
    identity_index = next((i for i, (shield, _) in enumerate(shield_columns) if shield == "identity"), None)
    optimistic_index = next((i for i, (shield, _) in enumerate(shield_columns) if shield == "optimistic"), None)

    # Group by model, agent, nu, using specified order and renaming
    output_lines = []
    for m_idx, model in enumerate(model_order):
        model_df = df[df['model'] == model]
        agents = [a for a in agent_order if a in model_df['agent'].unique()]
        model_block_lines = []
        for a_idx, agent in enumerate(agents):
            agent_df = model_df[model_df['agent'] == agent]
            nu_values = sorted(agent_df['nu'].unique(), key=lambda x: float(x))
            identity_risks = []  # Collect identity shield risks for each nu
            for n_idx, nu in enumerate(nu_values):
                row_df = agent_df[agent_df['nu'] == nu]
                # For each shield column, get (risk, allowed_pct)
                shield_data = []
                # Always get identity shield risk for agent label, even if not shown
                match_identity = row_df[(row_df['shield'] == 'identity') & (row_df['shield_name'] == 'identity')]
                if not match_identity.empty:
                    identity_risk = match_identity.iloc[0]['risk']
                    try:
                        identity_risk = round(float(identity_risk), 3)
                    except Exception:
                        pass
                else:
                    identity_risk = '-'
                identity_risks.append(identity_risk)
                for shield, shield_name in shield_columns:
                    if shield == "offline" and not split_offline:
                        match = row_df[(row_df['shield'] == 'offline') & (row_df['shield_name'] == agent)]
                    elif shield == "mem1" and not split_offline:
                        match = row_df[(row_df['shield'] == 'offline') & (row_df['shield_name'] == agent + '-1')]
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
                allowed_pcts = [allowed_pct if (identity_index is None or idx != identity_index) and (optimistic_index is None or idx != optimistic_index) and allowed_pct != '-' and risk != '-' and float(risk) <= float(nu) else float('-inf')
                                for idx, (risk, allowed_pct) in enumerate(shield_data)]
                allowed_pcts = [float(x) if x != '-' and x != float('-inf') else float('-inf') for x in allowed_pcts]
                max_allowed_pct = max(allowed_pcts) if allowed_pcts else float('-inf')
                # Build latex row
                row = []
                # Model column
                if a_idx == 0 and n_idx == 0:
                    row.append(f"\\multirow{{{len(agents)*len(nu_values)}}}{{*}}{{{model_rename[model]}}}")
                elif n_idx == 0:
                    row.append('')
                else:
                    row.append('')
                # Agent column
                if n_idx == 0:
                    # For the last nu (last row for this agent), append identity risk in brackets
                    agent_label = agent_rename[agent]
                    if len(nu_values) > 0:
                        last_identity_risk = identity_risks[-1] if len(identity_risks) == len(nu_values) and len(nu_values) > 0 else None
                        if last_identity_risk is not None:
                            def format_val(val):
                                if val == '-':
                                    return '-'
                                val_str = f"{float(val):.3f}"
                                if val_str.startswith('0.'):
                                    return val_str[1:]
                                elif val_str[0] != '0':
                                    return '~~{=}1'
                                else:
                                    return val_str
                            last_identity_risk_str = format_val(last_identity_risk)
                        else:
                            last_identity_risk_str = '-'
                    else:
                        last_identity_risk_str = '-'
                    # Only add bracket for the last row for this agent
                    if len(nu_values) == 1:
                        agent_label = f"{agent_label} [{last_identity_risk_str}]"
                    else:
                        agent_label = f"{agent_label}"
                    row.append(f"\\multirow{{{len(nu_values)}}}{{*}}{{{agent_label}}}")
                elif n_idx == len(nu_values) - 1:
                    # Last row for this agent: append identity risk in brackets
                    agent_label = agent_rename[agent]
                    last_identity_risk = identity_risks[-1]
                    def format_val(val):
                        if val == '-':
                            return '-'
                        val_str = f"{float(val):.3f}"
                        if val_str.startswith('0.'):
                            return val_str[1:]
                        elif val_str[0] != '0':
                            return '~~{=}1'
                        else:
                            return val_str
                    last_identity_risk_str = format_val(last_identity_risk)
                    row.append(f"({last_identity_risk_str})")
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
                        (identity_index is None or idx != identity_index) and (optimistic_index is None or idx != optimistic_index) and
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
                            return '~~{=}1'
                        else:
                            return val_str
                    risk_str = format_val(risk)
                    allowed_str = format_val(allowed_pct)
                    row.append(latex_cell(risk_str, color=risk_color, bold=bold))
                    row.append(latex_cell(allowed_str, color=allowed_color, bold=bold))
                model_block_lines.append(' & '.join(row) + r" \\")
            # Add cmidrule after each agent except last
            if a_idx < len(agents) - 1:
                # cmidrule for agent block: starts at agent column (2), ends at last column
                model_block_lines.append(f" \\cmidrule(lr){{2-{num_columns}}}")
        # Add model block to output
        output_lines.extend(model_block_lines)
        # Add cmidrule between models except after last
        if m_idx < len(model_order) - 1:
            # cmidrule for model block: starts at model column (1), ends at last column
            output_lines.append(f" \\cmidrule(lr){{1-{num_columns}}}")

    # Print output
    for line in output_lines:
        print(line)

if __name__ == '__main__':
    create_latex_table_data()