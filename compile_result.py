import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def format_value(value, precision=1, is_time=False):
    """Helper function to format numbers for LaTeX, handling NaN."""
    if pd.isna(value):
        return "N/A"
    if is_time: # This specific flag might be less relevant if format_avg_std_time handles precision
        return f"{value:.3f}" 
    return f"{value:.{precision}f}"

def format_avg_std_time(avg_val, std_val, precision=3):
    """Formats average and standard deviation as 'Avg ± Std' for LaTeX."""
    if pd.isna(avg_val):
        return "N/A"
    
    avg_str = f"{avg_val:.{precision}f}"
    
    if pd.isna(std_val) or std_val == 0.0: # Treat 0.0 std dev as not needing to show ± 0.000
        return avg_str 
    else:
        std_str = f"{std_val:.{precision}f}"
        return f"{avg_str} $\\pm$ {std_str}"

def process_solver_data(filepath, solver_display_name):
    """
    Reads a CSV file, calculates success stats, time stats, duration stats, and length stats, and returns them.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return None

    # Ensure success columns are boolean
    if 'warm_success' in df.columns:
        df['warm_success_bool'] = df['warm_success'].astype(str).str.lower() == 'true'
    else:
        print(f"Warning: 'warm_success' column not found in {filepath}. Assuming 0 success.")
        df['warm_success_bool'] = False
        
    if 'cold_success' in df.columns:
        df['cold_success_bool'] = df['cold_success'].astype(str).str.lower() == 'true'
    else:
        print(f"Warning: 'cold_success' column not found in {filepath}. Assuming 0 success.")
        df['cold_success_bool'] = False

    # Columns to convert to numeric
    numeric_cols = {
        'warm_time_s': 'warm_time_s',
        'cold_time_s': 'cold_time_s',
        'warm_traj_actual_duration': 'warm_traj_actual_duration',
        'cold_traj_actual_duration': 'cold_traj_actual_duration',
        'warm_traj_length_metric': 'warm_traj_length_metric',
        'cold_traj_length_metric': 'cold_traj_length_metric'
    }

    for col_name, original_col_name in numeric_cols.items():
        if original_col_name in df.columns:
            df[col_name] = pd.to_numeric(df[original_col_name], errors='coerce')
        else:
            print(f"Warning: '{original_col_name}' column not found in {filepath}.")
            df[col_name] = np.nan


    total_runs = len(df)
    
    base_stats = {
        'name': solver_display_name,
        'warm_total': total_runs, 'warm_successful': 0, 'warm_rate': 0.0,
        'avg_warm_time': np.nan, 'std_warm_time': np.nan,
        'avg_warm_duration': np.nan, 'std_warm_duration': np.nan,
        'avg_warm_length': np.nan, 'std_warm_length': np.nan,
        'cold_total': total_runs, 'cold_successful': 0, 'cold_rate': 0.0,
        'avg_cold_time': np.nan, 'std_cold_time': np.nan,
        'avg_cold_duration': np.nan, 'std_cold_duration': np.nan,
        'avg_cold_length': np.nan, 'std_cold_length': np.nan,
    }
    if total_runs == 0:
        return base_stats

    warm_successful_runs = df['warm_success_bool'].sum()
    cold_successful_runs = df['cold_success_bool'].sum()

    base_stats['warm_successful'] = warm_successful_runs
    base_stats['cold_successful'] = cold_successful_runs
    base_stats['warm_rate'] = (warm_successful_runs / total_runs) * 100 if total_runs > 0 else 0.0
    base_stats['cold_rate'] = (cold_successful_runs / total_runs) * 100 if total_runs > 0 else 0.0

    # Stats for successful WARM starts
    successful_warm_df = df[df['warm_success_bool']]
    if not successful_warm_df.empty:
        for metric_prefix, col_name_suffix in [('time', '_s'), ('duration', '_actual_duration'), ('length', '_length_metric')]:
            col_name = f'warm_traj{col_name_suffix}' if metric_prefix != 'time' else 'warm_time_s' # Adjusted for time col name
            if col_name in successful_warm_df:
                valid_data = successful_warm_df[col_name].dropna()
                if not valid_data.empty:
                    base_stats[f'avg_warm_{metric_prefix}'] = valid_data.mean()
                    if len(valid_data) > 1:
                        base_stats[f'std_warm_{metric_prefix}'] = valid_data.std()
                    elif len(valid_data) == 1:
                        base_stats[f'std_warm_{metric_prefix}'] = 0.0


    # Stats for successful COLD starts
    successful_cold_df = df[df['cold_success_bool']]
    if not successful_cold_df.empty:
        for metric_prefix, col_name_suffix in [('time', '_s'), ('duration', '_actual_duration'), ('length', '_length_metric')]:
            col_name = f'cold_traj{col_name_suffix}' if metric_prefix != 'time' else 'cold_time_s' # Adjusted for time col name
            if col_name in successful_cold_df:
                valid_data = successful_cold_df[col_name].dropna()
                if not valid_data.empty:
                    base_stats[f'avg_cold_{metric_prefix}'] = valid_data.mean()
                    if len(valid_data) > 1:
                        base_stats[f'std_cold_{metric_prefix}'] = valid_data.std()
                    elif len(valid_data) == 1:
                        base_stats[f'std_cold_{metric_prefix}'] = 0.0
    return base_stats

# --- Main script ---
dcol_file = 'benchmark_results_dcol_mintime.csv'
trajopt_file = 'benchmark_results_trajopt_mintime.csv'

dcol_stats = process_solver_data(dcol_file, "Dircol")
trajopt_stats = process_solver_data(trajopt_file, "TrajOpt (iLQR)")

# --- Generate LaTeX Table ---
latex_output = []
latex_output.append("\\begin{table}[ht]")
latex_output.append("\\centering")
latex_output.append("\\caption{Solver Performance for Minimum Time Problems}")
latex_output.append("\\label{tab:solver_performance_mintime}")
# Adjusted tabular for new columns: Method(l), Start(l), Total(r), Success(r), Rate(r), Time(r), Duration(r), Length(r)
latex_output.append("\\begin{tabular}{llrrrrrr}") 
latex_output.append("\\toprule")
latex_output.append("Method & Start & Total & Success & Rate(\\%) & Time (Avg$\\pm$Std)(s) & Duration (Avg$\\pm$Std)(s) & Length (Avg$\\pm$Std)(m) \\\\")
latex_output.append("\\midrule")

solvers_data = []
if dcol_stats:
    solvers_data.append(dcol_stats)
if trajopt_stats:
    solvers_data.append(trajopt_stats)

for i, stats in enumerate(solvers_data):
    if stats:
        # Warm start
        latex_output.append(f"\\multirow{{2}}{{*}}{{{stats['name']}}} & Warm & "
                            f"{stats['warm_total']} & "
                            f"{stats['warm_successful']} & "
                            f"{format_value(stats['warm_rate'])} & "
                            f"{format_avg_std_time(stats['avg_warm_time'], stats['std_warm_time'], precision=3)} & "
                            f"{format_avg_std_time(stats['avg_warm_duration'], stats['std_warm_duration'], precision=2)} & "
                            f"{format_avg_std_time(stats['avg_warm_length'], stats['std_warm_length'], precision=2)} \\\\")
        # Cold start
        latex_output.append(f" & Cold & "
                            f"{stats['cold_total']} & "
                            f"{stats['cold_successful']} & "
                            f"{format_value(stats['cold_rate'])} & "
                            f"{format_avg_std_time(stats['avg_cold_time'], stats['std_cold_time'], precision=3)} & "
                            f"{format_avg_std_time(stats['avg_cold_duration'], stats['std_cold_duration'], precision=2)} & "
                            f"{format_avg_std_time(stats['avg_cold_length'], stats['std_cold_length'], precision=2)} \\\\")
        if i < len(solvers_data) - 1:
             latex_output.append("\\midrule")

latex_output.append("\\bottomrule")
latex_output.append("\\end{tabular}")
latex_output.append("\\end{table}")

print("\n".join(latex_output))

# --- Optional: Print a summary to console ---
print("\n--- Summary ---")
for stats in solvers_data:
    if stats:
        print(f"\n{stats['name']}:")
        print(f"  Warm Start: {stats['warm_successful']}/{stats['warm_total']} successful ({format_value(stats['warm_rate'])}%)")
        print(f"    Time (Avg ± Std): {format_avg_std_time(stats['avg_warm_time'], stats['std_warm_time'], precision=3)}s")
        print(f"    Duration (Avg ± Std): {format_avg_std_time(stats['avg_warm_duration'], stats['std_warm_duration'], precision=2)}s")
        print(f"    Length (Avg ± Std): {format_avg_std_time(stats['avg_warm_length'], stats['std_warm_length'], precision=2)}m")
        print(f"  Cold Start: {stats['cold_successful']}/{stats['cold_total']} successful ({format_value(stats['cold_rate'])}%)")
        print(f"    Time (Avg ± Std): {format_avg_std_time(stats['avg_cold_time'], stats['std_cold_time'], precision=3)}s")
        print(f"    Duration (Avg ± Std): {format_avg_std_time(stats['avg_cold_duration'], stats['std_cold_duration'], precision=2)}s")
        print(f"    Length (Avg ± Std): {format_avg_std_time(stats['avg_cold_length'], stats['std_cold_length'], precision=2)}m")

# --- Generate Scatter Plots for Dircol ---
def generate_dircol_scatter_plots(filepath):
    try:
        df_dcol = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Plotting data file '{filepath}' not found.")
        return
    except Exception as e:
        print(f"Error reading plotting data file '{filepath}': {e}")
        return

    plot_metrics = [
        {'title': 'Runtime (Dircol)', 'warm_col': 'warm_time_s', 'cold_col': 'cold_time_s', 'unit': 's'},
        {'title': 'Trajectory Time (Dircol)', 'warm_col': 'warm_traj_actual_duration', 'cold_col': 'cold_traj_actual_duration', 'unit': 's'},
        {'title': 'Path Length (Dircol)', 'warm_col': 'warm_traj_length_metric', 'cold_col': 'cold_traj_length_metric', 'unit': 'rad'},
        {'title': 'Average Jerk (Dircol)', 'warm_col': 'warm_avg_jerk_metric', 'cold_col': 'cold_avg_jerk_metric', 'unit': r'rad/s^3'} # Adjust jerk unit if known
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6)) # 1 row, 4 columns
    #fig.suptitle('Dircol: Warm Start vs. Cold Start Performance Metrics (Successful Runs)', fontsize=16)

    for i, metric in enumerate(plot_metrics):
        ax = axes[i]
        
        if metric['warm_col'] not in df_dcol.columns or metric['cold_col'] not in df_dcol.columns:
            print(f"Warning: Columns for '{metric['title']}' not found in {filepath}. Skipping plot.")
            ax.set_title(f"{metric['title']}\n(Data Missing)")
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
            continue

        # Prepare data for the current plot
        warm_data = pd.to_numeric(df_dcol[metric['warm_col']], errors='coerce')
        cold_data = pd.to_numeric(df_dcol[metric['cold_col']], errors='coerce')
        
        # Combine into a temporary DataFrame for easy dropping of NaNs/Infs for the pair
        plot_df = pd.DataFrame({'warm': warm_data, 'cold': cold_data}).copy()
        
        # Drop rows where EITHER warm or cold data is NaN or Inf for this specific metric pair
        plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        plot_df.dropna(subset=['warm', 'cold'], inplace=True)

        if plot_df.empty:
            print(f"Warning: No valid data points for '{metric['title']}' after cleaning. Skipping plot.")
            ax.set_title(f"{metric['title']}\n(No Valid Data)")
            ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', transform=ax.transAxes)
            continue
            
        ax.scatter(plot_df['cold'], plot_df['warm'], alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
        
        # Add y=x line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0], 0),  # Lower limit, ensuring 0 is included if positive
            max(ax.get_xlim()[1], ax.get_ylim()[1])   # Upper limit
        ]
        if not np.isnan(lims[0]) and not np.isnan(lims[1]) and lims[0] < lims[1]: # Ensure lims are valid
             ax.plot(lims, lims, 'r--', alpha=0.75, lw=2, label='y=x')
             ax.set_xlim(lims)
             ax.set_ylim(lims)
        
        ax.set_title(metric['title'], fontsize=21)
        ax.set_xlabel(f"Cold Start ({metric['unit']})", fontsize=18)
        ax.set_ylabel(f"Warm Start ({metric['unit']})", fontsize=18)
        # set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle=':', alpha=0.7)
        if lims[0] < lims[1]: # Only add legend if y=x line was plotted
            ax.legend(fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig('dircol_scatter_plots.png', dpi=800, bbox_inches='tight')
    #plt.show()

# Call the plotting function
if dcol_file: # Ensure dcol_file is defined
    print("\nGenerating Dircol scatter plots...")
    generate_dircol_scatter_plots(dcol_file)