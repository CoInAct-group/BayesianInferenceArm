import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config_ukf as c 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import os # Needed for checking/creating the output directory
import seaborn as sns # Add seaborn import
from scipy.linalg import eigh # For eigendecomposition
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.cm as cm


# --- Global Plotting Style Configuration ---
# Seaborn style and global context
# The original script had plt.rcParams['legend.fontsize'] = 'xx-small' at the top.
# We use this value in sns.set_context's rc parameter to ensure it's fixed
# and not scaled by font_scale.
sns.set_theme(style="whitegrid", palette="muted", 
              rc={'legend.fontsize': 'xx-small'}) # Base theme with explicit legend fontsize

# Matplotlib rcParams for further global settings
plt.rcParams['lines.linewidth'] = 1.5
# plt.rcParams['figure.figsize'] = (15, 10) # Default, can be overridden per plot
plt.rcParams['savefig.dpi'] = 300 # Good default for saved figures
plt.rcParams['savefig.bbox'] = 'tight' # Good default for saved figures

# Force legend fontsize after seaborn setup (seaborn may override rcParams)
plt.rcParams['legend.fontsize'] = 'xx-small'

# Default alpha values and sizes (can be overridden locally)
GLOBAL_CI_ALPHA = 0.2
GLOBAL_LINE_ALPHA_LOW = 0.5 # For less prominent lines like "estimated"
GLOBAL_SCATTER_ALPHA = 0.5
GLOBAL_SCATTER_SIZE_DEFAULT = 20 # s parameter for scatter
GLOBAL_SCATTER_SIZE_SMALL = 5   # Smaller scatter markers (e.g., 'obs_size')
GLOBAL_LEGEND_FONTSIZE = 'xx-small'  # Global constant for legend fontsize
# --- End Global Plotting Style Configuration ---


def create_circle(center_x, center_y, radius, num_points=19):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y
def calculate_ellipse_points(mean, covariance_matrix, n_std=1.96, num_points=50):
    """
    Calculates the X and Y coordinates for a confidence ellipse.

    Args:
        mean (np.ndarray): 2D array [x_mean, y_mean] for the center of the ellipse.
        covariance_matrix (np.ndarray): 2x2 covariance matrix [[var_x, cov_xy], [cov_xy, var_y]].
        n_std (float): Number of standard deviations for the ellipse (e.g., 1.96 for 95% CI).
        num_points (int): Number of points to define the ellipse perimeter.

    Returns:
        tuple: (ellipse_x_coords, ellipse_y_coords)
               Returns ([], []) if covariance is invalid or contains NaNs.
    """
    if covariance_matrix is None or np.isnan(covariance_matrix).any() or mean is None or np.isnan(mean).any():
        return np.array([]), np.array([])

    # Ensure covariance matrix is 2x2
    if covariance_matrix.shape != (2, 2):
        print(f"Warning: Invalid covariance matrix shape: {covariance_matrix.shape}. Expected (2,2).")
        return np.array([]), np.array([])

    try:
        # Eigen-decomposition
        eigenvalues, eigenvectors = eigh(covariance_matrix) # Use eigh for symmetric matrices
    except np.linalg.LinAlgError:
        print("Warning: Eigen-decomposition failed for covariance matrix.")
        return np.array([]), np.array([])

    # Check for non-positive eigenvalues which can occur from numerical issues or non-positive semi-definite matrix
    if np.any(eigenvalues <= 0):
        print(f"Warning: Non-positive eigenvalues found: {eigenvalues}. Ellipse cannot be drawn.")
        # Attempt to draw a circle if one eigenvalue is ~0 and other is positive (very thin ellipse)
        # For simplicity, returning empty if any are non-positive for now.
        return np.array([]), np.array([])


    # Angle of the major axis
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Lengths of semi-axes
    # Eigenvalues are variances along principal axes
    semi_axis_a = n_std * np.sqrt(eigenvalues[0]) # Corresponds to first eigenvector
    semi_axis_b = n_std * np.sqrt(eigenvalues[1]) # Corresponds to second eigenvector
    
    # Ensure semi-axes are not NaN
    if np.isnan(semi_axis_a) or np.isnan(semi_axis_b):
        return np.array([]), np.array([])

    # Parametric points for a standard ellipse
    t = np.linspace(0, 2 * np.pi, num_points)
    ellipse_std_x = semi_axis_a * np.cos(t)
    ellipse_std_y = semi_axis_b * np.sin(t)

    # Rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])

    # Rotate and translate
    ellipse_rotated = R @ np.vstack([ellipse_std_x, ellipse_std_y])
    ellipse_x = ellipse_rotated[0, :] + mean[0]
    ellipse_y = ellipse_rotated[1, :] + mean[1]

    return ellipse_x, ellipse_y

def plotly_animation(results, show_fig=False, frame_decimation=4, annimation_speedup=1, output_filename=None, extra_text=None, file_type = None):
    org_len = len(results)
    if org_len == 0:
        print("Warning: results DataFrame is empty. Cannot generate plotly_animation.")
        return
    substract_len = org_len % frame_decimation
    results.drop(results.index[(org_len-substract_len):], inplace=True)
    results = results.iloc[::frame_decimation].copy()
    if results.empty:
        print(f"Warning: results DataFrame is empty after decimation with factor {frame_decimation}. Cannot generate plotly_animation.")
        return
    results['row_index'] = results.index
    endpoints = results.groupby(['run', 'trial']).tail(1)
    if c.elbow_down:
        shoulder_marker = {'size': 20, 'color': 'peru'}
        upper_arm_line = {'width': 16, 'color': 'darkorange'}
        lower_arm_line = {'width': 12, 'color': 'orange'}
        upper_arm_marker = {'size': 16, 'color': 'darkorange'}
        lower_arm_marker = {'size': 12, 'color': 'orange'}
    else:
        shoulder_marker = {'size': 15, 'color': 'darkorange'}
        upper_arm_line = {'width': 15, 'color': 'darkorange'}
        lower_arm_line = {'width': 15, 'color': 'darkorange'}
        upper_arm_marker = {'size': 15, 'color': 'darkorange'}
        lower_arm_marker = {'size': 15, 'color': 'darkorange'}

    lp_upper_arm_line_true = {'width': 15, 'color': 'steelblue'}
    lp_lower_arm_line_true = {'width': 15, 'color': 'steelblue'}
    lp_upper_arm_marker_true = {'size': 15, 'color': 'steelblue'}
    lp_lower_arm_marker_true = {'size': 15, 'color': 'steelblue'}
    rp_upper_arm_line_true = {'width': 5, 'color': 'steelblue'}
    rp_lower_arm_line_true = {'width': 5, 'color': 'steelblue'}
    rp_upper_arm_marker_true = {'size': 5, 'color': 'steelblue'}
    rp_lower_arm_marker_true = {'size': 5, 'color': 'steelblue'}


    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('True hand movement', 'Agent belief of hand movement'),
        horizontal_spacing=0.05
    )
    # shoulder both subplots
    fig.add_trace(go.Scatter(
        x=[results['true_shoulder_x'].iloc[0]],
        y=[results['true_shoulder_y'].iloc[0]],
        mode='markers',
        marker=lp_upper_arm_marker_true,
        name='Shoulder',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[results['true_shoulder_x'].iloc[0]],
        y=[results['true_shoulder_y'].iloc[0]],
        mode='markers',
        marker=shoulder_marker,
        name='Shoulder',
        legendgroup='posterior_arm',
        showlegend=False
    ), row=1, col=2)

    ### Add initial traces to Subplot 1 ###

    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_lower_arm_line_true,
        marker=lp_lower_arm_marker_true,
        name='Lower arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_upper_arm_line_true,
        marker=lp_upper_arm_marker_true,
        name='Upper arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    # Previous Positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='dodgerblue', opacity=0.5),
        name='Previous positions',
        legendgroup='true_trajectory',
        showlegend=True
    ), row=1, col=1)
    # planned trajectory
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.5),
        name='Planned trajectory',
        legendgroup='planned_trajectory',
        showlegend=True
    ), row=1, col=1)
    # Current Position
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=6, color='dodgerblue'),
        name='Current position',
        legendgroup='true_trajectory',
        showlegend=True
    ), row=1, col=1)

    # Target Circle
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='black'),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Target',
        legendgroup='target',
        showlegend=True
    ), row=1, col=1)

    # Input positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='darkblue', opacity=0.5),
        name='Input position',
        legendgroup='input_position',
        showlegend=True
    ), row=1, col=1)
    # Visual feedback
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.3),
        name='Visual feedback',
        legendgroup='visual_feedback',
        showlegend=True
    ), row=1, col=1)

    ### Add initial traces to Subplot 2 ###
    # Posterior hand uncertainty
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        # line=dict(color='grey'),
        line=dict(
            color='rgba(128, 128, 128, 0.5)',  # Line color
            width=2,                           # Line width
            # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
            dash='solid'
        ),            fill='toself',
        fillcolor='rgba(128, 128, 128, 0.5)',
        name='Posterior hand uncertainty',
        # legendgroup='group2',
        showlegend=False
    ), row=1, col=2)
    # Cartesian sigma points
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='green', opacity=0.5),
        name='Cartesian sigma points',
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lower_arm_line,
        marker=lower_arm_marker,
        name='Lower arm posterior',
        legendgroup='posterior_arm',
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=upper_arm_line,
        marker=upper_arm_marker,
        name='Upper arm posterior',
        legendgroup='posterior_arm',
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=rp_lower_arm_line_true,
        marker=rp_lower_arm_marker_true,
        name='Lower arm true',
        legendgroup='true_arm_overlay',
        showlegend=True
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=rp_upper_arm_line_true,
        marker=rp_upper_arm_marker_true,
        name='Upper arm true',
        legendgroup='true_arm_overlay',
        showlegend=True
    ), row=1, col=2)
    # Previous Positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='darkred', opacity=0.5),
        name='Previous positions posterior',
        legendgroup='posterior_trajectory',
        showlegend=True
    ), row=1, col=2)
    # planned trajectory
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.5),
        name='Planned trajectory',
        legendgroup='planned_trajectory',
        # showlegend=True
    ), row=1, col=2)
    # Current Position
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=6, color='darkred'),
        name='Current position posterior',
        legendgroup='posterior_trajectory',
        showlegend=True
    ), row=1, col=2)

    # Target Circle
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='black'),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Target',
        legendgroup='target',
        showlegend=True
    ), row=1, col=2)

    # Input positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='darkred', opacity=0.5),
        name='Input position posterior',
        legendgroup='input_position',
        showlegend=True
    ), row=1, col=2)
        # Visual feedback
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.3),
        name='Visual feedback',
        legendgroup='visual_feedback',
        showlegend=True
    ), row=1, col=2)
    # Body

    if c.task_type == "patterson2017":
        # Target out
        target_out_x, target_out_y = create_circle(
            results['p_target_out_x'].iloc[0],
            results['p_target_out_y'].iloc[0],
            results['r_target_out'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_out_x,
            y=target_out_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_out_x,
            y=target_out_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        # Target home
        target_home_x, target_home_y = create_circle(
            results['p_target_home_x'].iloc[0],
            results['p_target_home_y'].iloc[0],
            results['r_target_home'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_home_x,
            y=target_home_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target home',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_home_x,
            y=target_home_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        # Target final
        target_final_x, target_final_y = create_circle(
            results['p_target_final_x'].iloc[0],
            results['p_target_final_y'].iloc[0],
            results['r_target_out'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_final_x,
            y=target_final_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target final',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_final_x,
            y=target_final_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        
        

    elif c.task_type == "circular_following":
        outer_path_x, outer_path_y = create_circle(
            results['circular_target_movement_c_x'].iloc[0],
            results['circular_target_movement_c_y'].iloc[0],
            results['circular_target_movement_r'].iloc[0] + results['r_target'].iloc[0],
            num_points=72)

        inner_path_x, inner_path_y = create_circle(
            results['circular_target_movement_c_x'].iloc[0],
            results['circular_target_movement_c_y'].iloc[0],
            results['circular_target_movement_r'].iloc[0] - results['r_target'].iloc[0],
            num_points=72)

        fig.add_trace(go.Scatter(
            x=outer_path_x,
            y=outer_path_y,
            mode='lines',
            # line=dict(color='grey'),
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=inner_path_x,
            y=inner_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=outer_path_x,
            y=outer_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=inner_path_x,
            y=inner_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=1)

    frames = []
    i = -1

    total_iterations = len(results)
    # print(f"Total iterations: {total_iterations}")
    runs = results['run'].unique()
    frame_times = []
    # print(f"Total runs: {n_runs}")
    with tqdm(total=total_iterations, desc="Animation progress") as pbar:
        for run in runs:
            # print(f"Run: {run}")
            trials = results[results['run'] == run]['trial'].unique()

            for trial in trials:
                n_steps_this_trial = len(
                    results[(results['trial'] == trial) & (results['run'] == run)])
                results_trial = results[(results['trial'] == trial) & (
                    results['run'] == run)]
                
                # Pre-extract all sigma point data for the current trial
                all_sigmas_x_for_trial = []
                all_sigmas_y_for_trial = []
                if c.plot_sigma_points:
                    if 'sigmas_cartesian_transformed_0_x' in results_trial.columns: # Check if sigma data exists
                        for sigma_idx in range(9): # Assuming 9 sigma points
                            col_x_name = f'sigmas_cartesian_transformed_{sigma_idx}_x'
                            col_y_name = f'sigmas_cartesian_transformed_{sigma_idx}_y'
                            if col_x_name in results_trial.columns and col_y_name in results_trial.columns:
                                all_sigmas_x_for_trial.append(results_trial[col_x_name].tolist())
                                all_sigmas_y_for_trial.append(results_trial[col_y_name].tolist())
                            else:
                                # Append list of NaNs if a column is missing, to maintain structure
                                all_sigmas_x_for_trial.append([np.nan] * n_steps_this_trial)
                                all_sigmas_y_for_trial.append([np.nan] * n_steps_this_trial)
                    else:
                        # If no sigma data at all, create empty structures or fill with NaNs
                        for _ in range(9):
                            all_sigmas_x_for_trial.append([np.nan] * n_steps_this_trial)
                            all_sigmas_y_for_trial.append([np.nan] * n_steps_this_trial)

                # print(results['step'][results['trial'] == trial].steps)
                for trial_step in range(n_steps_this_trial):
                    time = results_trial['time_run'].iloc[trial_step] if c.task_type == 'bouncing' else results_trial['time'].iloc[trial_step]
                    label = f"Trial {trial} time {time:.3f}"
                    frame_times.append(label)
                    # print(f"Step: {trial_step}")
                    i += 1
                    # Subplot 1 data
                    circle_x1, circle_y1 = create_circle(
                        results_trial['target_x'].iloc[trial_step],
                        results_trial['target_y'].iloc[trial_step],
                        results_trial['r_target'].iloc[trial_step])
                    # circle_x2, circle_y2 = create_circle(
                    #     results_trial['vis_target_x'].iloc[trial_step],
                    #     results_trial['vis_target_y'].iloc[trial_step],
                    #     results_trial['r_target'].iloc[trial_step])


                    current_endpoints = endpoints[
                        # (endpoints['run'] == run) &
                        # (endpoints['trial'] == trial) &
                        (endpoints['row_index'] <= results_trial.index[trial_step])
                    ]

                    # Calculate Posterior Hand Uncertainty Ellipse for Subplot 2
                    ellipse_x_coords, ellipse_y_coords = np.array([]), np.array([]) # Default to empty
                    
                    hand_posterior_mu_x = results_trial['posterior_hand_x'].iloc[trial_step]
                    hand_posterior_mu_y = results_trial['posterior_hand_y'].iloc[trial_step]
                    hand_posterior_mu = np.array([hand_posterior_mu_x, hand_posterior_mu_y])

                    P_cart_est_col_name = 'P_est_cartesian_ukf'
                    if P_cart_est_col_name in results_trial.columns:
                        P_cart_est_matrix_full = results_trial[P_cart_est_col_name].iloc[trial_step]
                        
                        # Check if it's a valid 6x6 numpy array
                        if isinstance(P_cart_est_matrix_full, np.ndarray) and P_cart_est_matrix_full.shape == (6, 6):
                            # Extract the 2x2 submatrix for hand position (x, y)
                            covariance_matrix_hand = P_cart_est_matrix_full[0:2, 0:2]
                            
                            if not np.isnan(hand_posterior_mu).any() and not np.isnan(covariance_matrix_hand).any():
                                ellipse_x_coords, ellipse_y_coords = calculate_ellipse_points(
                                    hand_posterior_mu,
                                    covariance_matrix_hand,
                                    n_std=1.96 # For 95% credible interval
                                )
                            else: # Optional: log if not a 6x6 ndarray
                                print(f"Warning: {P_cart_est_col_name} is not a 6x6 ndarray at step {trial_step}. Shape: {getattr(P_cart_est_matrix_full, 'shape', 'N/A')}")
                        else: # Optional: log if column not found
                            print(f"Warning: Column {P_cart_est_col_name} not found at step {trial_step} for ellipse.")

                    # Cartesian sigma points - access pre-extracted data
                    if c.plot_sigma_points:
                        x_cart_sigmas = [all_sigmas_x_for_trial[j][trial_step] for j in range(len(all_sigmas_x_for_trial))]
                        y_cart_sigmas = [all_sigmas_y_for_trial[j][trial_step] for j in range(len(all_sigmas_y_for_trial))]
                    else:
                        x_cart_sigmas = []
                        y_cart_sigmas = []

                    # Always show all visual feedback points collected so far in this trial,
                    # regardless of whether the current step has visual feedback available.
                    vis_mask = results_trial['visual_feedback'].iloc[:trial_step + 1]
                    x_vis_feedback = results_trial['vis_hand_x'].iloc[:trial_step + 1][vis_mask]
                    y_vis_feedback = results_trial['vis_hand_y'].iloc[:trial_step + 1][vis_mask]
                    if len(x_vis_feedback) == 0:
                        x_vis_feedback = [np.nan]
                        y_vis_feedback = [np.nan]

                    frames.append(go.Frame(
                        data=[
                            # Shoulder both subplots
                            go.Scatter(
                                x=[results['true_shoulder_x'].iloc[0]],
                                y=[results['true_shoulder_y'].iloc[0]],
                            ),
                            go.Scatter(
                                x=[results['true_shoulder_x'].iloc[0]],
                                y=[results['true_shoulder_y'].iloc[0]],
                            ),
                            
                            # Subplot 1 Traces
                            # Lower arm
                            go.Scatter(
                                x=[results_trial['true_elbow_x'].iloc[trial_step],
                                   results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_elbow_y'].iloc[trial_step],
                                   results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Upper arm
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['true_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['true_elbow_y'].iloc[trial_step]],
                            ),
                            # Hand previous positions
                            go.Scatter(
                                x=results_trial['true_hand_x'].iloc[:trial_step],
                                y=results_trial['true_hand_y'].iloc[:trial_step],
                            ),
                            # planned trajectory
                            go.Scatter(
                                x=results_trial['p_planned_trajectory_x'].iloc[:trial_step],
                                y=results_trial['p_planned_trajectory_y'].iloc[:trial_step],
                            ),
                            
                            # Hand current position
                            go.Scatter(
                                x=[results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Target circle
                            go.Scatter(
                                x=circle_x1,
                                y=circle_y1,
                            ),
                            go.Scatter(
                                x=current_endpoints['true_hand_x'],
                                y=current_endpoints['true_hand_y'],
                            ),
                            # Visual feedback
                            go.Scatter(
                                x=x_vis_feedback,
                                y=y_vis_feedback,
                            ),
                            # Subplot 2 Traces
                            # Posterior Hand Uncertainty Ellipse
                            go.Scatter(
                                x=ellipse_x_coords,
                                y=ellipse_y_coords,
                            ),
                            # Cartesian sigma points
                            go.Scatter(
                                x=x_cart_sigmas,
                                y=y_cart_sigmas,
                            ),
                            # Lower arm
                            go.Scatter(
                                x=[results_trial['posterior_elbow_x'].iloc[trial_step],
                                   results_trial['posterior_hand_x'].iloc[trial_step]],
                                y=[results_trial['posterior_elbow_y'].iloc[trial_step],
                                   results_trial['posterior_hand_y'].iloc[trial_step]],
                            ),
                            # Upper arm
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['posterior_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['posterior_elbow_y'].iloc[trial_step]],
                            ),
                            # Lower arm true
                            go.Scatter(
                                x=[results_trial['true_elbow_x'].iloc[trial_step],
                                   results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_elbow_y'].iloc[trial_step],
                                   results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Upper arm true
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['true_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['true_elbow_y'].iloc[trial_step]],
                            ),
                            # Hand previous positions
                            go.Scatter(
                                x=results_trial['posterior_hand_x'].iloc[:trial_step],
                                y=results_trial['posterior_hand_y'].iloc[:trial_step],
                            ),
                            # planned trajectory
                            go.Scatter(
                                x=results_trial['p_planned_trajectory_x'].iloc[:trial_step],
                                y=results_trial['p_planned_trajectory_y'].iloc[:trial_step],
                            ),
                            # Hand current position
                            go.Scatter(
                                x=[results_trial['posterior_hand_x'].iloc[trial_step]],
                                y=[results_trial['posterior_hand_y'].iloc[trial_step]],
                            ),
                            # Target circle
                            go.Scatter(
                                x=circle_x1,
                                y=circle_y1,
                            ),
                            go.Scatter(
                                x=current_endpoints['posterior_hand_x'],
                                y=current_endpoints['posterior_hand_y'],
                            ),
                            # Visual feedback
                            go.Scatter(
                                x=x_vis_feedback,
                                y=y_vis_feedback,
                            ),

                        ],
                        name=str(i)
                    ))
                    pbar.update(1)

    # Assign frames to the figure
    fig.frames = frames

    print(f"Total frames: {len(frames)}")

    ### Create Slider Steps ###
    slider_steps = []
    # for i in tqdm(range(len(frames)), desc="Combining frames"):
    for i in range(len(frames)):
        slider_step = dict(
            method='animate',
            label=frame_times[i],
            args=[
                [str(i)],
                dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0)
                )
            ],
        )
        slider_steps.append(slider_step)

    # Create the slider
    sliders = [dict(
        active=0,
        currentvalue={'prefix': 'Time: '},
        pad={'t': 50},
        steps=slider_steps
    )]

    ### Set Axis Limits and Update Axes for Each Subplot ###

    # Margin for both subplots
    margin = .1

    # Subplot 1 Axis Ranges
    x_min = -.5
    x_max = .5
    y_min = -.3
    y_max = .7

    # Update axes for Subplot 1
    fig.update_xaxes(
        range=[x_min, x_max],
        title_text='X Position',
        scaleanchor='y',
        scaleratio=1,
        constrain='domain',
        row=1, col=1
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        title_text='Y Position',
        constrain='domain',
        row=1, col=1
    )

    # Update axes for Subplot 2
    fig.update_xaxes(
        range=[x_min, x_max],
        title_text='X Position',
        scaleanchor='y2',
        scaleratio=1,
        constrain='domain',
        row=1, col=2
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        title_text='',
        constrain='domain',
        row=1, col=2
    )

    ### Update Figure Layout ###
    fig.update_layout(
        width=1000,
        height=600,
        autosize=False,
        updatemenus=[
            {
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'redraw': False},
                            'transition': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'easing': 'linear'},
                            'fromcurrent': True,
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'showactive': False,
                'y': -0.1,
                'x': -0.01
            }
        ],
        sliders=sliders,
        showlegend=True,
    )

    # Display the animation
    if show_fig:
        fig.show(config={'responsive': False})

    file_name = 'outputs/plotly_animation.html' if output_filename == None else output_filename + '.html'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    fig.write_html(file_name)
    print(
        f'Plotly animated plot created with frame_decimation set to {frame_decimation}\n{len(results)} of {org_len} steps animated\nAnimation saved to: {file_name}')

def plotly_animation_kin(results, show_fig=False, frame_decimation=4, annimation_speedup=1, output_filename=None, extra_text=None, file_type = None):
    org_len = len(results)
    if org_len == 0:
        print("Warning: results DataFrame is empty. Cannot generate plotly_animation_kin.")
        return
    substract_len = org_len % frame_decimation
    results.drop(results.index[(org_len-substract_len):], inplace=True)
    results = results.iloc[::frame_decimation].copy()
    if results.empty:
        print(f"Warning: results DataFrame is empty after decimation with factor {frame_decimation}. Cannot generate plotly_animation_kin.")
        return

    upper_arm_line_true = {'width': 15, 'color': 'steelblue'}
    lower_arm_line_true = {'width': 15, 'color': 'steelblue'}
    upper_arm_marker_true = {'size': 15, 'color': 'steelblue'}
    lower_arm_marker_true = {'size': 15, 'color': 'steelblue'}

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[results['true_shoulder_x'].iloc[0]],
        y=[results['true_shoulder_y'].iloc[0]],
        mode='markers',
        marker=upper_arm_marker_true,
        name='Shoulder',
        legendgroup='true_arm',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lower_arm_line_true,
        marker=lower_arm_marker_true,
        name='Lower arm',
        legendgroup='true_arm',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=upper_arm_line_true,
        marker=upper_arm_marker_true,
        name='Upper arm',
        legendgroup='true_arm',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='dodgerblue', opacity=0.5),
        name='Hand trajectory',
        legendgroup='true_trajectory',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='black'),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Target',
        legendgroup='target',
        showlegend=True
    ))

    frames = []
    i = -1
    total_iterations = len(results)
    runs = results['run'].unique()
    frame_times = []
    cumulative_hand_x = []
    cumulative_hand_y = []
    with tqdm(total=total_iterations, desc="Animation progress") as pbar:
        for run in runs:
            trials = results[results['run'] == run]['trial'].unique()

            for trial in trials:
                n_steps_this_trial = len(
                    results[(results['trial'] == trial) & (results['run'] == run)])
                results_trial = results[(results['trial'] == trial) & (
                    results['run'] == run)]

                for trial_step in range(n_steps_this_trial):
                    time = results_trial['time_run'].iloc[trial_step] if c.task_type == 'bouncing' else results_trial['time'].iloc[trial_step]
                    label = f"Trial {trial} time {time:.3f}"
                    frame_times.append(label)
                    i += 1
                    circle_x1, circle_y1 = create_circle(
                        results_trial['target_x'].iloc[trial_step],
                        results_trial['target_y'].iloc[trial_step],
                        results_trial['r_target'].iloc[trial_step])

                    frames.append(go.Frame(
                        data=[
                            go.Scatter(
                                x=[results['true_shoulder_x'].iloc[0]],
                                y=[results['true_shoulder_y'].iloc[0]],
                            ),
                            go.Scatter(
                                x=[results_trial['true_elbow_x'].iloc[trial_step],
                                   results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_elbow_y'].iloc[trial_step],
                                   results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['true_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['true_elbow_y'].iloc[trial_step]],
                            ),
                            go.Scatter(
                                x=list(cumulative_hand_x),
                                y=list(cumulative_hand_y),
                            ),
                            go.Scatter(
                                x=circle_x1,
                                y=circle_y1,
                            ),
                        ],
                        name=str(i)
                    ))
                    cumulative_hand_x.append(results_trial['true_hand_x'].iloc[trial_step])
                    cumulative_hand_y.append(results_trial['true_hand_y'].iloc[trial_step])
                    pbar.update(1)

    fig.frames = frames

    print(f"Total frames: {len(frames)}")

    slider_steps = []
    for i in range(len(frames)):
        slider_step = dict(
            method='animate',
            label=frame_times[i],
            args=[
                [str(i)],
                dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0)
                )
            ],
        )
        slider_steps.append(slider_step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': 'Time: '},
        pad={'t': 50},
        steps=slider_steps
    )]

    x_min = -.5
    x_max = .5
    y_min = -.3
    y_max = .7

    fig.update_xaxes(
        range=[x_min, x_max],
        title_text='X Position',
        scaleanchor='y',
        scaleratio=1,
        constrain='domain',
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        title_text='Y Position',
        constrain='domain',
    )

    fig.update_layout(
        title='True hand movement',
        width=600,
        height=600,
        autosize=False,
        updatemenus=[
            {
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'redraw': False},
                            'transition': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'easing': 'linear'},
                            'fromcurrent': True,
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'showactive': False,
                'y': -0.1,
                'x': -0.01
            }
        ],
        sliders=sliders,
        showlegend=True,
    )

    if show_fig:
        fig.show(config={'responsive': False})

    file_name = 'outputs/plotly_animation_kin.html' if output_filename == None else output_filename + '.html'

    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    fig.write_html(file_name)
    print(
        f'Plotly kinematic animation created with frame_decimation set to {frame_decimation}\n{len(results)} of {org_len} steps animated\nAnimation saved to: {file_name}')

def plotly_animation2(results, show_fig=False, frame_decimation=1, annimation_speedup=.5, output_filename=None, extra_text=None, file_type = None):
    org_len = len(results)
    if org_len == 0:
        print("Warning: results DataFrame is empty. Cannot generate plotly_animation.")
        return
    substract_len = org_len % frame_decimation
    results.drop(results.index[(org_len-substract_len):], inplace=True)
    results = results.iloc[::frame_decimation].copy()
    if results.empty:
        print(f"Warning: results DataFrame is empty after decimation with factor {frame_decimation}. Cannot generate plotly_animation.")
        return
    results['row_index'] = results.index
    endpoints = results.groupby(['run', 'trial']).tail(1)
    if c.elbow_down:
        shoulder_marker = {'size': 20, 'color': 'peru'}
        upper_arm_line = {'width': 16, 'color': 'darkorange'}
        lower_arm_line = {'width': 12, 'color': 'orange'}
        upper_arm_marker = {'size': 16, 'color': 'darkorange'}
        lower_arm_marker = {'size': 12, 'color': 'orange'}
    else:
        shoulder_marker = {'size': 15, 'color': 'darkorange'}
        upper_arm_line = {'width': 15, 'color': 'darkorange'}
        lower_arm_line = {'width': 15, 'color': 'darkorange'}
        upper_arm_marker = {'size': 15, 'color': 'darkorange'}
        lower_arm_marker = {'size': 15, 'color': 'darkorange'}

    # print(results['visual_feedback'])

    lp_upper_arm_line_true = {'width': 15, 'color': 'steelblue'}
    lp_lower_arm_line_true = {'width': 15, 'color': 'steelblue'}
    lp_upper_arm_marker_true = {'size': 15, 'color': 'steelblue'}
    lp_lower_arm_marker_true = {'size': 15, 'color': 'steelblue'}
    rp_upper_arm_line_true = {'width': 5, 'color': 'steelblue'}
    rp_lower_arm_line_true = {'width': 5, 'color': 'steelblue'}
    rp_upper_arm_marker_true = {'size': 5, 'color': 'steelblue'}
    rp_lower_arm_marker_true = {'size': 5, 'color': 'steelblue'}


    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('', ''),
        horizontal_spacing=0.05,
        column_widths=[0.75, 0.25]
    )
    # shoulder both subplots
    fig.add_trace(go.Scatter(
        x=[results['true_shoulder_x'].iloc[0]],
        y=[results['true_shoulder_y'].iloc[0]],
        mode='markers',
        marker=lp_upper_arm_marker_true,
        name='Shoulder',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[results['true_shoulder_x'].iloc[0]],
        y=[results['true_shoulder_y'].iloc[0]],
        mode='markers',
        marker=lp_upper_arm_marker_true,
        name='Shoulder',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=2)

    ### Add initial traces to Subplot 1 ###

    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_lower_arm_line_true,
        marker=lp_lower_arm_marker_true,
        name='Lower arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_upper_arm_line_true,
        marker=lp_upper_arm_marker_true,
        name='Upper arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    # Previous Positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='dodgerblue', opacity=0.5),
        name='Previous positions',
        legendgroup='true_trajectory',
        showlegend=True
    ), row=1, col=1)
    # planned trajectory
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.5),
        name='Posterior position',
        legendgroup='planned_trajectory',
        showlegend=True
    ), row=1, col=1)
    # Current Position
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=6, color='dodgerblue'),
        name='Current position',
        legendgroup='true_trajectory',
        showlegend=True
    ), row=1, col=1)

    # Target Circle
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='black'),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Target',
        legendgroup='target',
        showlegend=True
    ), row=1, col=1)

    # Input positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='darkblue', opacity=0.5),
        name='Input position',
        legendgroup='input_position',
        showlegend=True
    ), row=1, col=1)
    # Visual feedback
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.3),
        name='Visual feedback',
        legendgroup='visual_feedback',
        showlegend=True
    ), row=1, col=1)

    ### Add initial traces to Subplot 2 (zoomed-in copy of Subplot 1) ###
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_lower_arm_line_true,
        marker=lp_lower_arm_marker_true,
        name='Lower arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_upper_arm_line_true,
        marker=lp_upper_arm_marker_true,
        name='Upper arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=2)
    # Previous Positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='dodgerblue', opacity=0.5),
        name='Previous positions',
        legendgroup='true_trajectory',
        showlegend=False
    ), row=1, col=2)
    # planned trajectory
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.5),
        name='Posterior position',
        legendgroup='planned_trajectory',
        showlegend=False
    ), row=1, col=2)
    # Current Position
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=6, color='dodgerblue'),
        name='Current position',
        legendgroup='true_trajectory',
        showlegend=False
    ), row=1, col=2)
    # Target Circle
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='black'),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Target',
        legendgroup='target',
        showlegend=False
    ), row=1, col=2)
    # Input positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='darkblue', opacity=0.5),
        name='Input position',
        legendgroup='input_position',
        showlegend=False
    ), row=1, col=2)
    # Visual feedback
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.3),
        name='Visual feedback',
        legendgroup='visual_feedback',
        showlegend=False
    ), row=1, col=2)
    # Body

    if c.task_type == "patterson2017":
        # Target out
        target_out_x, target_out_y = create_circle(
            results['p_target_out_x'].iloc[0],
            results['p_target_out_y'].iloc[0],
            results['r_target_out'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_out_x,
            y=target_out_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_out_x,
            y=target_out_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        # Target home
        target_home_x, target_home_y = create_circle(
            results['p_target_home_x'].iloc[0],
            results['p_target_home_y'].iloc[0],
            results['r_target_home'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_home_x,
            y=target_home_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target home',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_home_x,
            y=target_home_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        # Target final
        target_final_x, target_final_y = create_circle(
            results['p_target_final_x'].iloc[0],
            results['p_target_final_y'].iloc[0],
            results['r_target_out'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_final_x,
            y=target_final_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target final',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_final_x,
            y=target_final_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        
        

    elif c.task_type == "circular_following":
        outer_path_x, outer_path_y = create_circle(
            results['circular_target_movement_c_x'].iloc[0],
            results['circular_target_movement_c_y'].iloc[0],
            results['circular_target_movement_r'].iloc[0] + results['r_target'].iloc[0],
            num_points=72)

        inner_path_x, inner_path_y = create_circle(
            results['circular_target_movement_c_x'].iloc[0],
            results['circular_target_movement_c_y'].iloc[0],
            results['circular_target_movement_r'].iloc[0] - results['r_target'].iloc[0],
            num_points=72)

        fig.add_trace(go.Scatter(
            x=outer_path_x,
            y=outer_path_y,
            mode='lines',
            # line=dict(color='grey'),
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=inner_path_x,
            y=inner_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=outer_path_x,
            y=outer_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=inner_path_x,
            y=inner_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=1)

    frames = []
    i = -1

    total_iterations = len(results)
    # print(f"Total iterations: {total_iterations}")
    runs = results['run'].unique()
    frame_times = []
    # print(f"Total runs: {n_runs}")
    with tqdm(total=total_iterations, desc="Animation progress") as pbar:
        for run in runs:
            # print(f"Run: {run}")
            trials = results[results['run'] == run]['trial'].unique()

            for trial in trials:
                n_steps_this_trial = len(
                    results[(results['trial'] == trial) & (results['run'] == run)])
                results_trial = results[(results['trial'] == trial) & (
                    results['run'] == run)]
                
                # Pre-extract all sigma point data for the current trial
                all_sigmas_x_for_trial = []
                all_sigmas_y_for_trial = []
                if c.plot_sigma_points:
                    if 'sigmas_cartesian_transformed_0_x' in results_trial.columns: # Check if sigma data exists
                        for sigma_idx in range(9): # Assuming 9 sigma points
                            col_x_name = f'sigmas_cartesian_transformed_{sigma_idx}_x'
                            col_y_name = f'sigmas_cartesian_transformed_{sigma_idx}_y'
                            if col_x_name in results_trial.columns and col_y_name in results_trial.columns:
                                all_sigmas_x_for_trial.append(results_trial[col_x_name].tolist())
                                all_sigmas_y_for_trial.append(results_trial[col_y_name].tolist())
                            else:
                                # Append list of NaNs if a column is missing, to maintain structure
                                all_sigmas_x_for_trial.append([np.nan] * n_steps_this_trial)
                                all_sigmas_y_for_trial.append([np.nan] * n_steps_this_trial)
                    else:
                        # If no sigma data at all, create empty structures or fill with NaNs
                        for _ in range(9):
                            all_sigmas_x_for_trial.append([np.nan] * n_steps_this_trial)
                            all_sigmas_y_for_trial.append([np.nan] * n_steps_this_trial)

                # print(results['step'][results['trial'] == trial].steps)
                for trial_step in range(n_steps_this_trial):
                    time = results_trial['time_run'].iloc[trial_step] if c.task_type == 'bouncing' else results_trial['time'].iloc[trial_step]
                    label = f"Trial {trial} time {time:.3f}"
                    frame_times.append(label)
                    # print(f"Step: {trial_step}")
                    i += 1
                    # Subplot 1 data
                    circle_x1, circle_y1 = create_circle(
                        results_trial['target_x'].iloc[trial_step],
                        results_trial['target_y'].iloc[trial_step],
                        results_trial['r_target'].iloc[trial_step])
                    # circle_x2, circle_y2 = create_circle(
                    #     results_trial['vis_target_x'].iloc[trial_step],
                    #     results_trial['vis_target_y'].iloc[trial_step],
                    #     results_trial['r_target'].iloc[trial_step])


                    current_endpoints = endpoints[
                        # (endpoints['run'] == run) &
                        # (endpoints['trial'] == trial) &
                        (endpoints['row_index'] <= results_trial.index[trial_step])
                    ]

                    # Calculate Posterior Hand Uncertainty Ellipse for Subplot 2
                    ellipse_x_coords, ellipse_y_coords = np.array([]), np.array([]) # Default to empty
                    
                    hand_posterior_mu_x = results_trial['posterior_hand_x'].iloc[trial_step]
                    hand_posterior_mu_y = results_trial['posterior_hand_y'].iloc[trial_step]
                    hand_posterior_mu = np.array([hand_posterior_mu_x, hand_posterior_mu_y])

                    P_cart_est_col_name = 'P_est_cartesian_ukf'
                    if P_cart_est_col_name in results_trial.columns:
                        P_cart_est_matrix_full = results_trial[P_cart_est_col_name].iloc[trial_step]
                        
                        # Check if it's a valid 6x6 numpy array
                        if isinstance(P_cart_est_matrix_full, np.ndarray) and P_cart_est_matrix_full.shape == (6, 6):
                            # Extract the 2x2 submatrix for hand position (x, y)
                            covariance_matrix_hand = P_cart_est_matrix_full[0:2, 0:2]
                            
                            if not np.isnan(hand_posterior_mu).any() and not np.isnan(covariance_matrix_hand).any():
                                ellipse_x_coords, ellipse_y_coords = calculate_ellipse_points(
                                    hand_posterior_mu,
                                    covariance_matrix_hand,
                                    n_std=1.96 # For 95% credible interval
                                )
                            else: # Optional: log if not a 6x6 ndarray
                                print(f"Warning: {P_cart_est_col_name} is not a 6x6 ndarray at step {trial_step}. Shape: {getattr(P_cart_est_matrix_full, 'shape', 'N/A')}")
                        else: # Optional: log if column not found
                            print(f"Warning: Column {P_cart_est_col_name} not found at step {trial_step} for ellipse.")

                    # Cartesian sigma points - access pre-extracted data
                    if c.plot_sigma_points:
                        x_cart_sigmas = [all_sigmas_x_for_trial[j][trial_step] for j in range(len(all_sigmas_x_for_trial))]
                        y_cart_sigmas = [all_sigmas_y_for_trial[j][trial_step] for j in range(len(all_sigmas_y_for_trial))]
                    else:
                        x_cart_sigmas = []
                        y_cart_sigmas = []

                    # Always show all visual feedback points collected so far in this trial,
                    # regardless of whether the current step has visual feedback available.
                    vis_mask = results_trial['visual_feedback'].iloc[:trial_step + 1]
                    x_vis_feedback = results_trial['vis_hand_x'].iloc[:trial_step + 1][vis_mask]
                    y_vis_feedback = results_trial['vis_hand_y'].iloc[:trial_step + 1][vis_mask]
                    if len(x_vis_feedback) == 0:
                        x_vis_feedback = [np.nan]
                        y_vis_feedback = [np.nan]

                    frames.append(go.Frame(
                        data=[
                            # Shoulder both subplots
                            go.Scatter(
                                x=[results['true_shoulder_x'].iloc[0]],
                                y=[results['true_shoulder_y'].iloc[0]],
                            ),
                            go.Scatter(
                                x=[results['true_shoulder_x'].iloc[0]],
                                y=[results['true_shoulder_y'].iloc[0]],
                            ),
                            
                            # Subplot 1 Traces
                            # Lower arm
                            go.Scatter(
                                x=[results_trial['true_elbow_x'].iloc[trial_step],
                                   results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_elbow_y'].iloc[trial_step],
                                   results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Upper arm
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['true_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['true_elbow_y'].iloc[trial_step]],
                            ),
                            # Hand previous positions
                            go.Scatter(
                                x=results_trial['true_hand_x'].iloc[:trial_step],
                                y=results_trial['true_hand_y'].iloc[:trial_step],
                            ),
                            # Posterior position
                            go.Scatter(
                                x=results_trial['posterior_hand_x'].iloc[:trial_step],
                                y=results_trial['posterior_hand_y'].iloc[:trial_step],
                            ),
                            
                            # Hand current position
                            go.Scatter(
                                x=[results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Target circle
                            go.Scatter(
                                x=circle_x1,
                                y=circle_y1,
                            ),
                            go.Scatter(
                                x=current_endpoints['true_hand_x'],
                                y=current_endpoints['true_hand_y'],
                            ),
                            # Visual feedback
                            go.Scatter(
                                x=x_vis_feedback,
                                y=y_vis_feedback,
                            ),
                            # Subplot 2 Traces (zoomed-in copy of Subplot 1)
                            # Lower arm
                            go.Scatter(
                                x=[results_trial['true_elbow_x'].iloc[trial_step],
                                   results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_elbow_y'].iloc[trial_step],
                                   results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Upper arm
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['true_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['true_elbow_y'].iloc[trial_step]],
                            ),
                            # Hand previous positions
                            go.Scatter(
                                x=results_trial['true_hand_x'].iloc[:trial_step],
                                y=results_trial['true_hand_y'].iloc[:trial_step],
                            ),
                            # planned trajectory
                            go.Scatter(
                                x=results_trial['p_planned_trajectory_x'].iloc[:trial_step],
                                y=results_trial['p_planned_trajectory_y'].iloc[:trial_step],
                            ),
                            # Hand current position
                            go.Scatter(
                                x=[results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Target circle
                            go.Scatter(
                                x=circle_x1,
                                y=circle_y1,
                            ),
                            # Endpoints
                            go.Scatter(
                                x=current_endpoints['true_hand_x'],
                                y=current_endpoints['true_hand_y'],
                            ),
                            # Visual feedback
                            go.Scatter(
                                x=x_vis_feedback,
                                y=y_vis_feedback,
                            ),

                        ],
                        name=str(i)
                    ))
                    pbar.update(1)

    # Assign frames to the figure
    fig.frames = frames

    print(f"Total frames: {len(frames)}")

    ### Create Slider Steps ###
    slider_steps = []
    # for i in tqdm(range(len(frames)), desc="Combining frames"):
    for i in range(len(frames)):
        slider_step = dict(
            method='animate',
            label=frame_times[i],
            args=[
                [str(i)],
                dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0)
                )
            ],
        )
        slider_steps.append(slider_step)

    # Create the slider
    sliders = [dict(
        active=0,
        currentvalue={'prefix': 'Time: '},
        pad={'t': 50},
        steps=slider_steps
    )]

    ### Set Axis Limits and Update Axes for Each Subplot ###

    # Margin for both subplots
    margin = .1

    # Subplot 1 Axis Ranges
    x_min = -.5
    x_max = .5
    y_min = -.3
    y_max = .7

    # Update axes for Subplot 1
    fig.update_xaxes(
        range=[x_min, x_max],
        title_text='X Position',
        scaleanchor='y',
        scaleratio=1,
        constrain='domain',
        row=1, col=1
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        title_text='Y Position',
        constrain='domain',
        row=1, col=1
    )

    # Update axes for Subplot 2 (zoomed-in, tall/narrow view of Subplot 1)
    zoom_x_min = -0.05
    zoom_x_max = 0.05
    zoom_y_min = -0.05
    zoom_y_max = 0.25
    fig.update_xaxes(
        range=[zoom_x_min, zoom_x_max],
        title_text='X Position',
        scaleanchor='y2',
        scaleratio=1,
        constrain='domain',
        row=1, col=2
    )
    fig.update_yaxes(
        range=[zoom_y_min, zoom_y_max],
        title_text='',
        constrain='domain',
        row=1, col=2
    )

    ### Update Figure Layout ###
    fig.update_layout(
        width=800,
        height=600,
        autosize=False,
        updatemenus=[
            {
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'redraw': False},
                            'transition': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'easing': 'linear'},
                            'fromcurrent': True,
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'showactive': False,
                'y': -0.1,
                'x': -0.01
            }
        ],
        sliders=sliders,
        showlegend=True,
    )

    # Display the animation
    if show_fig:
        fig.show(config={'responsive': False})

    file_name = 'outputs/plotly_animation.html' if output_filename == None else output_filename + '.html'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    fig.write_html(file_name)
    print(
        f'Plotly animated plot created with frame_decimation set to {frame_decimation}\n{len(results)} of {org_len} steps animated\nAnimation saved to: {file_name}')


def plotly_animation3(results, show_fig=False, frame_decimation=1, annimation_speedup=.5, output_filename=None, extra_text=None, file_type = None):
    org_len = len(results)
    if org_len == 0:
        print("Warning: results DataFrame is empty. Cannot generate plotly_animation.")
        return
    substract_len = org_len % frame_decimation
    results.drop(results.index[(org_len-substract_len):], inplace=True)
    results = results.iloc[::frame_decimation].copy()
    if results.empty:
        print(f"Warning: results DataFrame is empty after decimation with factor {frame_decimation}. Cannot generate plotly_animation.")
        return
    results['row_index'] = results.index
    endpoints = results.groupby(['run', 'trial']).tail(1)
    if c.elbow_down:
        shoulder_marker = {'size': 20, 'color': 'peru'}
        upper_arm_line = {'width': 16, 'color': 'darkorange'}
        lower_arm_line = {'width': 12, 'color': 'orange'}
        upper_arm_marker = {'size': 16, 'color': 'darkorange'}
        lower_arm_marker = {'size': 12, 'color': 'orange'}
    else:
        shoulder_marker = {'size': 15, 'color': 'darkorange'}
        upper_arm_line = {'width': 15, 'color': 'darkorange'}
        lower_arm_line = {'width': 15, 'color': 'darkorange'}
        upper_arm_marker = {'size': 15, 'color': 'darkorange'}
        lower_arm_marker = {'size': 15, 'color': 'darkorange'}

    # print(results['visual_feedback'])

    lp_upper_arm_line_true = {'width': 15, 'color': 'steelblue'}
    lp_lower_arm_line_true = {'width': 15, 'color': 'steelblue'}
    lp_upper_arm_marker_true = {'size': 15, 'color': 'steelblue'}
    lp_lower_arm_marker_true = {'size': 15, 'color': 'steelblue'}
    rp_upper_arm_line_true = {'width': 5, 'color': 'steelblue'}
    rp_lower_arm_line_true = {'width': 5, 'color': 'steelblue'}
    rp_upper_arm_marker_true = {'size': 5, 'color': 'steelblue'}
    rp_lower_arm_marker_true = {'size': 5, 'color': 'steelblue'}


    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('', ''),
        horizontal_spacing=0.05,
        column_widths=[0.50,0.50]
    )
    # shoulder both subplots
    fig.add_trace(go.Scatter(
        x=[results['true_shoulder_x'].iloc[0]],
        y=[results['true_shoulder_y'].iloc[0]],
        mode='markers',
        marker=lp_upper_arm_marker_true,
        name='Shoulder',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[results['true_shoulder_x'].iloc[0]],
        y=[results['true_shoulder_y'].iloc[0]],
        mode='markers',
        marker=lp_upper_arm_marker_true,
        name='Shoulder',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=2)

    ### Add initial traces to Subplot 1 ###

    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_lower_arm_line_true,
        marker=lp_lower_arm_marker_true,
        name='Lower arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_upper_arm_line_true,
        marker=lp_upper_arm_marker_true,
        name='Upper arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=1)
    # Previous Positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='dodgerblue', opacity=0.5),
        name='Previous positions',
        legendgroup='true_trajectory',
        showlegend=True
    ), row=1, col=1)
    # planned trajectory
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.5),
        name='Posterior position',
        legendgroup='planned_trajectory',
        showlegend=True
    ), row=1, col=1)
    # Current Position
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=6, color='dodgerblue'),
        name='Current position',
        legendgroup='true_trajectory',
        showlegend=True
    ), row=1, col=1)

    # Target Circle
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='black'),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Target',
        legendgroup='target',
        showlegend=True
    ), row=1, col=1)

    # Input positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='darkblue', opacity=0.5),
        name='Input position',
        legendgroup='input_position',
        showlegend=True
    ), row=1, col=1)
    # Visual feedback
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.3),
        name='Visual feedback',
        legendgroup='visual_feedback',
        showlegend=True
    ), row=1, col=1)

    ### Add initial traces to Subplot 2 (zoomed-in copy of Subplot 1) ###
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_lower_arm_line_true,
        marker=lp_lower_arm_marker_true,
        name='Lower arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        line=lp_upper_arm_line_true,
        marker=lp_upper_arm_marker_true,
        name='Upper arm',
        legendgroup='true_arm',
        showlegend=False
    ), row=1, col=2)
    # Previous Positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='dodgerblue', opacity=0.5),
        name='Previous positions',
        legendgroup='true_trajectory',
        showlegend=False
    ), row=1, col=2)
    # planned trajectory
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.5),
        name='Posterior position',
        legendgroup='planned_trajectory',
        showlegend=False
    ), row=1, col=2)
    # Current Position
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=6, color='dodgerblue'),
        name='Current position',
        legendgroup='true_trajectory',
        showlegend=False
    ), row=1, col=2)
    # Target Circle
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='black'),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Target',
        legendgroup='target',
        showlegend=False
    ), row=1, col=2)
    # Input positions
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='darkblue', opacity=0.5),
        name='Input position',
        legendgroup='input_position',
        showlegend=False
    ), row=1, col=2)
    # Visual feedback
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.3),
        name='Visual feedback',
        legendgroup='visual_feedback',
        showlegend=False
    ), row=1, col=2)
    # Body

    if c.task_type == "patterson2017":
        # Target out
        target_out_x, target_out_y = create_circle(
            results['p_target_out_x'].iloc[0],
            results['p_target_out_y'].iloc[0],
            results['r_target_out'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_out_x,
            y=target_out_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_out_x,
            y=target_out_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        # Target home
        target_home_x, target_home_y = create_circle(
            results['p_target_home_x'].iloc[0],
            results['p_target_home_y'].iloc[0],
            results['r_target_home'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_home_x,
            y=target_home_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target home',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_home_x,
            y=target_home_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        # Target final
        target_final_x, target_final_y = create_circle(
            results['p_target_final_x'].iloc[0],
            results['p_target_final_y'].iloc[0],
            results['r_target_out'].iloc[0],
            num_points=72)
        fig.add_trace(go.Scatter(
            x=target_final_x,
            y=target_final_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target final',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=target_final_x,
            y=target_final_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',  # Line color
                width=1,                           # Line width
                dash='dot'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            name='Target out',
            # legendgroup='group1',
            showlegend=False
        ), row=1, col=2)
        
        

    elif c.task_type == "circular_following":
        outer_path_x, outer_path_y = create_circle(
            results['circular_target_movement_c_x'].iloc[0],
            results['circular_target_movement_c_y'].iloc[0],
            results['circular_target_movement_r'].iloc[0] + results['r_target'].iloc[0],
            num_points=72)

        inner_path_x, inner_path_y = create_circle(
            results['circular_target_movement_c_x'].iloc[0],
            results['circular_target_movement_c_y'].iloc[0],
            results['circular_target_movement_r'].iloc[0] - results['r_target'].iloc[0],
            num_points=72)

        fig.add_trace(go.Scatter(
            x=outer_path_x,
            y=outer_path_y,
            mode='lines',
            # line=dict(color='grey'),
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=inner_path_x,
            y=inner_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=outer_path_x,
            y=outer_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=inner_path_x,
            y=inner_path_y,
            mode='lines',
            line=dict(
                color='rgba(128, 128, 128, 0.2)',  # Line color
                width=2,                           # Line width
                # Line style ('solid', 'dot', 'dash', 'longdash', 'dashdot', etc.)
                dash='solid'
            ),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.)',
            name='Target',
            # legendgroup='group2',
            showlegend=False
        ), row=1, col=1)

    frames = []
    i = -1

    total_iterations = len(results)
    # print(f"Total iterations: {total_iterations}")
    runs = results['run'].unique()
    frame_times = []
    # print(f"Total runs: {n_runs}")
    with tqdm(total=total_iterations, desc="Animation progress") as pbar:
        for run in runs:
            # print(f"Run: {run}")
            trials = results[results['run'] == run]['trial'].unique()

            for trial in trials:
                n_steps_this_trial = len(
                    results[(results['trial'] == trial) & (results['run'] == run)])
                results_trial = results[(results['trial'] == trial) & (
                    results['run'] == run)]
                
                # Pre-extract all sigma point data for the current trial
                all_sigmas_x_for_trial = []
                all_sigmas_y_for_trial = []
                if c.plot_sigma_points:
                    if 'sigmas_cartesian_transformed_0_x' in results_trial.columns: # Check if sigma data exists
                        for sigma_idx in range(9): # Assuming 9 sigma points
                            col_x_name = f'sigmas_cartesian_transformed_{sigma_idx}_x'
                            col_y_name = f'sigmas_cartesian_transformed_{sigma_idx}_y'
                            if col_x_name in results_trial.columns and col_y_name in results_trial.columns:
                                all_sigmas_x_for_trial.append(results_trial[col_x_name].tolist())
                                all_sigmas_y_for_trial.append(results_trial[col_y_name].tolist())
                            else:
                                # Append list of NaNs if a column is missing, to maintain structure
                                all_sigmas_x_for_trial.append([np.nan] * n_steps_this_trial)
                                all_sigmas_y_for_trial.append([np.nan] * n_steps_this_trial)
                    else:
                        # If no sigma data at all, create empty structures or fill with NaNs
                        for _ in range(9):
                            all_sigmas_x_for_trial.append([np.nan] * n_steps_this_trial)
                            all_sigmas_y_for_trial.append([np.nan] * n_steps_this_trial)

                # print(results['step'][results['trial'] == trial].steps)
                for trial_step in range(n_steps_this_trial):
                    time = results_trial['time_run'].iloc[trial_step] if c.task_type == 'bouncing' else results_trial['time'].iloc[trial_step]
                    label = f"Trial {trial} time {time:.3f}"
                    frame_times.append(label)
                    # print(f"Step: {trial_step}")
                    i += 1
                    # Subplot 1 data
                    circle_x1, circle_y1 = create_circle(
                        results_trial['target_x'].iloc[trial_step],
                        results_trial['target_y'].iloc[trial_step],
                        results_trial['r_target'].iloc[trial_step])
                    # circle_x2, circle_y2 = create_circle(
                    #     results_trial['vis_target_x'].iloc[trial_step],
                    #     results_trial['vis_target_y'].iloc[trial_step],
                    #     results_trial['r_target'].iloc[trial_step])


                    current_endpoints = endpoints[
                        # (endpoints['run'] == run) &
                        # (endpoints['trial'] == trial) &
                        (endpoints['row_index'] <= results_trial.index[trial_step])
                    ]

                    # Calculate Posterior Hand Uncertainty Ellipse for Subplot 2
                    ellipse_x_coords, ellipse_y_coords = np.array([]), np.array([]) # Default to empty
                    
                    hand_posterior_mu_x = results_trial['posterior_hand_x'].iloc[trial_step]
                    hand_posterior_mu_y = results_trial['posterior_hand_y'].iloc[trial_step]
                    hand_posterior_mu = np.array([hand_posterior_mu_x, hand_posterior_mu_y])

                    P_cart_est_col_name = 'P_est_cartesian_ukf'
                    if P_cart_est_col_name in results_trial.columns:
                        P_cart_est_matrix_full = results_trial[P_cart_est_col_name].iloc[trial_step]
                        
                        # Check if it's a valid 6x6 numpy array
                        if isinstance(P_cart_est_matrix_full, np.ndarray) and P_cart_est_matrix_full.shape == (6, 6):
                            # Extract the 2x2 submatrix for hand position (x, y)
                            covariance_matrix_hand = P_cart_est_matrix_full[0:2, 0:2]
                            
                            if not np.isnan(hand_posterior_mu).any() and not np.isnan(covariance_matrix_hand).any():
                                ellipse_x_coords, ellipse_y_coords = calculate_ellipse_points(
                                    hand_posterior_mu,
                                    covariance_matrix_hand,
                                    n_std=1.96 # For 95% credible interval
                                )
                            else: # Optional: log if not a 6x6 ndarray
                                print(f"Warning: {P_cart_est_col_name} is not a 6x6 ndarray at step {trial_step}. Shape: {getattr(P_cart_est_matrix_full, 'shape', 'N/A')}")
                        else: # Optional: log if column not found
                            print(f"Warning: Column {P_cart_est_col_name} not found at step {trial_step} for ellipse.")

                    # Cartesian sigma points - access pre-extracted data
                    if c.plot_sigma_points:
                        x_cart_sigmas = [all_sigmas_x_for_trial[j][trial_step] for j in range(len(all_sigmas_x_for_trial))]
                        y_cart_sigmas = [all_sigmas_y_for_trial[j][trial_step] for j in range(len(all_sigmas_y_for_trial))]
                    else:
                        x_cart_sigmas = []
                        y_cart_sigmas = []

                    # Show only the visual feedback points for steps where
                    # `visual_feedback` is True (and the sampled position is valid).
                    # Convert to a numpy bool array first, so that boolean masking
                    # works regardless of the source column dtype (bool / object / 0-1 numeric),
                    # otherwise pandas can silently fall back to label-based indexing and
                    # render visual feedback on steps where it should be hidden.
                    vis_mask = (
                        results_trial['visual_feedback']
                        .iloc[:trial_step + 1]
                        .to_numpy()
                        .astype(bool)
                    )
                    vis_hand_x_slice = results_trial['vis_hand_x'].iloc[:trial_step + 1].to_numpy()
                    vis_hand_y_slice = results_trial['vis_hand_y'].iloc[:trial_step + 1].to_numpy()
                    valid_vis = (
                        vis_mask
                        & ~pd.isna(vis_hand_x_slice)
                        & ~pd.isna(vis_hand_y_slice)
                    )
                    x_vis_feedback = vis_hand_x_slice[valid_vis]
                    y_vis_feedback = vis_hand_y_slice[valid_vis]
                    if len(x_vis_feedback) == 0:
                        x_vis_feedback = [np.nan]
                        y_vis_feedback = [np.nan]

                    frames.append(go.Frame(
                        data=[
                            # Shoulder both subplots
                            go.Scatter(
                                x=[results['true_shoulder_x'].iloc[0]],
                                y=[results['true_shoulder_y'].iloc[0]],
                            ),
                            go.Scatter(
                                x=[results['true_shoulder_x'].iloc[0]],
                                y=[results['true_shoulder_y'].iloc[0]],
                            ),
                            
                            # Subplot 1 Traces
                            # Lower arm
                            go.Scatter(
                                x=[results_trial['true_elbow_x'].iloc[trial_step],
                                   results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_elbow_y'].iloc[trial_step],
                                   results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Upper arm
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['true_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['true_elbow_y'].iloc[trial_step]],
                            ),
                            # Hand previous positions
                            go.Scatter(
                                x=results_trial['true_hand_x'].iloc[:trial_step],
                                y=results_trial['true_hand_y'].iloc[:trial_step],
                            ),
                            # Posterior position
                            go.Scatter(
                                x=results_trial['posterior_hand_x'].iloc[:trial_step],
                                y=results_trial['posterior_hand_y'].iloc[:trial_step],
                            ),
                            
                            # Hand current position
                            go.Scatter(
                                x=[results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Target circle
                            go.Scatter(
                                x=circle_x1,
                                y=circle_y1,
                            ),
                            go.Scatter(
                                x=current_endpoints['true_hand_x'],
                                y=current_endpoints['true_hand_y'],
                            ),
                            # Visual feedback
                            go.Scatter(
                                x=x_vis_feedback,
                                y=y_vis_feedback,
                            ),
                            # Subplot 2 Traces (zoomed-in copy of Subplot 1)
                            # Lower arm
                            go.Scatter(
                                x=[results_trial['true_elbow_x'].iloc[trial_step],
                                   results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_elbow_y'].iloc[trial_step],
                                   results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Upper arm
                            go.Scatter(
                                x=[results_trial['true_shoulder_x'].iloc[trial_step],
                                   results_trial['true_elbow_x'].iloc[trial_step]],
                                y=[results_trial['true_shoulder_y'].iloc[trial_step],
                                   results_trial['true_elbow_y'].iloc[trial_step]],
                            ),
                            # Hand previous positions
                            go.Scatter(
                                x=results_trial['true_hand_x'].iloc[:trial_step],
                                y=results_trial['true_hand_y'].iloc[:trial_step],
                            ),
                            # planned trajectory
                            go.Scatter(
                                x=results_trial['p_planned_trajectory_x'].iloc[:trial_step],
                                y=results_trial['p_planned_trajectory_y'].iloc[:trial_step],
                            ),
                            # Hand current position
                            go.Scatter(
                                x=[results_trial['true_hand_x'].iloc[trial_step]],
                                y=[results_trial['true_hand_y'].iloc[trial_step]],
                            ),
                            # Target circle
                            go.Scatter(
                                x=circle_x1,
                                y=circle_y1,
                            ),
                            # Endpoints
                            go.Scatter(
                                x=current_endpoints['true_hand_x'],
                                y=current_endpoints['true_hand_y'],
                            ),
                            # Visual feedback
                            go.Scatter(
                                x=x_vis_feedback,
                                y=y_vis_feedback,
                            ),

                        ],
                        name=str(i)
                    ))
                    pbar.update(1)

    # Assign frames to the figure
    fig.frames = frames

    print(f"Total frames: {len(frames)}")

    ### Create Slider Steps ###
    slider_steps = []
    # for i in tqdm(range(len(frames)), desc="Combining frames"):
    for i in range(len(frames)):
        slider_step = dict(
            method='animate',
            label=frame_times[i],
            args=[
                [str(i)],
                dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0)
                )
            ],
        )
        slider_steps.append(slider_step)

    # Create the slider
    sliders = [dict(
        active=0,
        currentvalue={'prefix': 'Time: '},
        pad={'t': 50},
        steps=slider_steps
    )]

    ### Set Axis Limits and Update Axes for Each Subplot ###

    # Margin for both subplots
    margin = .1

    x_min = -.5
    x_max = .5
    y_min = -.3
    y_max = .5
    zoom_x_min = -.2
    zoom_x_max = .25
    zoom_y_min = -.05
    zoom_y_max = .35
    

    # Update axes for Subplot 1
    fig.update_xaxes(
        range=[x_min, x_max],
        title_text='X Position',
        scaleanchor='y',
        scaleratio=1,
        constrain='domain',
        row=1, col=1
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        title_text='Y Position',
        constrain='domain',
        row=1, col=1
    )

    # Update axes for Subplot 2
    fig.update_xaxes(
        range=[zoom_x_min, zoom_x_max],
        title_text='X Position',
        scaleanchor='y2',
        scaleratio=1,
        constrain='domain',
        row=1, col=2
    )
    fig.update_yaxes(
        range=[zoom_y_min, zoom_y_max],
        title_text='',
        constrain='domain',
        row=1, col=2
    )

    ### Update Figure Layout ###
    fig.update_layout(
        width=1200,
        height=600,
        autosize=False,
        updatemenus=[
            {
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'redraw': False},
                            'transition': {'duration': ((1000*c.dt)*frame_decimation)/annimation_speedup, 'easing': 'linear'},
                            'fromcurrent': True,
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'showactive': False,
                'y': -0.1,
                'x': -0.01
            }
        ],
        sliders=sliders,
        showlegend=True,
    )

    # Display the animation
    if show_fig:
        fig.show(config={'responsive': False})

    file_name = 'outputs/plotly_animation.html' if output_filename == None else output_filename + '.html'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    fig.write_html(file_name)
    print(
        f'Plotly animated plot created with frame_decimation set to {frame_decimation}\n{len(results)} of {org_len} steps animated\nAnimation saved to: {file_name}')


def split_columns(df, col_names = None):
    if col_names is None:
        col_names = df.columns
    for col_name in col_names:
        if col_name not in df.columns: # Check if column still exists
            continue
        try:
            # Attempt to access the first element to check if it's list-like and has a length
            first_element = df[col_name].iloc[0]
            if not (hasattr(first_element, '__len__') and not isinstance(first_element, str)):
                # If not list-like (or is a string, which has len but shouldn't be split this way)
                # print(f"Skipping column {col_name}: not a list/array type suitable for this splitting.")
                continue

            # Split lists with multiple dimensions into separate columns
            # This part assumes that if a column is to be split, its elements are consistently lists/arrays of the same length.
            # It's crucial that tolist() works as expected.
            split_df = pd.DataFrame(df[col_name].tolist(), index=df.index)
            if split_df.shape[1] == 2: # Specifically for 2-element vectors
                # Rename the new columns based on the original column name
                split_df.columns = [
                    f"{col_name}_d{i+1}" for i in range(split_df.shape[1])]
                # Concatenate the new columns to the original dataframe
                df = pd.concat([df, split_df], axis=1)
                # Drop the original column
                df.drop(columns=[col_name], inplace=True)
            # else:
                # print(f"Column {col_name} not split: not 2 dimensions after tolist(), or already handled.")
        except Exception as e:
            # print(f"Error processing column {col_name} in split_columns: {e}. May have been already processed or is not a list/array type.")
            pass # Suppress error for columns that are not list-like or cannot be split this way

    return df

def split_covariance_matrices(df, col_names=None):
    """
    Splits columns containing 2x2 NumPy arrays (covariance matrices) into two new columns,
    each storing one row of the matrix.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_names (list[str], optional): A list of column names to process.
                                         If None, all columns will be considered.

    Returns:
        pd.DataFrame: The DataFrame with specified covariance matrix columns split.
    """
    if col_names is None:
        col_names = df.columns.tolist() # Process all columns if none specified
    
    processed_df = df.copy()

    for col_name in col_names:
        if col_name not in processed_df.columns:
            continue # Column might have been dropped

        try:
            # Check if the column's elements are consistently 2x2 numpy arrays
            is_cov_matrix_column = False
            # Efficiently check the first non-NaN entry
            first_valid_item = processed_df[col_name].dropna().iloc[0] if not processed_df[col_name].dropna().empty else None
            
            if isinstance(first_valid_item, np.ndarray) and first_valid_item.shape == (2, 2):
                is_cov_matrix_column = True
            
            if is_cov_matrix_column:
                rows_d1 = []
                rows_d2 = []
                for item in processed_df[col_name]:
                    if isinstance(item, np.ndarray) and item.shape == (2, 2):
                        rows_d1.append(item[0, :].tolist()) # First row [var_x, cov_xy]
                        rows_d2.append(item[1, :].tolist()) # Second row [cov_xy, var_y]
                    else:
                        # Handle cases where an item in the column is not a 2x2 matrix or is NaN
                        rows_d1.append([np.nan, np.nan])
                        rows_d2.append([np.nan, np.nan])
                
                new_col_d1_name = f"{col_name}_d1"
                new_col_d2_name = f"{col_name}_d2"
                
                processed_df[new_col_d1_name] = rows_d1
                processed_df[new_col_d2_name] = rows_d2
                processed_df.drop(columns=[col_name], inplace=True)
                # print(f"Split covariance matrix column: {col_name} -> {new_col_d1_name}, {new_col_d2_name}")
            # else:
                # print(f"Column {col_name} not split as a covariance matrix (first valid item type: {type(first_valid_item)}, shape: {getattr(first_valid_item, 'shape', None)}).")

        except Exception as e:
            print(f"Error processing column {col_name} in split_covariance_matrices: {e}")
            
    return processed_df

def plot_joint_angles(results_df, file_type="pdf", output_filename="outputs/joint_angles_plot", extra_text=None, plot_in_degrees=False):
    """
    Plots target vs actual/estimated j1/j2 angles, angular velocities,
    and joint torques over time_run using seaborn in a 3x2 grid.

    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.
                                   Requires columns like 'time_run', 'rad_j1',
                                   'rad_j1_target', 'rad_j2', etc., plus
                                   omega and torque columns (in radians/s).
        output_filename (str): Path to save the output PDF file.
        plot_in_degrees (bool): If True, plots angles and angular velocities in degrees.
    """
    output_filename = f"{output_filename}.{file_type}"
    units_text = "degrees" if plot_in_degrees else "radians"
    print(f"Generating joint angle, velocity, and torque plot ({units_text}): {output_filename}")

    df = results_df.copy()

    # Ensure necessary columns are numeric
    numeric_cols = [
        'time_run', 'dt',
        'true_rad_j1', 'posterior_rad_j1', 'rad_j1_target','posterior_sigma_rad_j1', 'rad_j1_target_radius',
        'true_rad_j2', 'posterior_rad_j2', 'rad_j2_target','posterior_sigma_rad_j2', 'rad_j2_target_radius',
        'true_omega_j1', 'posterior_omega_j1', 'posterior_sigma_omega_j1',
        'true_omega_j2', 'posterior_omega_j2', 'posterior_sigma_omega_j2',
        'torque_j1_efferent', 'torque_j1', 'torque_j1_sigma_scaled',
        'torque_j2_efferent', 'torque_j2', 'torque_j2_sigma_scaled',
        'rad_j1_target_intermediate_this_step', 'rad_j2_target_intermediate_this_step',
        'omega_j1_target_intermediate_this_step', 'omega_j2_target_intermediate_this_step',
        'torque_j1_ff', 'torque_j2_ff',
        'posterior_tau_ext_j1', 'posterior_tau_ext_j2',
        'posterior_sigma_tau_ext_j1', 'posterior_sigma_tau_ext_j2'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found for numeric conversion in plot_joint_angles.")

    angle_unit = "rad"
    omega_unit = "rad/s"

    if plot_in_degrees:
        angle_unit = "deg"
        omega_unit = "deg/s"
        angle_cols = [
            'true_rad_j1', 'posterior_rad_j1', 'rad_j1_target', 'posterior_sigma_rad_j1', 'rad_j1_target_radius',
            'true_rad_j2', 'posterior_rad_j2', 'rad_j2_target', 'posterior_sigma_rad_j2', 'rad_j2_target_radius',
            'rad_j1_target_intermediate_this_step', 'rad_j2_target_intermediate_this_step'
        ]
        omega_cols = [
            'true_omega_j1', 'posterior_omega_j1', 'posterior_sigma_omega_j1',
            'true_omega_j2', 'posterior_omega_j2', 'posterior_sigma_omega_j2',
            'omega_j1_target_intermediate_this_step', 'omega_j2_target_intermediate_this_step'
        ]
        for col in angle_cols + omega_cols:
            if col in df.columns:
                df[col] = np.rad2deg(df[col])

    # Define consistent colors
    color = 'green'
    color_estimated = 'green' # Keep green for estimated, but adjust style
    color_target = 'purple'
    color_est_target = 'purple' # Keep purple for est. target, but adjust style
    color_intermediate_this_step_target = 'orange'

    color_torque = 'red'
    color_torque_efferent = 'red'
    color_torque_ff = 'green'
    color_torque_ext = 'blue'
    color_torque_ideal = 'blue'
    obs_size = 5

    label_j1 = c.j1_label
    label_j2 = c.j2_label


    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(15, 12)) # 3 rows, 2 columns
    fig.suptitle(f'Joint Angle, Velocity, and Torque Control Performance ({units_text.capitalize()})')

    # --- Row 0: Joint Angles ---
    # j1 Angle Plot (Top Left)
    axes[0, 0].set_title(f'{label_j1} Angle')
    # Use radian columns for plotting
    # sns.scatterplot(ax=axes[0, 0], data=df, x='time_run', y='meas_prop_rad_j1', label='Observed', color=color_estimated, s=obs_size, alpha=0.5)
    sns.lineplot(ax=axes[0, 0], data=df, x='time_run', y='true_rad_j1', label='Actual', color=color)
    sns.lineplot(ax=axes[0, 0], data=df, x='time_run', y='posterior_rad_j1', label='Estimated', color=color_estimated, alpha = GLOBAL_LINE_ALPHA_LOW)
    if 'posterior_sigma_rad_j1' in df.columns:
        axes[0, 0].fill_between(df['time_run'],
                                df['posterior_rad_j1'] - 1.96 * df['posterior_sigma_rad_j1'],
                                df['posterior_rad_j1'] + 1.96 * df['posterior_sigma_rad_j1'],
                                color=color_estimated, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est.)') # Underscore to hide from legend if desired later
    sns.lineplot(ax=axes[0, 0], data=df, x='time_run', y='rad_j1_target', label='Target', color=color_target, linestyle='--')
    if 'rad_j1_target_radius' in df.columns and df['rad_j1_target_radius'].iloc[0] > 0:
        axes[0, 0].fill_between(df['time_run'],
                                df['rad_j1_target'] - df['rad_j1_target_radius'],
                                df['rad_j1_target'] + df['rad_j1_target_radius'],
                                color=color_target, alpha=GLOBAL_CI_ALPHA, label='_Target Radius')
    sns.lineplot(ax=axes[0, 0], data=df, x='time_run', y='rad_j1_target_intermediate_this_step', label='Intermediate Target', color=color_intermediate_this_step_target, linestyle='--')
    #sns.lineplot(ax=axes[0, 0], data=df, x='time_run', y='vis_rad_j1_target_mu', label='Est. Target', color=color_est_target, linestyle='--', linewidth = 2, alpha = 0.5)
    # Assuming lims_j1 contains radian limits
    # axes[0, 0].axhline(df['lim_j1_min'][0], color='black', linestyle='--', linewidth=1, label='Limits') # Radian Limits
    # axes[0, 0].axhline(df['lim_j1_max'][0], color='black', linestyle='--', linewidth=1)
    

    axes[0, 0].set_ylabel(f'Angle ({angle_unit})') # Update label
    axes[0, 0].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[0, 0].grid(True)

    # j2 Angle Plot (Top Right)
    axes[0, 1].set_title(f'{label_j2} Angle')
    # Use radian columns for plotting
    # sns.scatterplot(ax=axes[0, 1], data=df, x='time_run', y='meas_prop_rad_j2', label='Observed', color=color_estimated, s=obs_size, alpha=0.5)
    sns.lineplot(ax=axes[0, 1], data=df, x='time_run', y='true_rad_j2', label='Actual', color=color) # Use consistent actual color
    if 'posterior_rad_j2' in df.columns:
        sns.lineplot(ax=axes[0, 1], data=df, x='time_run', y='posterior_rad_j2', label='Estimated', color=color_estimated, alpha = GLOBAL_LINE_ALPHA_LOW) # Use consistent estimated color
    else:
        print("Warning: Column 'posterior_rad_j2' not found in results_df.")
    if 'posterior_sigma_rad_j2' in df.columns:
        axes[0, 1].fill_between(df['time_run'],
                                df['posterior_rad_j2'] - 1.96 * df['posterior_sigma_rad_j2'],
                                df['posterior_rad_j2'] + 1.96 * df['posterior_sigma_rad_j2'],
                                color=color_estimated, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est.)')
    else:
        print("Warning: Column 'posterior_sigma_rad_j2' not found in results_df.")
    sns.lineplot(ax=axes[0, 1], data=df, x='time_run', y='rad_j2_target', label='Target', color=color_target, linestyle='--') # Use consistent target color
    if 'rad_j2_target_radius' in df.columns:
        axes[0, 1].fill_between(df['time_run'],
                                df['rad_j2_target'] - df['rad_j2_target_radius'],
                                df['rad_j2_target'] + df['rad_j2_target_radius'],
                                color=color_target, alpha=GLOBAL_CI_ALPHA, label='_Target Radius')
    sns.lineplot(ax=axes[0, 1], data=df, x='time_run', y='rad_j2_target_intermediate_this_step', label='Intermediate Target', color=color_intermediate_this_step_target, linestyle='--')
    # sns.lineplot(ax=axes[0, 1], data=df, x='time_run', y='vis_rad_j2_target_mu', label='Est. Target', color=color_est_target, linestyle='--', linewidth = 2, alpha = 0.5) # Use consistent est. target color
    # axes[0, 1].set_ylabel('Angle (rad)') # Y label shared conceptually
    # Assuming lims_j2 contains radian limits
    # axes[0, 1].axhline(df['lim_j2_min'][0], color='black', linestyle='--', linewidth=1, label='Limits') # Radian Limits
    # axes[0, 1].axhline(df['lim_j2_max'][0], color='black', linestyle='--', linewidth=1)
    
    axes[0, 1].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[0, 1].grid(True)

    # --- Row 1: Joint Angular Velocities (rad/s) ---
    # j1 Omega Plot (Middle Left)
    axes[1, 0].set_title('j1 Angular Velocity')
    # Use radian/s columns for plotting
    # sns.scatterplot(ax=axes[1, 0], data=df, x='time_run', y='meas_prop_omega_j1', label='Observed', color=color_estimated, s=obs_size, alpha=0.5)
    sns.lineplot(ax=axes[1, 0], data=df, x='time_run', y='true_omega_j1', label='Actual', color=color)
    sns.lineplot(ax=axes[1, 0], data=df, x='time_run', y='posterior_omega_j1', label='Estimated', color=color_estimated, alpha = GLOBAL_LINE_ALPHA_LOW)
    if 'posterior_sigma_omega_j1' in df.columns:
        axes[1, 0].fill_between(df['time_run'],
                                df['posterior_omega_j1'] - 1.96 * df['posterior_sigma_omega_j1'],
                                df['posterior_omega_j1'] + 1.96 * df['posterior_sigma_omega_j1'],
                                color=color_estimated, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est.)')
    sns.lineplot(ax=axes[1, 0], data=df, x='time_run', y='omega_j1_target_intermediate_this_step', label='Intermediate Target', color=color_intermediate_this_step_target, linestyle='--')
    # sns.lineplot(ax=axes[1, 0], data=df, x='time_run', y='omega_j1_target', label='Actual Target', color=color_target, linestyle='--')
    # sns.lineplot(ax=axes[1, 0], data=df, x='time_run', y='vis_omega_j1_target_mu', label='Est. Target', color=color_est_target, linestyle='--', linewidth = 2, alpha = 0.5) # Updated Label & Style
    axes[1, 0].set_ylabel(f'Ang. Vel. ({omega_unit})') # Update label
    axes[1, 0].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[1, 0].grid(True)
    axes[1, 0].axhline(0, color='black', linestyle=':', linewidth=0.5) # Zero line

    # j2 Omega Plot (Middle Right)
    axes[1, 1].set_title('j2 Angular Velocity')
    # Use radian/s columns for plotting
    # sns.scatterplot(ax=axes[1, 1], data=df, x='time_run', y='meas_prop_omega_j2', label='Observed', color=color_estimated, s=obs_size, alpha=0.5)
    sns.lineplot(ax=axes[1, 1], data=df, x='time_run', y='true_omega_j2', label='Actual', color=color) # Use consistent actual color
    if 'posterior_omega_j2' in df.columns:
        sns.lineplot(ax=axes[1, 1], data=df, x='time_run', y='posterior_omega_j2', label='Estimated', color=color_estimated, alpha = GLOBAL_LINE_ALPHA_LOW) # Use consistent estimated color
    else:
        print(f"Warning: Column 'posterior_omega_j2' not found in df.")
    if 'posterior_sigma_omega_j2' in df.columns:
        axes[1, 1].fill_between(df['time_run'],
                                df['posterior_omega_j2'] - 1.96 * df['posterior_sigma_omega_j2'],
                                df['posterior_omega_j2'] + 1.96 * df['posterior_sigma_omega_j2'],
                                color=color_estimated, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est.)')
    else:
        print(f"Warning: Column 'posterior_sigma_omega_j2' not found in df.")
    sns.lineplot(ax=axes[1, 1], data=df, x='time_run', y='omega_j2_target_intermediate_this_step', label='Intermediate Target', color=color_intermediate_this_step_target, linestyle='--')
    # sns.lineplot(ax=axes[1, 1], data=df, x='time_run', y='omega_j2_target', label='Actual Target', color=color_target, linestyle='--') # Use consistent target color
    # sns.lineplot(ax=axes[1, 1], data=df, x='time_run', y='vis_omega_j2_target_mu', label='Est. Target', color=color_est_target, linestyle='--', linewidth = 2, alpha = 0.5) # Updated Label & Style

    # axes[1, 1].set_ylabel('Ang. Vel. (rad/s)') # Y label shared conceptually
    axes[1, 1].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[1, 1].grid(True)
    axes[1, 1].axhline(0, color='black', linestyle=':', linewidth=0.5) # Zero line

    # --- Row 2: Joint Torques ---
    # j1 Torques Plot (Bottom Left)
    axes[2, 0].set_title('j1 Torque')
    if 'torque_j1_sigma_scaled' in df.columns and 'dt' in df.columns:
        std_dev_applied_torque_j1 = df['torque_j1_sigma_scaled'] / np.sqrt(df['dt'])
        axes[2, 0].fill_between(df['time_run'],
                                df['torque_j1_efferent'] - 1.96 * std_dev_applied_torque_j1,
                                df['torque_j1_efferent'] + 1.96 * std_dev_applied_torque_j1,
                                color=color_torque_efferent, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Applied Torque)')
    # sns.lineplot(ax=axes[2, 0], data=df, x='time_run', y='torque_j1_efferent', label='Efferent Cmd', color=color_torque_efferent)
    sns.lineplot(ax=axes[2, 0], data=df, x='time_run', y='torque_j1_ff', label='FF Cmd', color=color_torque_ff, linewidth = 0.5)
    # 95% credible band for estimated external torque (posterior)
    if 'posterior_sigma_tau_ext_j1' in df.columns:
        sns.lineplot(ax=axes[2, 0], data=df, x='time_run', y='posterior_tau_ext_j1', label='Est. External Torque', color=color_torque_ext, linewidth = 0.5)
        axes[2, 0].fill_between(df['time_run'],
                                df['posterior_tau_ext_j1'] - 1.96 * df['posterior_sigma_tau_ext_j1'],
                                df['posterior_tau_ext_j1'] + 1.96 * df['posterior_sigma_tau_ext_j1'],
                                color=color_torque_ext, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est. External)')
    sns.scatterplot(ax=axes[2, 0], data=df, x='time_run', y='torque_j1', label='Applied Torque', color=color_torque, s=GLOBAL_SCATTER_SIZE_SMALL)
    # Optional: Plot efferent torque if saved and desired
    # if 'torque_j1_efferent' in df.columns:
    axes[2, 0].axhline(0, color='black', linestyle='--', linewidth=1) # Zero Torque line
    axes[2, 0].set_ylabel('Torque (Nm)') # Assuming Nm
    axes[2, 0].set_xlabel('time_run (s)')
    axes[2, 0].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[2, 0].grid(True)

    # j2 Torques Plot (Bottom Right)
    axes[2, 1].set_title('j2 Torque')
    if 'torque_j2_sigma_scaled' in df.columns and 'dt' in df.columns:
        std_dev_applied_torque_j2 = df['torque_j2_sigma_scaled'] / np.sqrt(df['dt'])
        axes[2, 1].fill_between(df['time_run'],
                                df['torque_j2_efferent'] - (1.96 * std_dev_applied_torque_j2),
                                df['torque_j2_efferent'] + (1.96 * std_dev_applied_torque_j2),
                                color=color_torque_efferent, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Applied Torque)')
    else:
        print(f"Warning: Column 'torque_j2_sigma_scaled' not found in df.")
    # sns.lineplot(ax=axes[2, 1], data=df, x='time_run', y='torque_j2_efferent', label='Efferent Cmd', color=color_torque_efferent)
    sns.lineplot(ax=axes[2, 1], data=df, x='time_run', y='torque_j2_ff', label='FF Cmd', color=color_torque_ff, linewidth = 0.5)
    # 95% credible band for estimated external torque (posterior)
    if 'posterior_sigma_tau_ext_j2' in df.columns:
        sns.lineplot(ax=axes[2, 1], data=df, x='time_run', y='posterior_tau_ext_j2', label='Est. External Torque', color=color_torque_ext, linewidth = 0.5)
        axes[2, 1].fill_between(df['time_run'],
                                df['posterior_tau_ext_j2'] - (1.96 * df['posterior_sigma_tau_ext_j2']),
                                df['posterior_tau_ext_j2'] + (1.96 * df['posterior_sigma_tau_ext_j2']),
                                color=color_torque_ext, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est. External)')
    sns.scatterplot(ax=axes[2, 1], data=df, x='time_run', y='torque_j2', label='Applied Torque', color=color_torque, s=GLOBAL_SCATTER_SIZE_SMALL)
    # Optional: Plot efferent torque if saved and desired
    # if 'torque_j2_efferent' in df.columns:
    axes[2, 1].axhline(0, color='black', linestyle='--', linewidth=1) # Zero Torque line
    # axes[2, 1].set_ylabel('Torque (Nm)') # Y label shared conceptually
    axes[2, 1].set_xlabel('time_run (s)')
    axes[2, 1].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[2, 1].grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap

    # Add config text if provided
    if extra_text is not None:
        # Create a new axis for the text box
        text_ax = fig.add_axes([0.1, 0.01, 0.8, 0.05])
        text_ax.axis('off')
        text_ax.text(0.5, 0.5, extra_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=text_ax.transAxes,
                    fontsize=8,
                    bbox=dict(facecolor='white', 
                             edgecolor='gray',
                             alpha=0.8,
                             boxstyle='round,pad=0.5'))
        # Adjust layout to make room for text box
        plt.subplots_adjust(bottom=0.15)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Save the plot
    try:
        plt.savefig(output_filename, format=file_type)
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig) # Close the figure to free memory


def plot_joint_angles_j1_locked(results_df, file_type="pdf", output_filename="outputs/joint_angles_plot", extra_text=None, plot_in_degrees=True):
    """
    Plots target vs actual/estimated j1/j2 angles, angular velocities,
    and joint torques over time_run using seaborn in a 3x2 grid.

    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.
                                   Requires columns like 'time_run', 'rad_j1',
                                   'rad_j1_target', 'rad_j2', etc., plus
                                   omega and torque columns (in radians/s).
        output_filename (str): Path to save the output PDF file.
        plot_in_degrees (bool): If True, plots angles and angular velocities in degrees.
    """
    output_filename = f"{output_filename}.{file_type}"
    units_text = "degrees" if plot_in_degrees else "radians"
    print(f"Generating joint angle, velocity, and torque plot ({units_text}): {output_filename}")

    df = results_df[results_df['trial'] == 0].copy()

    # Ensure necessary columns are numeric
    numeric_cols = [
        'time_run', 'dt',
        'true_rad_j1', 'posterior_rad_j1', 'rad_j1_target','posterior_sigma_rad_j1', 'rad_j1_target_radius',
        'true_rad_j2', 'posterior_rad_j2', 'rad_j2_target','posterior_sigma_rad_j2', 'rad_j2_target_radius',
        'true_omega_j1', 'posterior_omega_j1', 'posterior_sigma_omega_j1',
        'true_omega_j2', 'posterior_omega_j2', 'posterior_sigma_omega_j2',
        'torque_j1_efferent', 'torque_j1', 'torque_j1_sigma_scaled',
        'torque_j2_efferent', 'torque_j2', 'torque_j2_sigma_scaled',
        'rad_j1_target_intermediate_this_step', 'rad_j2_target_intermediate_this_step',
        'omega_j1_target_intermediate_this_step', 'omega_j2_target_intermediate_this_step',
        'torque_j1_ff', 'torque_j2_ff'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found for numeric conversion in plot_joint_angles.")

    angle_unit = "rad"
    omega_unit = "rad/s"

    if plot_in_degrees:
        angle_unit = "deg"
        omega_unit = "deg/s"
        # angle_cols = [
        #     'true_rad_j1', 'posterior_joint_rad_j1', 'rad_j1_target', 'posterior_joint_sigma_rad_j1', 'rad_j1_target_radius',
        #     'true_rad_j2', 'posterior_joint_rad_j2', 'rad_j2_target', 'posterior_joint_sigma_rad_j2', 'rad_j2_target_radius',
        #     'rad_j1_target_intermediate_this_step', 'rad_j2_target_intermediate_this_step'
        # ]
        # omega_cols = [
        #     'true_omega_j1', 'posterior_joint_omega_j1', 'posterior_joint_sigma_omega_j1',
        #     'true_omega_j2', 'posterior_joint_omega_j2', 'posterior_joint_sigma_omega_j2',
        #     'omega_j1_target_intermediate_this_step', 'omega_j2_target_intermediate_this_step'
        # ]
        # for col in angle_cols + omega_cols:
        #     if col in df.columns:
        #         df[col] = np.rad2deg(df[col])

    # Define consistent colors
    color = 'green'
    color_estimated = 'green' # Keep green for estimated, but adjust style
    color_target = 'purple'
    color_est_target = 'purple' # Keep purple for est. target, but adjust style
    color_intermediate_this_step_target = 'orange'

    color_torque = 'red'
    color_torque_efferent = 'red'
    color_torque_ff = 'green'
    color_torque_ideal = 'blue'
    obs_size = 5

    label_j1 = c.j1_label
    label_j2 = c.j2_label


    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 12)) # 3 rows, 1 column
    fig.suptitle(f'Joint Angle, Velocity, and Torque Control Performance ({units_text.capitalize()})')

    # j2 Angle Plot (Top)
    axes[0].set_title(f'{label_j2} Angle')
    # Use radian columns for plotting
    # sns.scatterplot(ax=axes[0], data=df, x='time_run', y='meas_prop_rad_j2', label='Observed', color=color_estimated, s=obs_size, alpha=0.5)
    sns.lineplot(ax=axes[0], data=df, x='time_run', y='true_rad_j2', label='Actual', color=color) # Use consistent actual color
    if 'posterior_rad_j2' in df.columns:
        sns.lineplot(ax=axes[0], data=df, x='time_run', y='posterior_rad_j2', label='Estimated', color=color_estimated, alpha = GLOBAL_LINE_ALPHA_LOW) # Use consistent estimated color
    else:
        print("Warning: Column 'posterior_rad_j2' not found in results_df.")
    if 'posterior_sigma_rad_j2' in df.columns:
        axes[0].fill_between(df['time_run'],
                                df['posterior_rad_j2'] - 1.96 * df['posterior_sigma_rad_j2'],
                                df['posterior_rad_j2'] + 1.96 * df['posterior_sigma_rad_j2'],
                                color=color_estimated, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est.)')
    else:
        print("Warning: Column 'posterior_sigma_rad_j2' not found in results_df.")
    sns.lineplot(ax=axes[0], data=df, x='time_run', y='rad_j2_target', label='Target', color=color_target, linestyle='--') # Use consistent target color
    if 'rad_j2_target_radius' in df.columns:
        axes[0].fill_between(df['time_run'],
                                df['rad_j2_target'] - df['rad_j2_target_radius'],
                                df['rad_j2_target'] + df['rad_j2_target_radius'],
                                color=color_target, alpha=GLOBAL_CI_ALPHA, label='_Target Radius')
    sns.lineplot(ax=axes[0], data=df, x='time_run', y='rad_j2_target_intermediate_this_step', label='Intermediate Target', color=color_intermediate_this_step_target, linestyle='--')
    # sns.lineplot(ax=axes[0], data=df, x='time_run', y='vis_rad_j2_target_mu', label='Est. Target', color=color_est_target, linestyle='--', linewidth = 2, alpha = 0.5) # Use consistent est. target color
    axes[0].set_ylabel(f'Angle ({angle_unit})') # Update label
    # Assuming lims_j2 contains radian limits
    # axes[0].axhline(df['lim_j2_min'][0], color='black', linestyle='--', linewidth=1, label='Limits') # Radian Limits
    # axes[0].axhline(df['lim_j2_max'][0], color='black', linestyle='--', linewidth=1)
    
    axes[0].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[0].grid(True)

    # --- Row 1: Joint Angular Velocities (rad/s) ---

    # j2 Omega Plot (Middle)
    axes[1].set_title('j2 Angular Velocity')
    # Use radian/s columns for plotting
    # sns.scatterplot(ax=axes[1], data=df, x='time_run', y='meas_prop_omega_j2', label='Observed', color=color_estimated, s=obs_size, alpha=0.5)
    sns.lineplot(ax=axes[1], data=df, x='time_run', y='true_omega_j2', label='Actual', color=color) # Use consistent actual color
    if 'posterior_omega_j2' in df.columns:
        sns.lineplot(ax=axes[1], data=df, x='time_run', y='posterior_omega_j2', label='Estimated', color=color_estimated, alpha = GLOBAL_LINE_ALPHA_LOW) # Use consistent estimated color
    else:
        print(f"Warning: Column 'posterior_omega_j2' not found in df.")
    if 'posterior_sigma_omega_j2' in df.columns:
        axes[1].fill_between(df['time_run'],
                                df['posterior_omega_j2'] - 1.96 * df['posterior_sigma_omega_j2'],
                                df['posterior_omega_j2'] + 1.96 * df['posterior_sigma_omega_j2'],
                                color=color_estimated, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Est.)')
    else:
        print(f"Warning: Column 'posterior_sigma_omega_j2' not found in df.")
    sns.lineplot(ax=axes[1], data=df, x='time_run', y='omega_j2_target_intermediate_this_step', label='Intermediate Target', color=color_intermediate_this_step_target, linestyle='--')
    # sns.lineplot(ax=axes[1], data=df, x='time_run', y='omega_j2_target', label='Actual Target', color=color_target, linestyle='--') # Use consistent target color
    # sns.lineplot(ax=axes[1], data=df, x='time_run', y='vis_omega_j2_target_mu', label='Est. Target', color=color_est_target, linestyle='--', linewidth = 2, alpha = 0.5) # Updated Label & Style

    axes[1].set_ylabel(f'Ang. Vel. ({omega_unit})')  # Update label
    axes[1].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[1].grid(True)
    axes[1].axhline(0, color='black', linestyle=':', linewidth=0.5) # Zero line

    # j2 Torques Plot (Bottom)
    axes[2].set_title('j2 Torque')
    if 'torque_j2_sigma_scaled' in df.columns and 'dt' in df.columns:
        std_dev_applied_torque_j2 = df['torque_j2_sigma_scaled'] / np.sqrt(df['dt'])
        axes[2].fill_between(df['time_run'],
                                df['torque_j2_efferent'] - (1.96 * std_dev_applied_torque_j2),
                                df['torque_j2_efferent'] + (1.96 * std_dev_applied_torque_j2),
                                color=color_torque_efferent, alpha=GLOBAL_CI_ALPHA, label='_95% CI (Applied Torque)')
    else:
        print(f"Warning: Column 'torque_j2_sigma_scaled' not found in df.")
    sns.lineplot(ax=axes[2], data=df, x='time_run', y='torque_j2_efferent', label='Efferent Cmd', color=color_torque_efferent)
    sns.lineplot(ax=axes[2], data=df, x='time_run', y='torque_j2_ff', label='FF Cmd', color=color_torque_ff)
    sns.scatterplot(ax=axes[2], data=df, x='time_run', y='torque_j2', label='Applied Torque', color=color_torque, s=GLOBAL_SCATTER_SIZE_SMALL)
    # Optional: Plot efferent torque if saved and desired
    # if 'torque_j2_efferent' in df.columns:
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1) # Zero Torque line
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].set_xlabel('time_run (s)')
    axes[2].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[2].grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap

    # Add config text if provided
    if extra_text is not None:
        # Create a new axis for the text box
        text_ax = fig.add_axes([0.1, 0.01, 0.8, 0.05])
        text_ax.axis('off')
        text_ax.text(0.5, 0.5, extra_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=text_ax.transAxes,
                    fontsize=8,
                    bbox=dict(facecolor='white', 
                             edgecolor='gray',
                             alpha=0.8,
                             boxstyle='round,pad=0.5'))
        # Adjust layout to make room for text box
        plt.subplots_adjust(bottom=0.15)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Save the plot
    try:
        plt.savefig(output_filename, format=file_type)
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig) # Close the figure to free memory

def plot_proprioceptive_errors(results_df, file_type="pdf", output_filename="outputs/proprioceptive_errors_plot"):
    """
    Plots normalized proprioceptive prediction errors (radian and omega for j1/j2)
    over time_run using seaborn in a 2x2 grid.

    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.
                                   Requires columns like 'time_run', 
                                   'prop_rad_j1_error_norm', etc.
        output_filename (str): Path to save the output PDF file.
    """
    output_filename = f"{output_filename}.{file_type}"
    print(f"Generating proprioceptive error plot: {output_filename}")

    error_cols = [
        'prop_rad_j1_error_norm', 'prop_rad_j2_error_norm',
        'prop_omega_j1_error_norm', 'prop_omega_j2_error_norm'
    ]
    for col in error_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found for numeric conversion in plot_proprioceptive_errors.")
            # Add a dummy column with NaNs if it's missing, so plotting doesn't fail catastrophically
            results_df[col] = np.nan


    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(15, 10))
    fig.suptitle('Proprioceptive Prediction Errors (Normalized)')

    color_error = 'orangered'
    significance_color = 'grey'
    significance_linestyle = '--'
    zero_line_color = 'black'
    zero_line_style = ':'

    # Top Row: Radian Errors
    # j1 Radian Error
    axes[0, 0].set_title('j1 Angle Prediction Error')
    sns.lineplot(ax=axes[0, 0], data=results_df, x='time_run', y='prop_rad_j1_error_norm', label='Norm. Error', color=color_error)
    axes[0, 0].axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)
    axes[0, 0].axhline(1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1, label='±1.96 SD')
    axes[0, 0].axhline(-1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1)
    axes[0, 0].set_ylabel('Normalized Error (Z-score)')
    axes[0, 0].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[0, 0].grid(True)

    # j2 Radian Error
    axes[0, 1].set_title('j2 Angle Prediction Error')
    sns.lineplot(ax=axes[0, 1], data=results_df, x='time_run', y='prop_rad_j2_error_norm', label='Norm. Error', color=color_error)
    axes[0, 1].axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)
    axes[0, 1].axhline(1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1, label='±1.96 SD')
    axes[0, 1].axhline(-1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1)
    # axes[0, 1].set_ylabel('Normalized Error (Z-score)') # Shared Y-label implicitly
    axes[0, 1].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[0, 1].grid(True)

    # Bottom Row: Omega Errors
    # j1 Omega Error
    axes[1, 0].set_title('j1 Angular Velocity Prediction Error')
    sns.lineplot(ax=axes[1, 0], data=results_df, x='time_run', y='prop_omega_j1_error_norm', label='Norm. Error', color=color_error)
    axes[1, 0].axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)
    axes[1, 0].axhline(1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1, label='±1.96 SD')
    axes[1, 0].axhline(-1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1)
    axes[1, 0].set_ylabel('Normalized Error (Z-score)')
    axes[1, 0].set_xlabel('time_run (s)')
    axes[1, 0].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[1, 0].grid(True)

    # j2 Omega Error
    axes[1, 1].set_title('j2 Angular Velocity Prediction Error')
    sns.lineplot(ax=axes[1, 1], data=results_df, x='time_run', y='prop_omega_j2_error_norm', label='Norm. Error', color=color_error)
    axes[1, 1].axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)
    axes[1, 1].axhline(1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1, label='±1.96 SD')
    axes[1, 1].axhline(-1.96, color=significance_color, linestyle=significance_linestyle, linewidth=1)
    # axes[1, 1].set_ylabel('Normalized Error (Z-score)') # Shared Y-label implicitly
    axes[1, 1].set_xlabel('time_run (s)')
    axes[1, 1].legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    # axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        plt.savefig(output_filename, format=file_type)
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig)

def plot_sigma_contributions(results_df, file_type="pdf", output_filename="outputs/sigma_contributions_plot"):
    """
    Plots the contributing sigma values for posterior angle and angular velocity calculations.

    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.
        output_filename (str): Path to save the output PDF file.
    """
    output_filename = f"{output_filename}.{file_type}"
    print(f"Generating sigma contributions plot: {output_filename}")

    sigma_cols = [
        'time_run',
        'prop_rad_j1_sigma', 'vis_rad_j1_sigma', 'rad_j1_from_omega_sigma', 'prior_sigma_rad_j1', 'posterior_sigma_rad_j1',
        'prop_rad_j2_sigma', 'vis_rad_j2_sigma', 'rad_j2_from_omega_sigma', 'prior_sigma_rad_j2', 'posterior_sigma_rad_j2',
        'prop_omega_j1_sigma', 'vis_omega_j1_sigma', 'prior_sigma_omega_j1', 'posterior_sigma_omega_j1',
        'prop_omega_j2_sigma', 'vis_omega_j2_sigma', 'prior_sigma_omega_j2', 'posterior_sigma_omega_j2',
    ]

    for col in sigma_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        else:
            # print(f"Warning: Sigma column '{col}' not found for numeric conversion in plot_sigma_contributions.")
            results_df[col] = np.nan # Add as NaN column to prevent plotting errors later

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(17, 10))
    fig.suptitle('Sigma Contributions to Posterior State Estimation')

    j1_label = c.j1_label
    j2_label = c.j2_label

    # --- j1 Angle Sigmas (Top Left) ---
    ax = axes[0, 0]
    ax.set_title(f'{j1_label} Angle Sigmas')
    sigmas_to_plot_j1_rad = {
        'Proprioceptive Sample': 'prop_rad_j1_sigma',
        'Visual Sample': 'vis_rad_j1_sigma',
        'From Omega': 'rad_j1_from_omega_sigma',
        'Prior': 'prior_sigma_rad_j1',
        'Posterior': 'posterior_sigma_rad_j1'
    }
    for label, col_name in sigmas_to_plot_j1_rad.items():
        if col_name in results_df.columns and not results_df[col_name].isnull().all():
            sns.lineplot(ax=ax, data=results_df, x='time_run', y=col_name, label=label)
    ax.set_ylabel('Sigma (rad)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    ax.grid(True)

    # --- j2 Angle Sigmas (Top Right) ---
    ax = axes[0, 1]
    ax.set_title(f'{j2_label} Angle Sigmas')
    sigmas_to_plot_j2_rad = {
        'Proprioceptive Sample': 'prop_rad_j2_sigma',
        'Visual Sample': 'vis_rad_j2_sigma',
        'From Omega': 'rad_j2_from_omega_sigma',
        'Prior': 'prior_sigma_rad_j2',
        'Posterior': 'posterior_sigma_rad_j2'
    }
    for label, col_name in sigmas_to_plot_j2_rad.items():
        if col_name in results_df.columns and not results_df[col_name].isnull().all():
            sns.lineplot(ax=ax, data=results_df, x='time_run', y=col_name, label=label)
    # ax.set_ylabel('Sigma (rad)') # Shared Y-label implicitly
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    ax.grid(True)

    # --- j1 Angular Velocity Sigmas (Bottom Left) ---
    ax = axes[1, 0]
    ax.set_title(f'{j1_label} Angular Velocity Sigmas')
    sigmas_to_plot_j1_omega = {
        'Proprioceptive Sample': 'prop_omega_j1_sigma',
        'Visual Sample': 'vis_omega_j1_sigma',
        'Prior': 'prior_sigma_omega_j1',
        'Posterior': 'posterior_sigma_omega_j1'
    }
    for label, col_name in sigmas_to_plot_j1_omega.items():
        if col_name in results_df.columns and not results_df[col_name].isnull().all():
            sns.lineplot(ax=ax, data=results_df, x='time_run', y=col_name, label=label)
    ax.set_ylabel('Sigma (rad/s)')
    ax.set_xlabel('time_run (s)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    ax.grid(True)

    # --- j2 Angular Velocity Sigmas (Bottom Right) ---
    ax = axes[1, 1]
    ax.set_title(f'{j2_label} Angular Velocity Sigmas')
    sigmas_to_plot_j2_omega = {
        'Proprioceptive Sample': 'prop_omega_j2_sigma',
        'Visual Sample': 'vis_omega_j2_sigma',
        'Prior': 'prior_sigma_omega_j2',
        'Posterior': 'posterior_sigma_omega_j2'
    }
    for label, col_name in sigmas_to_plot_j2_omega.items():
        if col_name in results_df.columns and not results_df[col_name].isnull().all():
            sns.lineplot(ax=ax, data=results_df, x='time_run', y=col_name, label=label)
    # ax.set_ylabel('Sigma (rad/s)') # Shared Y-label implicitly
    ax.set_xlabel('time_run (s)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        plt.savefig(output_filename, format='pdf')
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig)

def plot_normalized_innovations(results_df, file_type="pdf", output_filename="outputs/normalized_innovations_plot"):
    """
    Plots component-wise normalized measurement innovations over time_run.
    Grid: 3 rows, 2 columns.
    Columns: Joint 1/Visual X (left), Joint 2/Visual Y (right).
    Rows: Proprioceptive Angle, Proprioceptive Angular Velocity, Visual Hand Position.

    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.
                                   Requires columns like 'time_run', 
                                   'norm_innov_prop_rad_j1', 'norm_innov_vis_x', etc.
        output_filename (str): Path to save the output PDF file.
    """
    output_filename = f"{output_filename}.{file_type}"
    print(f"Generating normalized innovations plot: {output_filename}")

    # Columns to plot and their corresponding axes titles
    # (Column name in df, subplot title, subplot y-label)
    plot_specs = [
        ('norm_innov_prop_rad_j1', f'{c.j1_label} Angle Innov.', 'Norm. Innov. (Z-score)'),
        ('norm_innov_prop_rad_j2', f'{c.j2_label} Angle Innov.', ''), # No Y-label for right column typically
        ('norm_innov_prop_omega_j1', f'{c.j1_label} Ang. Vel. Innov.', 'Norm. Innov. (Z-score)'),
        ('norm_innov_prop_omega_j2', f'{c.j2_label} Ang. Vel. Innov.', ''),
        ('norm_innov_vis_x', 'Visual Hand X Innov.', 'Norm. Innov. (Z-score)'),
        ('norm_innov_vis_y', 'Visual Hand Y Innov.', '')
    ]

    # Ensure necessary columns are numeric and exist
    for col_name, _, _ in plot_specs:
        if col_name in results_df.columns:
            results_df[col_name] = pd.to_numeric(results_df[col_name], errors='coerce')
        else:
            print(f"Warning: Column '{col_name}' not found for numeric conversion in plot_normalized_innovations.")
            results_df[col_name] = np.nan # Add as NaN column to prevent plotting errors
    
    if 'time_run' not in results_df.columns:
        print("Error: 'time_run' column not found in results_df for plot_normalized_innovations.")
        return
    results_df['time_run'] = pd.to_numeric(results_df['time_run'], errors='coerce')


    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(15, 12))
    fig.suptitle('Component-wise Normalized Measurement Innovations')

    color_innovation = 'dodgerblue'
    ci_line_color = 'gray'
    ci_line_style = '--'
    zero_line_color = 'black'
    zero_line_style = ':'

    for i, (col_name, title, ylabel) in enumerate(plot_specs):
        row = i // 2
        col = i % 2
        ax = axes[row, col]


        # Add scatter plot with variable alpha
        if col_name in results_df.columns:
            y_series = results_df[col_name].dropna()
            if not y_series.empty:
                x_series = results_df.loc[y_series.index, 'time_run']
                abs_y = np.abs(y_series)
                
                min_alpha_scatter = 0.05  # Alpha for y near 0
                max_alpha_scatter = 1.0   # Alpha for abs(y) >= threshold
                threshold_scatter = 5.0
                
                # Initialize alphas to min_alpha_scatter
                alphas_scatter = np.full_like(abs_y, min_alpha_scatter, dtype=float)
                
                # Mask for values between a very small epsilon and the threshold
                scale_mask = (abs_y > 1e-6) & (abs_y < threshold_scatter)
                alphas_scatter[scale_mask] = min_alpha_scatter + \
                                           (max_alpha_scatter - min_alpha_scatter) * (abs_y[scale_mask] / threshold_scatter)
                
                # Mask for values at or beyond the threshold
                opaque_mask = abs_y >= threshold_scatter
                alphas_scatter[opaque_mask] = max_alpha_scatter
                
                # Scatter points on top
                ax.scatter(x_series, y_series, alpha=alphas_scatter, color=color_innovation, s=GLOBAL_SCATTER_SIZE_DEFAULT, zorder=3, label='_nolegend_') # Used GLOBAL_SCATTER_SIZE_DEFAULT

                # Add 10-observation moving average line
                if len(y_series) >= 1: # Check if there is data to average
                    moving_avg = y_series.rolling(window=10, center=True, min_periods=1).mean()
                    ax.plot(x_series, moving_avg, color='black', linestyle='-', label='10-obs MA', zorder=4)

        ax.axhline(0, color=zero_line_color, linestyle=zero_line_style, linewidth=1)
        ax.axhline(1.96, color=ci_line_color, linestyle=ci_line_style, linewidth=1, label='±1.96 SD (95% CI)')
        ax.axhline(-1.96, color=ci_line_color, linestyle=ci_line_style, linewidth=1)
        
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        if row == 2: # Bottom row
            ax.set_xlabel('time_run (s)')
        
        # Only show legend on the first subplot to avoid redundancy, or customize as needed
        if i == 0:
            ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
        else:
            ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE).set_visible(False)
        ax.grid(True)

    # Ensure all subplots share the full time_run range
    if not results_df['time_run'].empty:
        min_time = results_df['time_run'].min()
        max_time = results_df['time_run'].max()
        if pd.notna(min_time) and pd.notna(max_time):
            for ax_row in axes:
                for ax_col in ax_row:
                    ax_col.set_xlim(min_time, max_time)
        else:
            print("Warning: Could not determine valid min/max time_run for x-axis in plot_normalized_innovations.")
    else:
        print("Warning: 'time_run' column is empty in plot_normalized_innovations.")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        plt.savefig(output_filename, format=file_type)
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig)

def plot_trajectory_analysis(results_df, file_type="pdf", output_filename="outputs/trajectory_analysis_plot", 
                               extra_text=None, planned_velocity_vector_sample_rate=10):
    """
    Creates a 2x2 panel plot showing:
    - Top Left: Intended vs actual velocity profiles
    - Top Right: Movement path in Cartesian space
    - Bottom Left: Alpha-beta distribution velocity curve
    - Bottom Right: (Reserved for future use)
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe containing trajectory data
    output_filename : str
        Path to save the output plot
    """
    output_filename = f"{output_filename}.{file_type}"
    # The following lines for local theme/context setting are to be removed.
    # sns.set_theme(style="whitegrid") 
    # base_rc_legend_fontsize = plt.rcParams['legend.fontsize'] 
    # sns.set_context("paper", font_scale=1.2, rc={"legend.fontsize": base_rc_legend_fontsize})
    
    # Create figure with 2x2 panels
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Top Left panel: Velocity profiles
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate actual velocity magnitude
    actual_vel_x = results_df['true_hand_vx']
    actual_vel_y = results_df['true_hand_vy']
    actual_vel_mag = np.sqrt(actual_vel_x**2 + actual_vel_y**2)
    
    # Create velocity profile plot using seaborn
    time = results_df['time']
    sns.lineplot(x=time, y=actual_vel_mag, ax=ax1, 
                 label='Actual Velocity') # Color from global palette, linewidth from rcParams
    
    if 'target_velocity_x' in results_df.columns and 'target_velocity_y' in results_df.columns:
        target_vel_x = results_df['target_velocity_x']
        target_vel_y = results_df['target_velocity_y']
        target_vel_mag = np.sqrt(target_vel_x**2 + target_vel_y**2)
        sns.lineplot(x=time, y=target_vel_mag, ax=ax1,
                     label='Intended Velocity', 
                     linestyle='--') # Color from global palette, linewidth from rcParams
    if 'v_planned_trajectory_x' in results_df.columns and 'v_planned_trajectory_y' in results_df.columns:
        planned_vel_x = results_df['v_planned_trajectory_x']
        planned_vel_y = results_df['v_planned_trajectory_y'] 
        planned_vel_mag = np.sqrt(planned_vel_x**2 + planned_vel_y**2)
        sns.lineplot(x=time, y=planned_vel_mag, ax=ax1,
                    label='Planned Velocity',
                    linestyle=':')  # Different linestyle to distinguish from intended
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity Magnitude (m/s)')
    ax1.set_title('Velocity Profiles', pad=20)
    ax1.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)  # Add missing legend call
    
    # Top Right panel: Movement path
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot movement path using seaborn
    sns.lineplot(x=results_df['true_hand_x'], 
                    y=results_df['true_hand_y'],
                    ax=ax2, label='Actual Path', sort=False) # Color from global palette, linewidth from rcParams
    
    # Plot target path if available
    if 'target_position_x' in results_df.columns and 'target_position_y' in results_df.columns:
        sns.lineplot(x=results_df['target_position_x'],
                     y=results_df['target_position_y'],
                     ax=ax2, label='Target Path',
                     linestyle='--', sort=False) # Color from global palette, linewidth from rcParams

    # Plot UKF perceived movement path if available
    if 'posterior_hand_x' in results_df.columns and 'posterior_hand_y' in results_df.columns:
        sns.lineplot(x=results_df['posterior_hand_x'],
                     y=results_df['posterior_hand_y'],
                     ax=ax2, label='UKF Perceived Path',
                     linestyle='-.', sort=False) # Dash-dot linestyle to distinguish from other paths

    # Plot planned trajectory path if available
    if ('p_planned_trajectory_x' in results_df.columns and 
        'p_planned_trajectory_y' in results_df.columns and
        'v_planned_trajectory_x' in results_df.columns and
        'v_planned_trajectory_y' in results_df.columns):
        
        x_planned = results_df['p_planned_trajectory_x']
        y_planned = results_df['p_planned_trajectory_y']
        vx_planned = results_df['v_planned_trajectory_x']
        vy_planned = results_df['v_planned_trajectory_y']
        
        # Subsample the points for plotting velocity vectors
        sampled_indices = np.arange(0, len(x_planned), planned_velocity_vector_sample_rate)
        
        if len(sampled_indices) > 0:
            # Use a scale factor for arrow lengths to make them visible but not overwhelming
            # This can be adjusted based on typical velocity magnitudes and plot scale
            velocity_scale_factor = 0.1  # Adjust as needed
            
            ax2.quiver(x_planned.iloc[sampled_indices], y_planned.iloc[sampled_indices],
                       vx_planned.iloc[sampled_indices] * velocity_scale_factor, 
                       vy_planned.iloc[sampled_indices] * velocity_scale_factor,
                       angles='xy', scale_units='xy', scale=1, color='purple', 
                       label='Planned Velocity Vectors', width=0.003, headwidth=3, headlength=5)

    
    # Plot start and end points with seaborn scatterplot
    sns.scatterplot(x=[results_df['true_hand_x'].iloc[0]], 
                    y=[results_df['true_hand_y'].iloc[0]],
                    ax=ax2, label='Start', s=100) # Color from global palette
    sns.scatterplot(x=[results_df['true_hand_x'].iloc[-1]], 
                    y=[results_df['true_hand_y'].iloc[-1]],
                    ax=ax2, label='End', s=100) # Color from global palette

    # --- Adjust axis limits for square aspect ratio ---
    x_coords_list = []
    y_coords_list = []

    if 'true_hand_x' in results_df.columns and not results_df['true_hand_x'].empty:
        x_coords_list.append(results_df['true_hand_x'])
    if 'true_hand_y' in results_df.columns and not results_df['true_hand_y'].empty:
        y_coords_list.append(results_df['true_hand_y'])

    if 'target_position_x' in results_df.columns and not results_df['target_position_x'].empty:
        x_coords_list.append(results_df['target_position_x'])
    if 'target_position_y' in results_df.columns and not results_df['target_position_y'].empty:
        y_coords_list.append(results_df['target_position_y'])
    
    if 'p_planned_trajectory_x' in results_df.columns and not results_df['p_planned_trajectory_x'].empty:
        x_coords_list.append(results_df['p_planned_trajectory_x'])
    if 'p_planned_trajectory_y' in results_df.columns and not results_df['p_planned_trajectory_y'].empty:
        y_coords_list.append(results_df['p_planned_trajectory_y'])

    # Include UKF perceived path in axis limits calculation
    if 'posterior_hand_x' in results_df.columns and not results_df['posterior_hand_x'].empty:
        x_coords_list.append(results_df['posterior_hand_x'])
    if 'posterior_hand_y' in results_df.columns and not results_df['posterior_hand_y'].empty:
        y_coords_list.append(results_df['posterior_hand_y'])

    apply_custom_limits = False
    if x_coords_list and y_coords_list: # Both must have at least one series
        all_x_data = pd.concat(x_coords_list)
        all_y_data = pd.concat(y_coords_list)
        if not all_x_data.empty and not all_y_data.empty:
            apply_custom_limits = True
            
    if apply_custom_limits:
        min_x, max_x = all_x_data.min(), all_x_data.max()
        min_y, max_y = all_y_data.min(), all_y_data.max()

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        data_span_x = max_x - min_x
        data_span_y = max_y - min_y
        
        MIN_VISUAL_SPAN = 0.05  # Minimum width/height of the plot in data units (e.g., 5cm if units are meters)
        PADDING_FACTOR = 1.1    # 10% padding

        effective_span_x = max(data_span_x, MIN_VISUAL_SPAN)
        effective_span_y = max(data_span_y, MIN_VISUAL_SPAN)

        target_display_span = max(effective_span_x, effective_span_y)
        padded_display_span = target_display_span * PADDING_FACTOR
        
        # Ensure the span is not extremely small if all data was concentrated at a point
        # and MIN_VISUAL_SPAN was also very small or zero (though it's non-zero here).
        if padded_display_span < 1e-9: 
            padded_display_span = MIN_VISUAL_SPAN * PADDING_FACTOR if MIN_VISUAL_SPAN > 1e-9 else 0.1

        ax2.set_xlim(center_x - padded_display_span / 2, center_x + padded_display_span / 2)
        ax2.set_ylim(center_y - padded_display_span / 2, center_y + padded_display_span / 2)
    # If not apply_custom_limits, existing autoscaled limits will be used by set_aspect.
    # --- End of axis limits adjustment ---
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Movement Path', pad=20)
    
    # Equal aspect ratio for movement path, after limits are set
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)

    
    # Bottom Left panel: Alpha-beta distribution
    ax3 = fig.add_subplot(gs[1, 0])

    # Get movement plan duration from config
    movement_plan_duration = c.planned_max_time_target
    actual_time_points = results_df['time'] # X-axis for this plot: actual elapsed time

    # Calculate normalized time relative to the movement plan duration.
    # This normalized time (t_norm in [0,1]) is the input to the alpha-beta formula.
    normalized_time_for_formula = actual_time_points / movement_plan_duration

    # Get alpha and beta parameters from the DataFrame (taking the first value)
    alpha = results_df['trajectory_alpha'].iloc[0] if 'trajectory_alpha' in results_df.columns else 2.0
    beta = results_df['trajectory_beta'].iloc[0] if 'trajectory_beta' in results_df.columns else 2.0

    # Initialize the calculated velocity profile series with NaNs
    alpha_beta_velocity_profile = pd.Series(np.nan, index=actual_time_points.index)

    # Identify the segment of time points where normalized time is within [0, 1]
    valid_segment_mask = (normalized_time_for_formula >= 0) & (normalized_time_for_formula <= 1)
    t_input_values_for_profile = normalized_time_for_formula[valid_segment_mask].values

    if t_input_values_for_profile.size > 0:
        # Calculate the "raw" alpha-beta profile for the valid segment
        # Numpy's `**` operator: 0**0=1; 0**positive=0. For alpha,beta >= 1.
        raw_profile_segment = (t_input_values_for_profile ** (alpha - 1)) * \
                              ((1 - t_input_values_for_profile) ** (beta - 1))

        # Normalize the calculated segment so its peak is 1
        max_raw_value = np.nanmax(raw_profile_segment)
        if max_raw_value > 1e-9: # Avoid division by zero or tiny numbers
            normalized_profile_segment = raw_profile_segment / max_raw_value
        elif np.all(np.isnan(raw_profile_segment)) or raw_profile_segment.size == 0:
            normalized_profile_segment = raw_profile_segment # Preserve NaNs or empty array
        else: # Max is zero or very small (profile is essentially zero)
            normalized_profile_segment = np.zeros_like(raw_profile_segment)
        
        # Explicitly cast to float64 to avoid FutureWarning
        alpha_beta_velocity_profile.loc[valid_segment_mask] = normalized_profile_segment.astype(np.float64)

    sns.lineplot(x=actual_time_points, y=alpha_beta_velocity_profile, ax=ax3,
                 label='Intended Velocity Profile', sort=False) # Color from global palette, linewidth from rcParams
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Normalized Intended Velocity')
    ax3.set_title('Alpha-Beta Velocity Profile', pad=20)
    # ax3.grid(True, alpha=0.3) # This line to be removed
    ax3.legend(fontsize=GLOBAL_LEGEND_FONTSIZE) 
      
    # Adjust layout and save
    plt.tight_layout()
    
    # Add config text if provided
    if extra_text is not None:
        # Create a new axis for the text box
        text_ax = fig.add_axes([0.1, 0.01, 0.8, 0.05])
        text_ax.axis('off')
        text_ax.text(0.5, 0.5, extra_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=text_ax.transAxes,
                    fontsize=8,
                    bbox=dict(facecolor='white', 
                             edgecolor='gray',
                             alpha=0.8,
                             boxstyle='round,pad=0.5'))
        # Adjust layout to make room for text box
        plt.subplots_adjust(bottom=0.15)
        
    plt.savefig(output_filename, format=file_type) # dpi and bbox_inches from global rcParams
    plt.close()

def plot_visual_feedback_impact(results_df, file_type="pdf", output_filename="outputs/visual_feedback_impact_plot"):
    """
    Plots to understand how visual feedback impacts posterior joint angle and velocity estimates.
    Shows Kalman gain components for both visual and proprioceptive measurements.
    
    Grid: 3 rows, 2 columns
    - Top row: Visual X Kalman gains (left) and Proprioceptive angle Kalman gains (right)
    - Middle row: Visual Y Kalman gains (left) and Proprioceptive velocity Kalman gains (right)
    - Bottom row: Innovation analysis split by measurement type (visual vs proprioceptive)
    
    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results with UKF matrices
        output_filename (str): Path to save the output PDF file
    """
    output_filename = f"{output_filename}.{file_type}"
    print(f"Generating visual feedback impact analysis plot: {output_filename}")
    
    # Check for required columns
    required_cols = [
        'time_run',
        'kalman_gain_vis_x_to_rad_j1', 'kalman_gain_vis_x_to_rad_j2',
        'kalman_gain_vis_x_to_omega_j1', 'kalman_gain_vis_x_to_omega_j2',
        'kalman_gain_vis_y_to_rad_j1', 'kalman_gain_vis_y_to_rad_j2', 
        'kalman_gain_vis_y_to_omega_j1', 'kalman_gain_vis_y_to_omega_j2'
    ]
    
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for visual feedback impact analysis: {missing_cols}")
        print("Please update unpack_ukf_results_vectorized to save these values.")
        return
    
    # Convert to numeric
    for col in required_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(15, 12))
    fig.suptitle('Visual Feedback Impact on Joint State Estimates')
    
    # Define colors
    color_rad_j1 = 'blue'
    color_rad_j2 = 'red' 
    color_omega_j1 = 'green'
    color_omega_j2 = 'orange'
    
    # Top row: Visual X Kalman gains (left) and Proprioceptive angle Kalman gains (right)
    # Visual X gains (Top Left)
    ax = axes[0, 0]
    ax.set_title('Kalman Gains: Visual X → Joint States')
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_x_to_rad_j1', 
                label=f'K(vis_x → {c.j1_label}_angle)', color=color_rad_j1)
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_x_to_rad_j2',
                label=f'K(vis_x → {c.j2_label}_angle)', color=color_rad_j2)
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_x_to_omega_j1',
                label=f'K(vis_x → {c.j1_label}_vel)', color=color_omega_j1, linestyle='--')
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_x_to_omega_j2', 
                label=f'K(vis_x → {c.j2_label}_vel)', color=color_omega_j2, linestyle='--')
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax.set_ylabel('Kalman Gain')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    # Proprioceptive angle gains (Top Right)  
    ax = axes[0, 1]
    ax.set_title('Kalman Gains: Prop Angles → Joint States')
    if 'kalman_gain_prop_rad_j1_to_rad_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j1_to_rad_j1',
                    label=f'K(prop_{c.j1_label}_rad → {c.j1_label}_angle)', color=color_rad_j1)
    if 'kalman_gain_prop_rad_j1_to_rad_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j1_to_rad_j2',
                    label=f'K(prop_{c.j1_label}_rad → {c.j2_label}_angle)', color=color_rad_j2, alpha=0.7)
    if 'kalman_gain_prop_rad_j2_to_rad_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j2_to_rad_j1',
                    label=f'K(prop_{c.j2_label}_rad → {c.j1_label}_angle)', color=color_rad_j1, alpha=0.7)
    if 'kalman_gain_prop_rad_j2_to_rad_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j2_to_rad_j2',
                    label=f'K(prop_{c.j2_label}_rad → {c.j2_label}_angle)', color=color_rad_j2)
    if 'kalman_gain_prop_rad_j1_to_omega_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j1_to_omega_j1',
                    label=f'K(prop_{c.j1_label}_rad → {c.j1_label}_vel)', color=color_omega_j1, linestyle='--')
    if 'kalman_gain_prop_rad_j1_to_omega_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j1_to_omega_j2',
                    label=f'K(prop_{c.j1_label}_rad → {c.j2_label}_vel)', color=color_omega_j2, linestyle='--', alpha=0.7)
    if 'kalman_gain_prop_rad_j2_to_omega_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j2_to_omega_j1',
                    label=f'K(prop_{c.j2_label}_rad → {c.j1_label}_vel)', color=color_omega_j1, linestyle='--', alpha=0.7)
    if 'kalman_gain_prop_rad_j2_to_omega_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_rad_j2_to_omega_j2',
                    label=f'K(prop_{c.j2_label}_rad → {c.j2_label}_vel)', color=color_omega_j2, linestyle='--')
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    # Middle row: Visual Y Kalman gains (left) and Proprioceptive velocity Kalman gains (right)
    # Visual Y gains (Middle Left) - moved from top right
    ax = axes[1, 0] 
    ax.set_title('Kalman Gains: Visual Y → Joint States')
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_y_to_rad_j1',
                 label=f'K(vis_y → {c.j1_label}_angle)', color=color_rad_j1)
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_y_to_rad_j2',
                 label=f'K(vis_y → {c.j2_label}_angle)', color=color_rad_j2)
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_y_to_omega_j1',
                 label=f'K(vis_y → {c.j1_label}_vel)', color=color_omega_j1, linestyle='--')
    sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_vis_y_to_omega_j2',
                 label=f'K(vis_y → {c.j2_label}_vel)', color=color_omega_j2, linestyle='--')
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax.set_ylabel('Kalman Gain')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    # Proprioceptive velocity gains (Middle Right)
    ax = axes[1, 1]
    ax.set_title('Kalman Gains: Prop Velocities → Joint States')
    if 'kalman_gain_prop_omega_j1_to_rad_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j1_to_rad_j1',
                    label=f'K(prop_{c.j1_label}_vel → {c.j1_label}_angle)', color=color_rad_j1)
    if 'kalman_gain_prop_omega_j1_to_rad_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j1_to_rad_j2',
                    label=f'K(prop_{c.j1_label}_vel → {c.j2_label}_angle)', color=color_rad_j2, alpha=0.7)
    if 'kalman_gain_prop_omega_j2_to_rad_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j2_to_rad_j1',
                    label=f'K(prop_{c.j2_label}_vel → {c.j1_label}_angle)', color=color_rad_j1, alpha=0.7)
    if 'kalman_gain_prop_omega_j2_to_rad_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j2_to_rad_j2',
                    label=f'K(prop_{c.j2_label}_vel → {c.j2_label}_angle)', color=color_rad_j2)
    if 'kalman_gain_prop_omega_j1_to_omega_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j1_to_omega_j1',
                    label=f'K(prop_{c.j1_label}_vel → {c.j1_label}_vel)', color=color_omega_j1, linestyle='--')
    if 'kalman_gain_prop_omega_j1_to_omega_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j1_to_omega_j2',
                    label=f'K(prop_{c.j1_label}_vel → {c.j2_label}_vel)', color=color_omega_j2, linestyle='--', alpha=0.7)
    if 'kalman_gain_prop_omega_j2_to_omega_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j2_to_omega_j1',
                    label=f'K(prop_{c.j2_label}_vel → {c.j1_label}_vel)', color=color_omega_j1, linestyle='--', alpha=0.7)
    if 'kalman_gain_prop_omega_j2_to_omega_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='kalman_gain_prop_omega_j2_to_omega_j2',
                    label=f'K(prop_{c.j2_label}_vel → {c.j2_label}_vel)', color=color_omega_j2, linestyle='--')
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)

    # Bottom row: Innovation analysis and measurement availability
    # Innovation magnitudes (Bottom Left)
    ax = axes[2, 0]
    ax.set_title('Visual Innovation Magnitudes')
    if 'innovation_vis_x' in results_df.columns and 'innovation_vis_y' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='innovation_vis_x',
                    label='Visual X Innovation', color='purple')
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='innovation_vis_y', 
                    label='Visual Y Innovation', color='magenta')
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax.set_ylabel('Innovation (m)')
    ax.set_xlabel('time_run (s)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    # Measurement availability and update contributions (Bottom Right)
    ax = axes[2, 1]
    ax.set_title('Proprioceptive Innovation Magnitudes')
    
    # Show proprioceptive innovations instead of visual availability
    if 'innovation_prop_rad_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='innovation_prop_rad_j1',
                    label=f'Prop {c.j1_label} Angle', color=color_rad_j1)
    if 'innovation_prop_rad_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='innovation_prop_rad_j2',
                    label=f'Prop {c.j2_label} Angle', color=color_rad_j2)
    if 'innovation_prop_omega_j1' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='innovation_prop_omega_j1',
                    label=f'Prop {c.j1_label} Velocity', color=color_omega_j1, linestyle='--')
    if 'innovation_prop_omega_j2' in results_df.columns:
        sns.lineplot(ax=ax, data=results_df, x='time_run', y='innovation_prop_omega_j2',
                    label=f'Prop {c.j2_label} Velocity', color=color_omega_j2, linestyle='--')
    
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax.set_ylabel('Innovation (rad, rad/s)')
    ax.set_xlabel('time_run (s)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        plt.savefig(output_filename, format=file_type)
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)

def plot_visual_vs_proprioceptive_comparison(results_visual_df, results_no_visual_df, file_type = "pdf",
                                            output_filename="outputs/visual_vs_proprioceptive_comparison_plot"):
    """
    Compares joint velocity estimation performance between conditions with and without visual feedback.
    Shows how visual feedback affects posterior uncertainty and estimation accuracy.
    
    Grid: 2 rows, 2 columns
    - Top row: Joint velocity estimation uncertainty (posterior sigma) comparison
    - Bottom row: Joint velocity estimation error comparison
    
    Args:
        results_visual_df (pd.DataFrame): Results with visual feedback enabled
        results_no_visual_df (pd.DataFrame): Results with visual feedback disabled
        output_filename (str): Path to save the output PDF file
    """
    output_filename = f"{output_filename}.{file_type}"
    print(f"Generating visual vs proprioceptive comparison plot: {output_filename}")
    
    required_cols = [
        'time_run', 
        'posterior_sigma_omega_j1', 'posterior_sigma_omega_j2',
        'true_omega_j1', 'true_omega_j2', 
        'posterior_omega_j1', 'posterior_omega_j2'
    ]
    
    # Check if required columns exist
    missing_visual = [col for col in required_cols if col not in results_visual_df.columns]
    missing_no_visual = [col for col in required_cols if col not in results_no_visual_df.columns]
    
    if missing_visual or missing_no_visual:
        print(f"Warning: Missing columns for comparison analysis.")
        print(f"Missing from visual feedback data: {missing_visual}")
        print(f"Missing from no visual feedback data: {missing_no_visual}")
        return
    
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(15, 10))
    fig.suptitle('Visual vs Proprioceptive-Only: Joint Velocity Estimation Comparison')
    
    # Define colors
    color_visual = 'blue'
    color_no_visual = 'red'
    color_true = 'black'
    
    # Top row: Posterior uncertainty comparison
    # J1 velocity uncertainty (Top Left)
    ax = axes[0, 0]
    ax.set_title(f'{c.j1_label} Angular Velocity Uncertainty')
    sns.lineplot(ax=ax, data=results_visual_df, x='time_run', y='posterior_sigma_omega_j1',
                label='With Visual Feedback', color=color_visual)
    sns.lineplot(ax=ax, data=results_no_visual_df, x='time_run', y='posterior_sigma_omega_j1', 
                label='Proprioceptive Only', color=color_no_visual)
    ax.set_ylabel('Posterior σ (rad/s)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    # J2 velocity uncertainty (Top Right)
    ax = axes[0, 1] 
    ax.set_title(f'{c.j2_label} Angular Velocity Uncertainty')
    sns.lineplot(ax=ax, data=results_visual_df, x='time_run', y='posterior_sigma_omega_j2',
                label='With Visual Feedback', color=color_visual)
    sns.lineplot(ax=ax, data=results_no_visual_df, x='time_run', y='posterior_sigma_omega_j2',
                label='Proprioceptive Only', color=color_no_visual)
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    # Bottom row: Estimation error comparison
    # Calculate absolute estimation errors
    if ('true_omega_j1' in results_visual_df.columns and 
        'posterior_omega_j1' in results_visual_df.columns):
        results_visual_df['omega_j1_error'] = np.abs(
            results_visual_df['true_omega_j1'] - results_visual_df['posterior_omega_j1'])
        results_no_visual_df['omega_j1_error'] = np.abs(
            results_no_visual_df['true_omega_j1'] - results_no_visual_df['posterior_omega_j1'])
    
    if ('true_omega_j2' in results_visual_df.columns and 
        'posterior_omega_j2' in results_visual_df.columns):
        results_visual_df['omega_j2_error'] = np.abs(
            results_visual_df['true_omega_j2'] - results_visual_df['posterior_omega_j2'])
        results_no_visual_df['omega_j2_error'] = np.abs(
            results_no_visual_df['true_omega_j2'] - results_no_visual_df['posterior_omega_j2'])
    
    # J1 velocity error (Bottom Left)
    ax = axes[1, 0]
    ax.set_title(f'{c.j1_label} Angular Velocity Estimation Error')
    if 'omega_j1_error' in results_visual_df.columns:
        sns.lineplot(ax=ax, data=results_visual_df, x='time_run', y='omega_j1_error',
                    label='With Visual Feedback', color=color_visual, alpha=0.7)
        sns.lineplot(ax=ax, data=results_no_visual_df, x='time_run', y='omega_j1_error',
                    label='Proprioceptive Only', color=color_no_visual, alpha=0.7)
    ax.set_ylabel('|Error| (rad/s)')
    ax.set_xlabel('time_run (s)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    # J2 velocity error (Bottom Right)
    ax = axes[1, 1]
    ax.set_title(f'{c.j2_label} Angular Velocity Estimation Error')
    if 'omega_j2_error' in results_visual_df.columns:
        sns.lineplot(ax=ax, data=results_visual_df, x='time_run', y='omega_j2_error',
                    label='With Visual Feedback', color=color_visual, alpha=0.7)
        sns.lineplot(ax=ax, data=results_no_visual_df, x='time_run', y='omega_j2_error', 
                    label='Proprioceptive Only', color=color_no_visual, alpha=0.7)
    ax.set_xlabel('time_run (s)')
    ax.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        plt.savefig(output_filename, format=file_type)
        print(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)

def analyze_visual_feedback_impact_statistics(results_df, output_filename="outputs/visual_feedback_impact_stats.txt"):
    """
    Calculates and saves summary statistics about how visual feedback impacts joint velocity estimation.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame with visual feedback impact data
        output_filename (str): Path to save the statistics text file
    """
    print(f"Calculating visual feedback impact statistics: {output_filename}")
    
    stats_lines = []
    stats_lines.append("Visual Feedback Impact on Joint Velocity Estimation - Summary Statistics")
    stats_lines.append("=" * 80)
    stats_lines.append("")
    
    # Check for required columns
    required_cols = [
        'kalman_gain_vis_x_to_omega_j1', 'kalman_gain_vis_x_to_omega_j2',
        'kalman_gain_vis_y_to_omega_j1', 'kalman_gain_vis_y_to_omega_j2',
        'cross_cov_vis_x_omega_j1', 'cross_cov_vis_x_omega_j2',
        'cross_cov_vis_y_omega_j1', 'cross_cov_vis_y_omega_j2',
        'posterior_sigma_omega_j1', 'posterior_sigma_omega_j2',
        'visual_feedback'
    ]
    
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        stats_lines.append(f"Warning: Missing required columns: {missing_cols}")
        stats_lines.append("Cannot compute all statistics.")
        stats_lines.append("")
    
    # Filter for when visual feedback is available
    if 'visual_feedback' in results_df.columns:
        visual_available = results_df[results_df['visual_feedback'] == True]
        no_visual = results_df[results_df['visual_feedback'] == False]
        
        stats_lines.append(f"Data Summary:")
        stats_lines.append(f"  Total timesteps: {len(results_df)}")
        stats_lines.append(f"  Timesteps with visual feedback: {len(visual_available)} ({100*len(visual_available)/len(results_df):.1f}%)")
        stats_lines.append(f"  Timesteps without visual feedback: {len(no_visual)} ({100*len(no_visual)/len(results_df):.1f}%)")
        stats_lines.append("")
    else:
        visual_available = results_df
        stats_lines.append("Note: visual_feedback column not found, analyzing all data")
        stats_lines.append("")
    
    # Kalman Gain Statistics
    if all(col in visual_available.columns for col in ['kalman_gain_vis_x_to_omega_j1', 'kalman_gain_vis_y_to_omega_j2']):
        stats_lines.append("Kalman Gain Statistics (Visual→Joint Velocity):")
        stats_lines.append("-" * 50)
        
        # Visual X to joint velocities
        k_vis_x_omega_j1 = visual_available['kalman_gain_vis_x_to_omega_j1'].dropna()
        k_vis_x_omega_j2 = visual_available['kalman_gain_vis_x_to_omega_j2'].dropna()
        
        if len(k_vis_x_omega_j1) > 0:
            stats_lines.append(f"Visual X → {c.j1_label} velocity:")
            stats_lines.append(f"  Mean: {k_vis_x_omega_j1.mean():.6f}")
            stats_lines.append(f"  Std:  {k_vis_x_omega_j1.std():.6f}")
            stats_lines.append(f"  Range: [{k_vis_x_omega_j1.min():.6f}, {k_vis_x_omega_j1.max():.6f}]")
            stats_lines.append(f"  % Non-zero: {100 * (k_vis_x_omega_j1.abs() > 1e-9).sum() / len(k_vis_x_omega_j1):.1f}%")
        
        if len(k_vis_x_omega_j2) > 0:
            stats_lines.append(f"Visual X → {c.j2_label} velocity:")
            stats_lines.append(f"  Mean: {k_vis_x_omega_j2.mean():.6f}")
            stats_lines.append(f"  Std:  {k_vis_x_omega_j2.std():.6f}")
            stats_lines.append(f"  Range: [{k_vis_x_omega_j2.min():.6f}, {k_vis_x_omega_j2.max():.6f}]")
            stats_lines.append(f"  % Non-zero: {100 * (k_vis_x_omega_j2.abs() > 1e-9).sum() / len(k_vis_x_omega_j2):.1f}%")
        
        # Visual Y to joint velocities
        k_vis_y_omega_j1 = visual_available['kalman_gain_vis_y_to_omega_j1'].dropna()
        k_vis_y_omega_j2 = visual_available['kalman_gain_vis_y_to_omega_j2'].dropna()
        
        if len(k_vis_y_omega_j1) > 0:
            stats_lines.append(f"Visual Y → {c.j1_label} velocity:")
            stats_lines.append(f"  Mean: {k_vis_y_omega_j1.mean():.6f}")
            stats_lines.append(f"  Std:  {k_vis_y_omega_j1.std():.6f}")
            stats_lines.append(f"  Range: [{k_vis_y_omega_j1.min():.6f}, {k_vis_y_omega_j1.max():.6f}]")
            stats_lines.append(f"  % Non-zero: {100 * (k_vis_y_omega_j1.abs() > 1e-9).sum() / len(k_vis_y_omega_j1):.1f}%")
        
        if len(k_vis_y_omega_j2) > 0:
            stats_lines.append(f"Visual Y → {c.j2_label} velocity:")
            stats_lines.append(f"  Mean: {k_vis_y_omega_j2.mean():.6f}")
            stats_lines.append(f"  Std:  {k_vis_y_omega_j2.std():.6f}")
            stats_lines.append(f"  Range: [{k_vis_y_omega_j2.min():.6f}, {k_vis_y_omega_j2.max():.6f}]")
            stats_lines.append(f"  % Non-zero: {100 * (k_vis_y_omega_j2.abs() > 1e-9).sum() / len(k_vis_y_omega_j2):.1f}%")
        
        stats_lines.append("")
    
    # Cross-Covariance Statistics
    if all(col in visual_available.columns for col in ['cross_cov_vis_x_omega_j1', 'cross_cov_vis_y_omega_j2']):
        stats_lines.append("Cross-Covariance Statistics (Visual↔Joint Velocity):")
        stats_lines.append("-" * 50)
        
        cross_cov_cols = [
            ('cross_cov_vis_x_omega_j1', f'Visual X ↔ {c.j1_label} velocity'),
            ('cross_cov_vis_x_omega_j2', f'Visual X ↔ {c.j2_label} velocity'),
            ('cross_cov_vis_y_omega_j1', f'Visual Y ↔ {c.j1_label} velocity'),
            ('cross_cov_vis_y_omega_j2', f'Visual Y ↔ {c.j2_label} velocity')
        ]
        
        for col_name, description in cross_cov_cols:
            if col_name in visual_available.columns:
                data = visual_available[col_name].dropna()
                if len(data) > 0:
                    stats_lines.append(f"{description}:")
                    stats_lines.append(f"  Mean: {data.mean():.6f}")
                    stats_lines.append(f"  Std:  {data.std():.6f}")
                    stats_lines.append(f"  Range: [{data.min():.6f}, {data.max():.6f}]")
                    stats_lines.append(f"  % Non-zero: {100 * (data.abs() > 1e-9).sum() / len(data):.1f}%")
        
        stats_lines.append("")
    
    # Uncertainty Reduction Analysis
    if ('posterior_sigma_omega_j1' in results_df.columns and 
        'posterior_sigma_omega_j2' in results_df.columns and
        'visual_feedback' in results_df.columns):
        
        stats_lines.append("Uncertainty Analysis:")
        stats_lines.append("-" * 50)
        
        # Compare uncertainty with vs without visual feedback
        visual_sigma_j1 = results_df[results_df['visual_feedback'] == True]['posterior_sigma_omega_j1'].dropna()
        no_visual_sigma_j1 = results_df[results_df['visual_feedback'] == False]['posterior_sigma_omega_j1'].dropna()
        
        visual_sigma_j2 = results_df[results_df['visual_feedback'] == True]['posterior_sigma_omega_j2'].dropna()
        no_visual_sigma_j2 = results_df[results_df['visual_feedback'] == False]['posterior_sigma_omega_j2'].dropna()
        
        if len(visual_sigma_j1) > 0 and len(no_visual_sigma_j1) > 0:
            reduction_j1 = (no_visual_sigma_j1.mean() - visual_sigma_j1.mean()) / no_visual_sigma_j1.mean() * 100
            stats_lines.append(f"{c.j1_label} velocity uncertainty:")
            stats_lines.append(f"  With visual feedback:    {visual_sigma_j1.mean():.6f} ± {visual_sigma_j1.std():.6f} rad/s")
            stats_lines.append(f"  Without visual feedback: {no_visual_sigma_j1.mean():.6f} ± {no_visual_sigma_j1.std():.6f} rad/s")
            stats_lines.append(f"  Reduction: {reduction_j1:.1f}%")
        
        if len(visual_sigma_j2) > 0 and len(no_visual_sigma_j2) > 0:
            reduction_j2 = (no_visual_sigma_j2.mean() - visual_sigma_j2.mean()) / no_visual_sigma_j2.mean() * 100
            stats_lines.append(f"{c.j2_label} velocity uncertainty:")
            stats_lines.append(f"  With visual feedback:    {visual_sigma_j2.mean():.6f} ± {visual_sigma_j2.std():.6f} rad/s")
            stats_lines.append(f"  Without visual feedback: {no_visual_sigma_j2.mean():.6f} ± {no_visual_sigma_j2.std():.6f} rad/s")
            stats_lines.append(f"  Reduction: {reduction_j2:.1f}%")
        
        stats_lines.append("")
    
    # Save statistics to file
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        with open(output_filename, 'w') as f:
            for line in stats_lines:
                f.write(line + '\n')
        print(f"Statistics saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving statistics: {e}")
    
    # Also print key findings to console
    print("\nKey Findings:")
    for line in stats_lines:
        if "Reduction:" in line or "% Non-zero:" in line:
            print(f"  {line.strip()}")

def plot_reaching_trajectories_patterson2017(results_df, folder_path = None, extra_text = None, file_type = "pdf", output_filename = None):
    """
    Plots reaching trajectories from the experiment.
    - Subplot 1: Shows trajectories for all even-numbered trials.
    - Subplot 2: Shows trajectory for the last trial.
    """
    output_filename = output_filename + file_type
    # output_filename = f"{folder_path}/reaching_trajectories.{file_type}"
    print(f"Generating reaching trajectories plot: {output_filename}")

    # Check for required columns
    required_cols = [
        'trial', 'true_hand_x', 'true_hand_y', 'visual_feedback',
        'p_target_home_x', 'p_target_home_y',
        'p_target_out_x', 'p_target_out_y',
        'p_target_final_x', 'p_target_final_y',
        'r_target'
    ]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for reaching trajectories plot: {missing_cols}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle('Reaching Trajectories Analysis')

    # --- Subplot 1: Trials by Visual Feedback ---
    ax1 = axes[0]
    ax1.set_title('Trajectories by Visual Feedback')

    # Get target positions and radius
    home_pos_x = results_df['p_target_home_x'].iloc[0]
    home_pos_y = results_df['p_target_home_y'].iloc[0]
    out_pos_x = results_df['p_target_out_x'].iloc[0]
    out_pos_y = results_df['p_target_out_y'].iloc[0]
    target_radius_out = results_df['r_target_out'].iloc[0]
    target_radius_home = results_df['r_target_home'].iloc[0]


    # Create target circles
    home_circle_x, home_circle_y = create_circle(home_pos_x, home_pos_y, target_radius_home)
    out_circle_x, out_circle_y = create_circle(out_pos_x, out_pos_y, target_radius_out)

    # Plot target circles
    ax1.plot(home_circle_x, home_circle_y, color='green', label='Home Position', zorder=5)
    ax1.plot(out_circle_x, out_circle_y, color='red', label='Out Position', zorder=5)

    # Get all trials and split them by visual feedback availability, excluding the last trial
    all_trials = sorted(results_df['trial'].unique())
    last_trial_num = max(all_trials) if all_trials else None
    
    trials_to_plot = [t for t in all_trials if t % 2 == 0 and t != last_trial_num]

    with_vis_trials = []
    no_vis_trials = []

    for trial_num in trials_to_plot:
        # Check if any step in the trial had visual feedback
        if results_df[results_df['trial'] == trial_num]['visual_feedback'].any():
            with_vis_trials.append(trial_num)
        else:
            no_vis_trials.append(trial_num)
    
    # Plot trials with visual feedback in light green
    if with_vis_trials:
        alphas_vis = np.linspace(0.3, 1.0, len(with_vis_trials))
        for i, trial_num in enumerate(with_vis_trials):
            trial_df = results_df[results_df['trial'] == trial_num]
            sns.lineplot(ax=ax1, x='true_hand_x', y='true_hand_y', data=trial_df,
                        color='lightgreen', legend=False, sort=False)
        ax1.plot([], [], color='lightgreen', label='With Visual Feedback') # Legend entry

    # Plot trajectories for trials without visual feedback
    if no_vis_trials:
        alphas_no_vis = np.linspace(0.15, 1.0, len(no_vis_trials))
        for i, trial_num in enumerate(no_vis_trials):
            trial_df = results_df[results_df['trial'] == trial_num]
            sns.lineplot(ax=ax1, x='true_hand_x', y='true_hand_y', data=trial_df,
                         alpha=alphas_no_vis[i], color='C0', legend=False, sort=False)
        ax1.plot([], [], color='C0', label='Without Visual Feedback') # Legend entry

        # Highlight first, middle, and last of the no-feedback trials
        special_trials = []
        if len(no_vis_trials) > 0:
            special_trials.append(no_vis_trials[0])
        if len(no_vis_trials) > 2:
            special_trials.append(no_vis_trials[len(no_vis_trials) // 2])
        if len(no_vis_trials) > 1:
            special_trials.append(no_vis_trials[-1])
        
        special_trial_alphas = ['deepskyblue', 'royalblue', 'navy']
        
        for i, trial_num in enumerate(sorted(list(set(special_trials)))):
            trial_df = results_df[results_df['trial'] == trial_num]
            sns.lineplot(ax=ax1, x='true_hand_x', y='true_hand_y', data=trial_df,
                         linewidth=5, color=special_trial_alphas[i], legend=False, sort=False, zorder=4)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    ax1.set_aspect('equal', adjustable='box')

    # --- Subplot 2: Last Trial Trajectory ---
    ax2 = axes[1]
    ax2.set_title('Trajectory of Last Trial')

    if last_trial_num is not None:
        last_trial_df = results_df[results_df['trial'] == last_trial_num]

        # Get target positions for the last trial
        home_pos_x_last = last_trial_df['p_target_home_x'].iloc[0]
        home_pos_y_last = last_trial_df['p_target_home_y'].iloc[0]
        out_pos_x_last = last_trial_df['p_target_out_x'].iloc[0]
        out_pos_y_last = last_trial_df['p_target_out_y'].iloc[0]
        final_pos_x_last = last_trial_df['p_target_final_x'].iloc[0]
        final_pos_y_last = last_trial_df['p_target_final_y'].iloc[0]

        # Create target circles for the last trial
        final_circle_x_last, final_circle_y_last = create_circle(final_pos_x_last, final_pos_y_last, target_radius_out)

        # Plot target circles
        ax2.plot(home_circle_x, home_circle_y, color='green', label='Home Position', zorder=5)
        ax2.plot(out_circle_x, out_circle_y, color='red', label='Out Position', zorder=5)
        ax2.plot(final_circle_x_last, final_circle_y_last, color='purple', label='Final Position', zorder=5)

        # Plot trajectory for the last trial
        sns.lineplot(ax=ax2, x='true_hand_x', y='true_hand_y', data=last_trial_df, label=f'Trial {last_trial_num}', sort=False, color='C1')

        # Plot planned trajectory for the last trial
        if 'p_planned_trajectory_x' in last_trial_df.columns and 'p_planned_trajectory_y' in last_trial_df.columns:
            sns.lineplot(ax=ax2, x='p_planned_trajectory_x', y='p_planned_trajectory_y', data=last_trial_df,
                         label='Planned Trajectory', sort=False, color='orange', linestyle='--')

    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.legend(fontsize=GLOBAL_LEGEND_FONTSIZE)
    ax2.set_aspect('equal', adjustable='box')

    # Finalize and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add config text if provided
    if extra_text is not None:
        # Create a new axis for the text box
        text_ax = fig.add_axes([0.1, 0.01, 0.8, 0.05])
        text_ax.axis('off')
        text_ax.text(0.5, 0.5, extra_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=text_ax.transAxes,
                    fontsize=8,
                    bbox=dict(facecolor='white', 
                             edgecolor='gray',
                             alpha=0.8,
                             boxstyle='round,pad=0.5'))
        # Adjust layout to make room for text box
        plt.subplots_adjust(bottom=0.1)
    
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_circular_follower_task(results_df, output_filename=None, extra_text="", file_type="pdf"):
    """
    Plots the circular following task trajectory for trial 0 in a 2x3 grid of subplots,
    each showing a specific time interval.
    """
    required_cols = ['true_hand_x', 'true_hand_y', 'target_x', 'target_y',
                     'circular_target_movement_c_x', 'circular_target_movement_c_y',
                     'circular_target_movement_r', 'r_target', 'time', 'trial', 'visual_feedback']
    if any(col not in results_df.columns for col in required_cols):
        print("Dataframe not fully unpacked or missing required columns for circular follower plot.")
        for col in required_cols:
            if col not in results_df.columns:
                print(f"Missing column: {col}")
        return
    
    trial_0_df = results_df[results_df['trial'] == 0].copy()

    if trial_0_df.empty:
        print("No data for trial 0. Cannot plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    time_intervals = [(2, 5), (5, 6), (6, 10), (10, 14), (14, 18), (18, 22)]
    
    # Task parameters from dataframe
    circle_center_x = trial_0_df['circular_target_movement_c_x'].iloc[0]
    circle_center_y = trial_0_df['circular_target_movement_c_y'].iloc[0]
    circle_center = (circle_center_x, circle_center_y)
    circle_radius = trial_0_df['circular_target_movement_r'].iloc[0]
    target_radius = trial_0_df['r_target'].iloc[0]

    for i, (start_time, end_time) in enumerate(time_intervals):
        ax = axes[i]
        
        # Filter data for the time interval
        time_mask = (trial_0_df['time'] >= start_time) & (trial_0_df['time'] < end_time)
        df_interval = trial_0_df[time_mask]

        ax.set_title(f"Time: {start_time}-{end_time} s")

        if not df_interval.empty:
            # Plot true hand trajectory
            sns.lineplot(data=df_interval, x='true_hand_x', y='true_hand_y', sort=False, ax=ax, label='True Hand Path', color='blue')
            
            # Plot target trajectory
            sns.lineplot(data=df_interval, x='target_x', y='target_y', sort=False, ax=ax, label='Target Path', color='red', linestyle='--')

            # Plot visual hand feedback for 5-6s in the second pane
            if i == 1:
                df_vis = df_interval[df_interval['visual_feedback']]
                if not df_vis.empty and 'vis_hand_x' in df_vis.columns:
                    sns.scatterplot(data=df_vis, x='vis_hand_x', y='vis_hand_y', ax=ax, label='Visual Hand Feedback', color='green', marker='x', s=50)

        # Add target path rings
        outer_ring = patches.Circle(circle_center, circle_radius + target_radius, fill=False, color='gray', linestyle='--', label='Target Boundary')
        inner_ring = patches.Circle(circle_center, circle_radius - target_radius, fill=False, color='gray', linestyle='--')
        ax.add_patch(outer_ring)
        ax.add_patch(inner_ring)
        
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize='small')
        ax.set_xlim(circle_center_x - circle_radius - target_radius - 0.05, circle_center_x + circle_radius + target_radius + 0.05)
        ax.set_ylim(circle_center_y - circle_radius - target_radius - 0.05, circle_center_y + circle_radius + target_radius + 0.05)

    # Hide unused subplots
    if len(time_intervals) < len(axes):
        for i in range(len(time_intervals), len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Circular Following Task Performance (Trial 0)", fontsize=16)

    if output_filename:
        # Add extra_text to the plot if it exists
        if extra_text:
            plt.figtext(0.5, 0.01, extra_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

        # Ensure the filename has the correct extension
        if not output_filename.endswith(f".{file_type}"):
            output_filename += f".{file_type}"
        
        plt.savefig(output_filename, bbox_inches='tight')
    
    plt.close(fig)

def plot_mean_circular_follower_task(results_df, output_filename=None, extra_text="", file_type="pdf"):
    """
    Plots the mean circular following task trajectory across multiple trials and visual offsets in a 2x3 grid.
    Each subplot shows the mean path for a specific time interval.
    Mean paths are colored by visual offset.
    """
    required_cols = ['true_hand_x', 'true_hand_y', 'time', 'trial', 'dt',
                     'visual_offset_x', 'visual_offset_y',
                     'circular_target_movement_c_x', 'circular_target_movement_c_y',
                     'circular_target_movement_r', 'r_target']
    if any(col not in results_df.columns for col in required_cols):
        print("Dataframe not fully unpacked or missing required columns for mean circular follower plot.")
        for col in required_cols:
            if col not in results_df.columns:
                print(f"Missing column: {col}")
        return

    df = results_df.copy()
    numeric_cols_to_check = ['true_hand_x', 'true_hand_y', 'time', 'dt', 'visual_offset_x', 'visual_offset_y']
    for col in numeric_cols_to_check:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols_to_check, inplace=True)

    unique_offsets = df[['visual_offset_x', 'visual_offset_y']].drop_duplicates().values
    n_offsets = len(unique_offsets)
    
    colors = cm.viridis(np.linspace(0, 1, n_offsets))

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle("Mean Circular Following Task Performance by Visual Offset", fontsize=16)

    time_intervals = [(2, 5), (5, 6), (6, 10), (10, 14), (14, 18), (18, 22)]

    # Task parameters from dataframe (assumed to be the same across all)
    circle_center_x = df['circular_target_movement_c_x'].iloc[0]
    circle_center_y = df['circular_target_movement_c_y'].iloc[0]
    circle_center = (circle_center_x, circle_center_y)
    circle_radius = df['circular_target_movement_r'].iloc[0]
    target_radius = df['r_target'].iloc[0]

    for offset_idx, offset in enumerate(unique_offsets):
        offset_df = df[(df['visual_offset_x'] == offset[0]) & (df['visual_offset_y'] == offset[1])]
        
        unique_trials = sorted(offset_df['trial'].unique())
        n_trials = len(unique_trials)
        if n_trials == 0:
            continue

        # --- Calculate Mean Trajectory for this offset ---
        dt = offset_df['dt'].iloc[0]
        min_time = offset_df['time'].min()
        max_time = offset_df['time'].max()
        master_time = np.arange(min_time, max_time, dt)
        max_len = len(master_time)

        interp_x = np.zeros((n_trials, max_len))
        interp_y = np.zeros((n_trials, max_len))

        for i, trial_num in enumerate(unique_trials):
            trial_df = offset_df[offset_df['trial'] == trial_num]
            original_time = trial_df['time'].values
            if not np.all(np.diff(original_time) >= 0):
                interp_x[i, :] = np.nan
                interp_y[i, :] = np.nan
                continue
            interp_x[i, :] = np.interp(master_time, original_time, trial_df['true_hand_x'].values)
            interp_y[i, :] = np.interp(master_time, original_time, trial_df['true_hand_y'].values)

        mean_x = np.nanmean(interp_x, axis=0)
        mean_y = np.nanmean(interp_y, axis=0)
        mean_path_df = pd.DataFrame({'time': master_time, 'mean_x': mean_x, 'mean_y': mean_y})
        
        offset_label = f"Offset ({offset[0]:.2f}, {offset[1]:.2f})"

        for i, (start_time, end_time) in enumerate(time_intervals):
            ax = axes[i]
            
            # Plot segment of the mean path
            interval_mean_df = mean_path_df[(mean_path_df['time'] >= start_time) & (mean_path_df['time'] < end_time)]
            if not interval_mean_df.empty:
                ax.plot(interval_mean_df['mean_x'], interval_mean_df['mean_y'], color=colors[offset_idx], linewidth=2.5, label=offset_label, zorder=4+offset_idx)

    for i, (start_time, end_time) in enumerate(time_intervals):
        ax = axes[i]
        ax.set_title(f"Time: {start_time}-{end_time} s")

        outer_ring = patches.Circle(circle_center, circle_radius + target_radius, fill=False, color='gray', linestyle='--')
        inner_ring = patches.Circle(circle_center, circle_radius - target_radius, fill=False, color='gray', linestyle='--')
        ax.add_patch(outer_ring)
        ax.add_patch(inner_ring)

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(circle_center_x - circle_radius - target_radius - 0.05, circle_center_x + circle_radius + target_radius + 0.05)
        ax.set_ylim(circle_center_y - circle_radius - target_radius - 0.05, circle_center_y + circle_radius + target_radius + 0.05)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize='small')

    # Hide unused subplots
    if len(time_intervals) < len(axes):
        for i in range(len(time_intervals), len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_filename:
        if extra_text:
            plt.figtext(0.5, 0.01, extra_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        if not output_filename.endswith(f".{file_type}"):
            output_filename += f".{file_type}"
        plt.savefig(output_filename, bbox_inches='tight')

    plt.close(fig)

def plot_final_errors_and_biases(results, output_filename=None, extra_text="", file_type="pdf"):
    """
    Plots final errors and biases from simulation results.
    """
    # This function is not provided in the code block, so it's left unchanged.
    pass
