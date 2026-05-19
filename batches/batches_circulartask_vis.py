import numpy as np
import config_ukf as c
import visualisation as vis


batch_name = "circular_task_sim_vis1" # Defines folder name
save_results = True
param_grid = {
    "task_type": ["circular_following"],
    "use_optimal_control_planner": [True],
    "use_receeding_horizon": [True],
    "constant_remaining_time": [True],
    "apply_proprioceptive_noise": [False],
    "apply_visual_noise": [False],
    "apply_motor_noise": [False],
    "planned_max_time_target": [0.2],
    "max_time_per_trial": [8],
    "vary_p_shoulder_init": [True],
    "n_runs": [1],
    # "simulation_seed": [42],
    "n_trials": [1],
    # "visual_offset": [np.array([0.0, 0.0]), np.array([0.05, 0.0]), np.array([-0.05, 0.0])],
    "visual_offset": [np.array([-0.05, 0.0])],
    "vis_p_sigma": [0.001],
    "prop_rad_sigma": [0.015, 0.06],
    "prop_omega_sigma": [0.015, 0.06],
    "prop_unit": ["rad"],
    "visual_feedback": [True],
    "ukf_std_rad_j1_init": [1.0],
    "ukf_std_rad_j2_init": [1.0],
    "ukf_std_omega_j1_init": [1.0],
    "ukf_std_omega_j2_init": [1.0],
    "ukf_external_force_noise_sigma": [0.1],
    "torque_sigma_prop": [0.05]
}

allowed_pairs = [
    {"prop_rad_sigma": 0.015, "prop_omega_sigma": 0.06},
    {"prop_rad_sigma": 0.06, "prop_omega_sigma": 0.015},
    # {"prop_rad_sigma": 0.08, "prop_omega_sigma": 0.015},
    # {"prop_rad_sigma": 0.06, "prop_omega_sigma": 0.06},
]
# Define visualization functions to run for each batch iteration
plot_functions = [  
    vis.plotly_animation3,
    # vis.plot_joint_angles
]
plot_extra_text = [
    "proprioceptive_offset_rad_j1", "proprioceptive_offset_omega_j1", 
    "proprioceptive_offset_rad_j2", "proprioceptive_offset_omega_j2", 
    "j1_motor_flexion_bias", "j1_motor_extension_bias", 
    "j2_motor_flexion_bias", "j2_motor_extension_bias"
]
plot_file_type = "png" # "pdf" or "png" # TODO: add png
reps_resample = 1 # repetitions per parameter setting, spread across different cores, e.g. if param_grid gives 3 combinations and reps = 4, then 12 cores will be used
reps_identical = 1 # repetitions per parameter setting, spread across different cores, e.g. if param_grid gives 3 combinations and reps = 4, then 12 cores will be used


# Define all cols that should be saved after batch run
save_all_cols = False
save_cols = [ 
    "dt", 
    "visual_offset_x", "visual_offset_y", "r_target",
    'seed', 
    'step', 'time', 'trial', 'run', 'run_name',
    'true_hand_x', 'true_hand_y', 'vis_hand_x', 'vis_hand_y','circular_target_rads_cum',
    'posterior_hand_x', 'posterior_hand_y',
    'torque_j1', 'torque_j2', 'torque_j1_ff', 'torque_j2_ff',
    'true_rad_j1', 'true_rad_j2', 'true_omega_j1', 'true_omega_j2', 'true_alpha_j1', 'true_alpha_j2',
    'posterior_rad_j1', 'posterior_rad_j2', 'posterior_omega_j1', 'posterior_omega_j2',
    'posterior_sigma_rad_j1', 'posterior_sigma_omega_j1',
    'posterior_sigma_rad_j2', 'posterior_sigma_omega_j2',
    'vis_hand_j1', 'vis_hand_j2',
    'rad_j1_target', 'rad_j2_target', 
    'target_x', 'target_y', 'vis_p_sigma',
    'visual_feedback',
    'rad_j1_target', 'rad_j2_target',
    'm_upper_arm_believed_offset', 'm_lower_arm_believed_offset',
    ]
manipulated_vars = list(param_grid.keys())

for param in manipulated_vars:
    if param not in save_cols:
        save_cols.append(param)
