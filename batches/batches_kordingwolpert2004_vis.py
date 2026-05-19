import numpy as np
import config_ukf as c
import visualisation as vis


batch_name = "kordingwolpert_sim_vis1" # Defines folder name
save_results = True
param_grid = {
    "task_type": ["kordingwolpert2004"],
    "planned_max_time_target" : [2.0],
    "max_time_per_trial" : [2.0],
    "apply_proprioceptive_noise": [False],
    "apply_visual_noise": [False],
    "apply_motor_noise": [False],
    "r_target" : [0.005],
    "n_runs": [1],
    "n_trials": [1],
    "visual_offset": [np.array([0.0, 0.0]), np.array([0.01, 0.0]), np.array([-0.01, 0.0])],
    "prop_rad_sigma": [0.015, 0.06],
    "prop_omega_sigma": [0.015, 0.06],
    "prop_unit": ["rad"],
    "vis_p_sigma": [0.001],
    # "visual_blur": [0.001, 0.0015, 0.003],
    "visual_blur": [0.001],
    "visual_intervention_bool": [True],
    "visual_feedback_bool_onset": [0.1],
    "visual_feedback_duration": [0.1],
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
]
# Define visualization functions to run for each batch iteration
plot_functions = [  
    vis.plotly_animation2, 
    # vis.plot_joint_angles,
]
plot_file_type = "pdf" # "pdf" or "png" # TODO: add png
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
    ]
manipulated_vars = list(param_grid.keys())

for param in manipulated_vars:
    if param not in save_cols:
        save_cols.append(param)
