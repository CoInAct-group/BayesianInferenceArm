import numpy as np
import config_ukf as c
import visualisation as vis


batch_name = "seq_targets_test_1" # Defines folder name
save_results = True
param_grid = {
    "task_type": ["seq_targets"],
    "p_target_list": [((.1, .2), (.2, .2), (.2, .1))],
    "planned_max_time_target" : [1.0],
    "max_time_per_trial" : [1.0],
    "apply_proprioceptive_noise": [False],
    "apply_visual_noise": [False],
    "apply_motor_noise": [False],
    "r_target" : [0.025],
    # "self_terminate": [True],
    "n_runs": [1],
    "n_trials": [3]
}

# Define visualization functions to run for each batch iteration
plot_functions = [  
    vis.plotly_animation_kin, 
    # vis.plot_joint_angles,
]
plot_file_type = "pdf" # "pdf" or "png" # TODO: add png
reps_resample = 1 # repetitions per parameter setting, spread across different cores, e.g. if param_grid gives 3 combinations and reps = 4, then 12 cores will be used
reps_identical = 1 # repetitions per parameter setting, spread across different cores, e.g. if param_grid gives 3 combinations and reps = 4, then 12 cores will be used


# Define all cols that should be saved after batch run
save_all_cols = False
save_cols = [
    "dt", "r_target",
    'seed', 
    'step', 'time', 'trial', 'run', 'run_name',
    'true_hand_x', 'true_hand_y', 'posterior_hand_x', 'posterior_hand_y',
    'torque_j1', 'torque_j2', 'torque_j1_ff', 'torque_j2_ff',
    'true_rad_j1', 'true_rad_j2', 'true_omega_j1', 'true_omega_j2', 'true_alpha_j1', 'true_alpha_j2',
    'rad_j1_target', 'rad_j2_target', 
    'target_x', 'target_y', 
    'rad_j1_target', 'rad_j2_target',
    ]
manipulated_vars = list(param_grid.keys())

for param in manipulated_vars:
    if param not in save_cols:
        save_cols.append(param)
