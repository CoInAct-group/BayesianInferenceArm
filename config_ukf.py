import numpy as np

# Simulation settings
simulation_seed = None # Will be overwritten by batch_runs_parallel.py if run from there
n_runs = 1
n_trials = 1
trials_to_animate = 1
plot_sigma_points = False
elbow_down = False # down indicates 

# If false, the center of the distribution is used is returned, but still does bayesian inference.
apply_proprioceptive_noise = True 
apply_visual_noise = True
apply_motor_noise = True
passive_movement = False
movement_target = True

# Time settings (Time is given in seconds)
dt = 0.01
max_time_per_trial = 1

use_optimal_control_planner = True
use_receeding_horizon = True
constant_remaining_time = False
min_time_before_movement = 0.
self_terminate = False
frame_decimation = int(np.clip(1,int((0.01/dt)*3), 100))

Q_lqr = np.diag([25.0, 25.0, 10.0, 10.0])
R_lqr = np.diag([0.1, 0.1])
R_lqr_delta = np.diag([0.0, 0.0])
Q_lqr_final_multiplier_position = 1000.0
Q_lqr_final_multiplier_velocity = 10.0

if elbow_down:
    j1_label = 'shoulder-rotation'
    j2_label = 'shoulder-extension'
else:
    j1_label = 'shoulder'
    j2_label = 'elbow'

# Trial settings
# Arm initial position settings
vary_p_shoulder_init = False
if elbow_down:
    p_shoulder_z = 0.4
    apply_gravity_acceleration = False # only relevant for elbow_down = True

p_shoulder_init = np.array([0.2, -0.2])
p_shoulder_init_r = 0.05
p_hand_init = np.array([0.0, 0.0])

# Settings for j1 locked tasks
j1_locked_angle = 30.0 # Angle of shoulder rotation joint when j1 is locked
deg_j2_hand_init = 160.0 # Initial elbow angle of hand in degrees
deg_j2_target_init = np.array([-80.0, -80.0]) # Target angle of hand in degrees, relative to deg_j2_hand_init. Negative values is extension dir relative to hand init.
omega_j2_target_init = 0.0 # Initial angular velocity of hand in degrees/sec

# Target settings
use_optimal_control_planner = True
planned_max_time_target = 1
p_target_list = [[0.1, 0.1], [0.1, 0.2]]
p_target_static = np.array([0.0, 0.38])
p_target_odd = p_hand_init
p_target_even = np.array([0.2, 0.3])
task_type = 'seq_targets'
task_types = [
    'seq_targets',
    'j1_locked_reaching', 'repeated_reaching', 'circular_following', 
    'tapping', 'patterson2017', 'fournerett1997', 'maze_seq_reaching', 
    'patterson2017_j1_locked','kordingwolpert2004', 'seq_reaching', 'roll1982' 
]
r_target = 0.025
v_target = np.array([0.0, 0.0])

# Arm segment lengths (m)
len_upper_arm = 0.33
len_lower_arm = 0.45

# Arm segment masses (kg)
m_upper_arm = 2.0
m_lower_arm = 1.7

# Belief offsets applied to the agent's internal model (planning/inference)
# Set to zero to use true values; non-zero to simulate misspecification
len_upper_arm_believed_offset = 0.0
len_lower_arm_believed_offset = 0.0
m_upper_arm_believed_offset = 0.0
m_lower_arm_believed_offset = 0.0

# Joint angle limits (degrees)
if elbow_down:
    lim_j1_min = 45.0
    lim_j1_max = 175.0
    lim_j2_min = -45.0 
    lim_j2_max = np.rad2deg(np.arccos(p_shoulder_z/(len_upper_arm+len_lower_arm))) # max shoulder flexion which keeps hand on the table with elbow fully extended
else:
    lim_j1_min = -30.0
    lim_j1_max = 135.0
    lim_j2_min = 0.0
    lim_j2_max = 170.0

# Joint max torque (Nm)
torque_j1_max = 50.0
torque_j2_max = 30.0
limit_rfd = True
time_to_max_force = 0.25 # in seconds

# Torque noise is modelled as discretized white noise, i.e. units is Nm/sqrt(s).
ukf_external_force_noise_sigma = 0.1 # Nm/sqrt(s) - Set to 0 to disable external force noise; assumes that all forces are correctly modelled by the agent UKF
torque_sigma_const = 0.0  # Constant torque noise density (e.g., Nm/sqrt(s))
torque_sigma_prop = 0.05   # proportional torque noise density factor (unitless, scales efferent torque, then effectively becomes Nm/sqrt(s))

# UKF disturbance torque augmentation (for estimating unmodeled forces)
est_tau_ext = False
ukf_std_tau_ext_init = 0.0 # Initial SD for disturbance torques (Nm)
ukf_tau_ext_rw_sigma = 0.0 # Random-walk noise density for disturbance torques (Nm/sqrt(s))

damping_factor_j1 = 1.5 # Damping factor 
damping_factor_j2 = 1.5 # Damping factor 

damping_factor_believed_offset_j1 = 0.0 # Damping factor 
damping_factor_believed_offset_j2 = 0.0 # Damping factor 


# Hand friction (task-space) settings
apply_hand_friction = False
# Viscous friction coefficients in task space (N·s/m). Use anisotropic values if desired.
hand_friction_c = 0.0

# Force field settings
force_field_type = 'none' # 'none', 'constant', 'curl'
force_field_on = [0,0]
force_field_vector = np.array([0.0, 0.0])
force_field_magnitude = 0.0

dampen_torque = True

# Bayesian sensor integration parameters
# Sensory noise (proprioceptive and visual) is modelled as discretized white noise, i.e. units is units/sqrt(s).
prop_unit = "rad"
prop_rad_sigma = 0.015
prop_omega_sigma = 0.06


# Proprioceptive interventions
proprioceptive_feedback_rad = True
proprioceptive_feedback_omega = True

proprioceptive_intervention_bool = False
proprioceptive_intervention_on = [0,0]
proprioceptive_intervention_on_angle = 0.0
# Constant offsets (e.g. vibration motors)
proprioceptive_offset_rad_j1 = 0.0
proprioceptive_offset_omega_j1 = 0.0
proprioceptive_multiplier_omega_j1 = 1.0
proprioceptive_offset_rad_j2 = 0.0
proprioceptive_offset_omega_j2 = 0.0
proprioceptive_multiplier_omega_j2 = 1.0
j1_motor_flexion_bias = 1.0 # Set to 0 for no bias. Set to 1.1 for positive 10% bias, only applied in the same direction as the bias
j1_motor_extension_bias = 1.0 # Set to 0 for no bias. Set to 1.1 for positive 10% bias, only applied in the same direction as the bias
j2_motor_flexion_bias = 1.0 # Set to 0 for no bias. Set to 1.1 for positive 10% bias, only applied in the same direction as the bias
j2_motor_extension_bias = 1.0 # Set to 0 for no bias. Set to 1.1 for positive 10% bias, only applied in the same direction as the bias

# Visual parameters
vis_p_sigma = 0.001 # in meters/sqrt(s)
# Interventions
# Visual interventions
# Defines when the virtual hand is visible (Target is always visible)
apply_visual_innovation = True # If False, the visual innovation is not applied to the UKF. Allows the model to "compare" its state to the visual feedback.
visual_feedback = False
visual_feedback_first_step = False
visual_feedback_on = [0,0]
visual_intervention_bool = False
visual_feedback_bool_onset = 0.1
visual_feedback_duration = 0.5
visual_blur = vis_p_sigma
visual_intervention_on = [0,0]
visual_offset = np.array([0.0, 0.0])
visual_feedback_rotation = 0.0 # in degrees, applied independently of visual_feedback_bool, positive is counterclockwise


# UKF Parameters
# UKF Algorithm Parameters
ukf_alpha = 1e-3  # Default alpha (spread of sigma points)
ukf_beta = 2.0    # Default beta (incorporates prior knowledge of state distribution, 2 is optimal for Gaussian)
ukf_kappa = 0.0   # Default kappa (secondary scaling parameter, usually 0 or 3-L)

# Initial State Estimate Covariance (P_ukf diagonals - standard deviations)
# These define the initial uncertainty of the state variables.
# Values are in degrees or degrees/sec and will be converted to radians in the agent.
ukf_std_rad_j1_init = 1.0      # Initial uncertainty for rad_j1 (degrees)
ukf_std_rad_j2_init = 1.0      # Initial uncertainty for rad_j2 (degrees)
ukf_std_omega_j1_init = 1.0    # Initial uncertainty for omega_j1 (degrees/sec)
ukf_std_omega_j2_init = 1.0    # Initial uncertainty for omega_j2 (degrees/sec)

# Process Noise Covariance (Q_ukf diagonals - standard deviations)
# These represent the expected noise/uncertainty in the state dynamics model per time step.
ukf_process_noise_std_rad_j1 = 0.05     # Process noise std for rad_j1 (degrees)
ukf_process_noise_std_rad_j2 = 0.05     # Process noise std for rad_j2 (degrees)
ukf_process_noise_std_omega_j1 = 0.1    # Process noise std for omega_j1 (degrees/sec)
ukf_process_noise_std_omega_j2 = 0.1    # Process noise std for omega_j2 (degrees/sec)

# Batch run settings
# simulation_seed = 42 # Will be overwritten by batch_runs_parallel.py if run from there
batch_run = False      # Indicates if the simulation is part of a batch run