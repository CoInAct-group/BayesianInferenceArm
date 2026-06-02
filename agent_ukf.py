import pandas as pd
import numpy as np
import config_ukf as c
import time
import os
from scipy.linalg import solve_discrete_are

class Agent:
    def __init__(self):
        """Initializes the Agent with parameters from the config file and sets up initial state."""
        # Create a random number generator instance
        if hasattr(c, 'simulation_seed') and c.simulation_seed is not None:
            self.seed = c.simulation_seed
        else:
            self.seed = int(time.time() * 1000) + os.getpid() 

        self.rng = np.random.default_rng(self.seed) 
        # Simulation parameters
        self.n_trials = c.n_trials
        self.n_runs = c.n_runs
        self.dt = c.dt
        self.run = 0
        self.trial = 0
        self.time_run = 0.0
        self.n_steps_max = int(c.max_time_per_trial / c.dt) + 1

        # Results storage
        self.results_columns = None
        self.results = None 
        self.row_saver = 0  

    def initiate_state(self):
        self.step = 0
        self.step_run = 0
        self.time = 0.0 
        self.dampen_torque = c.dampen_torque
        self.force_field_type = c.force_field_type
        if self.force_field_type != 'none':
            self.force_field_on = c.force_field_on
            self.force_field_vector = c.force_field_vector
            self.force_field_magnitude = c.force_field_magnitude
        self.apply_hand_friction = c.apply_hand_friction

        # Agent parameters
        self.elbow_down = c.elbow_down
        self.trial_ended_by_agent = False
        self.self_terminate = c.self_terminate

        # Arm parameters - Static
        if self.elbow_down:
            self.p_shoulder_z = c.p_shoulder_z
        
        self.p_shoulder = np.array(c.p_shoulder_init, dtype=float)

        # Store true physical parameters from config
        self.true_len_upper_arm = c.len_upper_arm
        self.true_len_lower_arm = c.len_lower_arm
        self.true_m_upper_arm = c.m_upper_arm
        self.true_m_lower_arm = c.m_lower_arm

        self.lim_j1_min = np.deg2rad(c.lim_j1_min)
        self.lim_j1_max = np.deg2rad(c.lim_j1_max)
        self.lim_j2_min = np.deg2rad(c.lim_j2_min)
        self.lim_j2_max = np.deg2rad(c.lim_j2_max)

        # Arm parameters - Initial Cartesian State 
        self.p_hand_init = np.array(c.p_hand_init, dtype=float)
        self.p_elbow = np.array([np.nan, np.nan])
        self.p_hand = self.p_hand_init 
        self.p_hand_prev = self.p_hand
        self.v_hand = np.array([0.0, 0.0])
        self.rad_j2_hand_init = np.deg2rad(c.deg_j2_hand_init)
        
        # Arm parameters - Dynamic
        self.rad_j1 = np.nan # Initialized later by IK
        self.rad_j2 = np.nan    # Initialized later by IK
        self.omega_j1 = 0.0 
        self.omega_j2 = 0.0 
        self.alpha_j1 = 0.0 
        self.alpha_j2 = 0.0 

        # Torque parameters
        self.torque_j1 = 0.0
        self.torque_j2 = 0.0
        self.torque_j3 = 0.0

        # Joint torque limits
        self.torque_j1_max = c.torque_j1_max
        self.torque_j2_max = c.torque_j2_max
        self.rfd_j1_max = self.torque_j1_max * (self.dt/c.time_to_max_force)
        self.rfd_j2_max = self.torque_j2_max * (self.dt/c.time_to_max_force)
        self.limit_rfd = c.limit_rfd

        self.damping_factor_j1 = c.damping_factor_j1
        self.damping_factor_j2 = c.damping_factor_j2

        self.setup_target()
        self.setup_true_state()

    def setup_target(self):
        # Target parameters
        self.p_target_list = c.p_target_list
        self.task_type = c.task_type
        self.j1_locked = True if "j1_locked" in self.task_type else False
        self.j1_locked_angle_rad = np.deg2rad(c.j1_locked_angle)
        self.p_target = np.array(c.p_target_static, dtype=float)
        self.p_target_prev = self.p_target # previous step target position
        self.v_target = c.v_target
        self.r_target = c.r_target
        self.r_target_out = None
        self.r_target_home = None
        self.max_time_per_trial = c.max_time_per_trial
        self.p_target_final = None
        self.p_target_home = None
        self.p_target_out = None
        self.min_time_before_movement = c.min_time_before_movement

        # Target parameters in joint space
        self.rad_j1_target, self.rad_j2_target, _ = self.inverse_kinematics(self.p_target, clip_limits=False, true_physics=True)
        self.rad_j1_target_radius = 0.0
        self.rad_j2_target_radius = 0.0
        self.omega_j1_target = 0.0
        self.omega_j2_target = 0.0

        # Trajectory planning parameters: Simplified planner
        self.planned_max_time_target = c.planned_max_time_target  # Maximum time allowed to reach target
        self.trajectory_planned = False
        self.trajectory_start_time = 0.0
        self.trajectory_current_pos = None
        self.trajectory_target_pos = None # This will represent the end point of the current segment plan
        self.trajectory_origin_pos = None # Start point of the current segment plan
        self.trajectory_start_vel_joint = None # Initial joint velocities at the start of the trajectory segment
        self.trajectory_k_hump_joint = None # Scaling factor for the Beta PDF velocity hump
        self.planned_segment_displacement = None # Displacement vector for the current segment plan

        # Trajectory planning parameters: Optimal control planner
        self.use_receeding_horizon = c.use_receeding_horizon
        self.constant_remaining_time = c.constant_remaining_time
        self.planned_optimal_states = None
        self.planned_optimal_torques = None
        self.torque_j1_ff = 0.0
        self.torque_j2_ff = 0.0
        self.planned_state = np.array([np.nan, np.nan, np.nan, np.nan])
        self.Q_lqr = c.Q_lqr      # State cost (pos, vel)
        self.Q_lqr_final_multiplier_position = c.Q_lqr_final_multiplier_position
        self.Q_lqr_final_multiplier_velocity = c.Q_lqr_final_multiplier_velocity
        self.R_lqr = c.R_lqr   
        self.R_lqr_delta = c.R_lqr_delta
        self.target_moved = False

        # Intermediate target values for next step, these are used to guide movement
        self.rad_j1_target_intermediate = np.nan
        self.rad_j2_target_intermediate = np.nan
        self.omega_j1_target_intermediate = 0.0
        self.omega_j2_target_intermediate = 0.0

        # Intermediate target values for this step, these are used to line up plotting of intermediate targets (otherwise they will be 1 off)
        self.rad_j1_target_intermediate_this_step = self.rad_j1_target_intermediate
        self.rad_j2_target_intermediate_this_step = self.rad_j2_target_intermediate
        self.omega_j1_target_intermediate_this_step = self.omega_j1_target_intermediate
        self.omega_j2_target_intermediate_this_step = self.omega_j2_target_intermediate

        # Trajectory in cartesian space
        self.p_planned_trajectory = np.array([np.nan, np.nan])
        self.v_planned_trajectory = np.array([np.nan, np.nan])

        # Circular following task parameters
        self.circular_target_movement_r = 0.0
        self.circular_target_movement_c = 0.0
        self.circular_target_movement_time = 0.0
        self.circular_target_rads_cum = 0.0

    def setup_true_state(self): 
        """
        Sets up the initial state of the agent.
        """

        self.rad_j1, self.rad_j2, limits_violated_clipped = self.inverse_kinematics(self.p_hand_init, clip_limits=True)
        if limits_violated_clipped:
            raise ValueError(f"Initial hand position resulted in limits violation: j1={np.rad2deg(self.rad_j1):.2f}, j2={np.rad2deg(self.rad_j2):.2f}, p_shoulder={self.p_shoulder}")
        self.p_elbow, self.p_hand = self.forward_kinematics(self.rad_j1, self.rad_j2) # Calculate p_hand from self.rad_j1/j2
        self.p_hand_prev = self.p_hand
        self.I_j1, _, self.I_j2, _ = self.calculate_moments_of_inertia(self.rad_j2)
        torque_g_j1_mu, _ = self.calculate_torque_due_to_gravity(
            joint_index=0, current_rad_j1=self.rad_j1, current_rad_j2=self.rad_j2
        )
        torque_g_j2_mu, _ = self.calculate_torque_due_to_gravity(
            joint_index=1, current_rad_j1=self.rad_j1, current_rad_j2=self.rad_j2
        )
        self.torque_j1 = -torque_g_j1_mu
        self.torque_j2 = -torque_g_j2_mu
        self.torque_j1_efferent = self.torque_j1
        self.torque_j2_efferent = self.torque_j2
        self.tau_ext_j1 = 0.0
        self.tau_ext_j2 = 0.0

    def final_initialization(self):
        """
        Final initialization of the agent.
        """
        self.apply_torques_to_joints(0, 0)
        self.target_state_in_joint_space_exact()

    def start_new_run(self, run):
        """Prepares the agent for a new run by resetting its state and trial count."""
        self.reset()
        self.final_initialization()
        self.run = run
        self.time_run = 0.0

    def start_new_trial(self, trial):
        """
        Prepares the agent for a new trial.
        Resets and step count.
        If target type is 'tapping', alternates target position based on trial number.
        """
        self.trial = trial
        self.target_reached = False
        self.trial_ended_by_agent = False

        if self.task_type == "seq_targets":
            self.setup_seq_targets()

        elif self.task_type in c.task_types:
            raise ValueError(f"Task type {c.task_type}, is not yet implemented in the setup_task method.")
        else:
            raise ValueError(f"Invalid target type: {c.task_type}, must be one of: {c.task_types}")

        self.target_state_in_joint_space_exact()

    def start_new_step(self, step, step_run):
        """
        Executes one simulation step.
        Updates time, target position, visual and proprioceptive representations, moments of inertia,
        target state in joint space, expected and sampled sensory inputs, estimates arm and target states,
        calculates ideal and Bayesian torques, applies torques, updates arm kinematics, and saves the step.
        """

        self.step = step
        self.step_run = step_run
        self.update_time()
        # Update state of the arm from previous step
        self.apply_torques_to_joints(self.torque_j1, self.torque_j2)
        self.p_elbow, self.p_hand = self.forward_kinematics(self.rad_j1, self.rad_j2, true_physics=True)
        self.v_hand = self.update_velocity_hand(true_physics=True)
        # self.I_j1, _, self.I_j2, _ = self.calculate_moments_of_inertia(self.rad_j2) # Calc new moments of inertia

        # Move target and update target state in joint space
        self.move_target()
        self.target_state_in_joint_space_exact()

        # Update target and visual and proprioceptive representations of target and arm (incl. experimental interventions)
        # self.update_visual_and_proprioceptive_representations()

        # Predict expected sensory inputs
        # self.update_process_noise() # Expected process (acceleration) noise, given previous efferent torque and state
        # self.ukf_predict(torque_j1_efferent=self.torque_j1_efferent, torque_j2_efferent=self.torque_j2_efferent)

        # Sample proprioceptive and visual input
        # self.ukf_update()


        self.torque_j1, self.torque_j2 = self.calc_torques(current_state = np.array([self.rad_j1, self.rad_j2, self.omega_j1, self.omega_j2], dtype=float))


        # self.torque_j1, self.torque_j2 = self.motor_noise(
        #             self.torque_j1_efferent, self.torque_j2_efferent, 
        #             self.torque_j1_sigma_scaled, self.torque_j2_sigma_scaled)

        self.check_target_reached() 
        self.save_step() 
        # self.debug()

    def debug(self):
        if self.time > 2.8 and self.time < 3.1:
            print(f"time: {self.time}, vis_feedback: {self.visual_feedback}")
            
    def setup_seq_targets(self):
        """
        Set a sequence of target locations, given a list. Once within a trigger dist of each target, shift to next target in sequence. 
        """
        self.p_hand_init = np.array([0.0, 0.0])
        n_targets = len(self.p_target_list)
        self.p_target = np.array(self.p_target_list[self.trial])
        if self.trial == n_targets-1:
            self.self_terminate = False
        else:
            self.self_terminate = True

    def setup_circular_following_task(self):
        """
        Sets up a trial for the circular following task.
        """
        self.reset()
        self.p_hand_init = np.array([0.0, 0.0])
        self.r_target = 0.025
        self.circular_target_movement_r = 0.15
        self.circular_target_movement_c = np.array([0, self.circular_target_movement_r])
        self.circular_target_rads_cum = (np.pi/2)*3
        p_target_start = self.circular_target_movement_c + np.array([0, self.circular_target_movement_r])
        self.rad_j1_target, self.rad_j2_target, _ = self.inverse_kinematics(p_target_start, clip_limits=False, true_physics=True)
        self.circular_target_movement_time = 4.0
        self.self_terminate = False
        self.self_initiate_movement = False
        self.min_time_before_movement = 0.0

        # Set visual feedback
        self.visual_feedback_all_steps = np.repeat(False, self.n_steps_max)
        self.visual_feedback_all_steps[int(3.0/self.dt):int(4.0/self.dt)] = True
        self.visual_intervention_bool_all_steps = np.repeat(True, self.n_steps_max)
        self.visual_intervention_bool_all_steps[int(3.0/self.dt):int(4.0/self.dt)] = True

    def initiate_agent(self):
        """
        Initializes the agent's arm state based on the initial hand position.
        Calculates initial joint angles using inverse kinematics, sets initial j2 and hand positions,
        calculates initial moments of inertia, initializes the results DataFrame, and saves the first step.
        """
        self.initiate_state()
        self.test_ik_validity() # Test validity of IK for hand start position
        self.test_ik_validity(hand_target_position=self.p_target) # Test validity of IK for target position
        self.init_results()

    def test_ik_validity(self, hand_target_position=None):
        """
        Tests the validity of inverse kinematics solutions for various shoulder positions
        sampled on a circle, given a fixed hand target position.

        The shoulder positions are sampled on a circle defined by c.p_shoulder_init (center)
        and c.p_shoulder_init_r (radius) from the config.

        This method assumes the agent has been initialized (e.g., via __init__ and initiate_state())
        so that arm parameters (lengths, joint limits) and default p_hand_init are available.

        Args:
            hand_target_position (np.ndarray, optional): The [x, y] target position for the hand.
                                                         If None, defaults to self.p_hand_init.
            p_shoulder_arg (np.ndarray, optional): Specific shoulder position to use for each IK call.
                                                   If None, self.p_shoulder (as potentially set by the loop) is used.


        Returns:
            list[tuple[np.ndarray, bool, float, float]]: A list of tuples, where each tuple contains:
                - shoulder_position (np.ndarray): The tested shoulder [x, y] position.
                - is_valid (bool): True if IK solution is within joint limits (not outside_limits), False otherwise.
                - j1_angle (float): The calculated j1 angle in radians (NaN if IK failed or solution invalid).
                - j2_angle (float): The calculated j2 angle in radians (NaN if IK failed or solution invalid).
        """
        _, _, outside_limits = self.inverse_kinematics(self.p_hand, clip_limits=False)
        if outside_limits:
            raise ValueError(f"p_hand ({self.p_hand}) given p_shoulder ({self.p_shoulder}) is outside reachable workspace")
        
        _, _, outside_limits = self.inverse_kinematics(self.p_hand, clip_limits=False)
        if outside_limits:
            raise ValueError(f"p_hand ({self.p_hand}) given p_shoulder ({self.p_shoulder}) is outside reachable workspace")
        
        required_attrs = ['p_hand_init', 'elbow_down', 'lim_j1_min', 'lim_j1_max', 
                          'lim_j2_min', 'lim_j2_max', 'true_len_upper_arm', 'true_len_lower_arm', 'p_shoulder']
        for attr in required_attrs:
            if not hasattr(self, attr):
                print(f"Error: Required agent attribute '{attr}' is not initialized. "
                      "Ensure agent.initiate_state() has been called.")
                return []

        original_p_shoulder = np.copy(self.p_shoulder) # Save original agent shoulder position

        center_shoulder = np.array(c.p_shoulder_init, dtype=float)
        radius_shoulder = c.p_shoulder_init_r

        if hand_target_position is None:
            if np.isnan(self.p_hand_init).any():
                 print("Error: self.p_hand_init is NaN and no hand_target_position provided.")
                 return []
            target_pos_for_ik = self.p_hand_init
        else:
            target_pos_for_ik = np.array(hand_target_position, dtype=float)
            if np.isnan(target_pos_for_ik).any():
                print("Error: Provided hand_target_position is NaN.")
                return []

        num_points = 72
        results = []

        for i in range(num_points):
            angle = i * (2 * np.pi / num_points)
            x_shoulder = center_shoulder[0] + radius_shoulder * np.cos(angle)
            y_shoulder = center_shoulder[1] + radius_shoulder * np.sin(angle)
            current_p_shoulder = np.array([x_shoulder, y_shoulder])

            j1_result, j2_result, is_valid_ik = np.nan, np.nan, False
            
            try:

                j1_calc, j2_calc, outside_limits_flag = self.inverse_kinematics(
                    target_pos_for_ik, 
                    clip_limits=False, 
                    p_shoulder_arg=current_p_shoulder
                )
                
                if not outside_limits_flag and not np.isnan(j1_calc) and not np.isnan(j2_calc):
                    is_valid_ik = True
                    j1_result = j1_calc
                    j2_result = j2_calc
                else:
                    is_valid_ik = False
                    if not np.isnan(j1_calc): j1_result = j1_calc 
                    if not np.isnan(j2_calc): j2_result = j2_calc


            except ValueError as e:
                is_valid_ik = False # j1_result, j2_result remain NaN

            results.append((np.copy(current_p_shoulder), is_valid_ik, j1_result, j2_result))
        
        self.p_shoulder = original_p_shoulder 
        
        num_valid_results = sum(1 for _, valid, _, _ in results if valid)
        if num_valid_results < num_points:
            # Print invalid shoulder positions and their IK results
            print("\nInvalid shoulder positions and their IK results:")
            for p_shoulder, valid, j1, j2 in results:
                if not valid:
                    print(f"  Shoulder: {p_shoulder}, J1: {np.rad2deg(j1):.1f}({np.rad2deg(self.lim_j1_min):.1f}-{np.rad2deg(self.lim_j1_max):.1f})°, J2: {np.rad2deg(j2):.1f}({np.rad2deg(self.lim_j2_min):.1f}-{np.rad2deg(self.lim_j2_max):.1f})°")
            raise ValueError(f"Only {num_valid_results} out of {num_points} shoulder positions resulted in a valid IK solution (unclipped solution within joint limits).")

    def init_results(self):
        """Initializes the results DataFrame with a fixed set of columns based on agent attributes.
        The DataFrame is pre-allocated to an estimated maximum number of rows.
        `self.results_columns` is populated once.
        `self.row_saver` is reset to 0.
        """
        if self.results_columns is None:
            # Get all current attribute names
            all_vars = list(vars(self).keys())
            # Define the columns to save (exclude DataFrames, lists of columns, etc.)
            self.results_columns = [col for col in all_vars if col not in ['results', 'results_columns']]

        # Create DataFrame using the stored column list
        num_rows = self.n_steps_max * c.n_trials * c.n_runs
        self.results = pd.DataFrame(columns=self.results_columns, index=range(num_rows))
        self.row_saver = 0

    def complete_results(self):
        """Removes unused pre-allocated rows from the results DataFrame."""
        self.results.drop(self.results.index[self.row_saver:], inplace=True)

    def save_step(self):
        """Saves the current state of agent attributes to the results DataFrame.
        Uses the columns defined in `self.results_columns`.
        Increments `self.row_saver`.

        Raises:
            ValueError: If an attribute in `self.results_columns` is missing or if data length mismatches.
            IndexError: If `self.row_saver` is out of bounds for the DataFrame.
        """
        if self.results_columns is None:
             print("Error: Results columns not initialized before saving step.")
             self.init_results() # Attempt to initialize if not done

        # Get values corresponding ONLY to the defined columns
        try:
            values_to_save = [getattr(self, col) for col in self.results_columns]
        except AttributeError as e:
            print(f"Error accessing attribute during save_step: {e}")
            # This might happen if an attribute in self.results_columns was deleted
            raise

        # Assign the values to the current row using the defined columns
        try:
            # Ensure the number of values matches the number of columns
            if len(values_to_save) != len(self.results.columns):
                raise ValueError(f"Data length mismatch: Trying to save {len(values_to_save)} values into {len(self.results.columns)} columns.")

            # Assign data
            self.results.loc[self.row_saver, self.results_columns] = values_to_save
        except ValueError as e:
            print(f"Error saving step {self.row_saver} to DataFrame: {e}")
            print(f"Columns ({len(self.results.columns)}): {self.results.columns.tolist()}")
            print(f"Values ({len(values_to_save)}): {values_to_save}")
            raise 
        except IndexError:
            print(f"Error: row_saver index ({self.row_saver}) likely out of bounds for DataFrame.")
            raise

        self.row_saver += 1

    def reset(self):
        """
        Resets the agent's state to initial conditions for a new trial or run.
        """
        self.initiate_state()

    def generate_circular_target_path(self, cumulative_rads):
         # Calculate position from cumulative angle
        curr_pos = np.array([np.cos(cumulative_rads), 
                             np.sin(cumulative_rads)]) * self.circular_target_movement_r + self.circular_target_movement_c
        return curr_pos

    def move_target(self):
        """
        Updates the target position based on its movement type.
        If `c.circular_target_movement` is True, moves the target along a circular path.
        Otherwise, updates the target position based on `self.v_target` (linear movement).
        """
        if self.j1_locked: # for j1_locked, also does target_state_in_joint_space_exact
            prev_p_target = self.p_target
            rad_j2_target = self.rad_j2_target + self.omega_j2_target * self.dt
            _, self.p_target = self.forward_kinematics(self.j1_locked_angle_rad, rad_j2_target)
            self.v_target = (self.p_target - prev_p_target) / self.dt
            return

        if self.task_type == 'circular_following':
            # Calculate angle increment each step
            angle_increment = -2*np.pi / (self.circular_target_movement_time/self.dt)
            self.circular_target_rads_cum += angle_increment
            self.p_target = self.generate_circular_target_path(self.circular_target_rads_cum)
            self.v_target = (self.p_target - self.p_target_prev) / self.dt
            self.p_target_prev = self.p_target
        else:
            self.p_target = self.p_target + self.v_target * self.dt

        # Use a tolerance-based check instead of exact float comparison
        if not np.all(np.isclose(self.v_target, 0.0)):
            self.target_moved = True
        else:
            self.target_moved = False


    def target_state_in_joint_space_exact(self):
        """
        Calculates the exact target state (position and velocity) in joint space using inverse kinematics.
        Also calculates the current estimated arm position in joint space from the true hand position.
        - `self.rad_j1/j2_target`: Target joint angles from `self.p_target` (clipped).
        - `self.rad_j1/j2_estimated`: Current arm joint angles from `self.p_hand` (not clipped).
        - `self.omega_j1/j2_target`: Target joint velocities based on change in `rad_j1/j2_target`.
        """
        if not self.j1_locked:
            prev_rad_j1_target = self.rad_j1_target
            prev_rad_j2_target = self.rad_j2_target
            
            # Store current p_target and p_shoulder for debugging if IK fails
            current_p_target_for_ik = np.copy(self.p_target)
            # Ensure p_shoulder is current if it can change dynamically (e.g. vary_p_shoulder_init effects)
            # For this function, self.p_shoulder is the one to use for the IK call for the target.
            current_p_shoulder_for_ik = np.copy(self.p_shoulder)

            self.rad_j1_target, self.rad_j2_target, limits_violated_target = self.inverse_kinematics(current_p_target_for_ik, clip_limits=True, true_physics=True)
            
            if np.isnan(self.rad_j1_target) or np.isnan(self.rad_j2_target) or np.isnan(self.omega_j1_target) or np.isnan(self.omega_j2_target):
                print(f"DEBUG target_state: rad_j1_target or rad_j2_target is NaN at step {self.step}, time {self.time:.3f}")
                print(f"  Input p_target to IK: {current_p_target_for_ik}")
                print(f"  Using p_shoulder for IK: {current_p_shoulder_for_ik}")
                print(f"  IK returned limits_violated_flag: {limits_violated_target}")
                # Forcing a re-run of IK with p_shoulder_arg specifically for more debug from IK itself if needed
                # self.inverse_kinematics(current_p_target_for_ik, clip_limits=True, p_shoulder_arg=current_p_shoulder_for_ik)

            if not np.isnan(self.rad_j1_target) and not np.isnan(self.rad_j2_target):
                self.omega_j1_target = (self.rad_j1_target - prev_rad_j1_target) / self.dt
                self.omega_j2_target = (self.rad_j2_target - prev_rad_j2_target) / self.dt
            else:
                self.omega_j1_target = 0.0
                self.omega_j2_target = 0.0
            
        self._calculate_target_radius_in_joint_space()

    def _calculate_target_radius_in_joint_space(self):
        """
        Calculates the approximate radius of the target in joint space.
        This is done by taking points on the circumference of the Cartesian target,
        transforming them to joint space, and finding the maximum deviation from the center.
        """
        p_center = self.p_target
        r = self.r_target

        if r <= 0:
            self.rad_j1_target_radius = 0.0
            self.rad_j2_target_radius = 0.0
            return

        # Sample points on the circumference of the target circle
        points_cartesian = [
            p_center + np.array([r, 0]),   # Right
            p_center + np.array([-r, 0]),  # Left
            p_center + np.array([0, r]),   # Top
            p_center + np.array([0, -r]),  # Bottom
        ]

        # Convert points to joint space
        points_joint = [self.inverse_kinematics(p, clip_limits=True)[0:2] for p in points_cartesian]
        
        # Filter out any NaN results from IK
        points_joint = [p for p in points_joint if not (np.isnan(p[0]) or np.isnan(p[1]))]

        if not points_joint:
            self.rad_j1_target_radius = 0.0
            self.rad_j2_target_radius = 0.0
            return

        rad_j1_center = self.rad_j1_target
        rad_j2_center = self.rad_j2_target

        if np.isnan(rad_j1_center) or np.isnan(rad_j2_center):
            self.rad_j1_target_radius = 0.0
            self.rad_j2_target_radius = 0.0
            return

        # Calculate deviations
        dev_j1 = [abs(p[0] - rad_j1_center) for p in points_joint]
        dev_j2 = [abs(p[1] - rad_j2_center) for p in points_joint]

        self.rad_j1_target_radius = max(dev_j1) if dev_j1 else 0.0
        self.rad_j2_target_radius = max(dev_j2) if dev_j2 else 0.0


    def calc_torques(self, current_state):
        if self.use_receeding_horizon:
            return self.get_trajectory_targets_oct_receeding_horizon(current_state=current_state)
        else:
            raise ValueError("Non-receeding horizon not yet implemented in calc_torques")

    def check_target_reached(self): 
        """
        NOTE Currently uses true target values, skips any target state estimation.
        Checks if the target has been reached in terms of position and velocity.
        Position: Compares estimated hand position to estimated target position.
        """
        # 1. Check position reached
        position_error = np.linalg.norm(self.p_hand - self.p_target)
        position_reached = position_error < self.r_target
            
        # Target is reached if all conditions are met
        self.target_reached = position_reached

        # if (self.target_reached and self.self_terminate) or (self.time >= self.max_time_per_trial) or ((self.time >= self.planned_max_time_target + self.min_time_before_movement+0.1) and self.self_terminate):
        if (self.target_reached and self.self_terminate) or (self.time >= self.max_time_per_trial):
            self.trial_ended_by_agent = True

    def calculate_torque_due_to_gravity(self, joint_index, current_rad_j1, current_rad_j2, true_physics=False):
        """Calculates torque due to gravity for a specific joint.
        The uncertainty (sigma) of the torque itself is considered zero as it only depends on
        joint angles which are treated as deterministic inputs for this calculation.

        Args:
            joint_index (int): 0 for J1 (shoulder rotation), 1 for J2 (shoulder flexion/elbow angle).
            current_rad_j1 (float): The current j1 angle.
            current_rad_j2 (float): The current j2 angle.

        Returns:
            tuple[float, float]: (mean_gravity_torque, sigma_gravity_torque).
                                 The returned sigma is always 0.
                                 Returns (np.nan, np.nan) if configuration is unreachable.
        """
        g = 9.81 # m/s^2
        torque_mu = 0.0
        torque_sigma = 0.0 # No intrinsic uncertainty in gravity torque calculation from angles

        if self.elbow_down:
            if joint_index == 0: # J1: Shoulder rotation in XY plane (around Z-axis)
                # Gravity acts along Z, so no direct torque on J1.
                # current_rad_j1 is not used here as gravity does not directly affect J1 rotation in this configuration.
                torque_mu = 0.0
            
            elif joint_index == 1: # J2: Shoulder flexion/extension in the arm's vertical plane
                L1 = self.true_len_upper_arm
                L2 = self.true_len_lower_arm
                m1 = self.true_m_upper_arm
                m2 = self.true_m_lower_arm
                z_s = self.p_shoulder_z
                rad_j2_flex = current_rad_j2 # Use passed parameter current_rad_j2

                if np.isnan(rad_j2_flex):
                    return np.nan, np.nan # Cannot calculate if current j2 angle is unknown

                # Torque on J2 due to upper arm (m1)
                # Lever arm for CoM of m1 in side view (horizontal distance from shoulder Z-axis to CoM of L1)
                # rad_j2_flex = 0 means upper arm vertical down. sin(0)=0, lever=0.
                # rad_j2_flex = pi/2 means upper arm horizontal. sin(pi/2)=1, lever=L1/2.
                # Torque = -m1*g*(L1/2)*sin(rad_j2_flex) (negative for flexion-reducing torque)
                torque_g_m1_on_j2 = -m1 * g * (L1 / 2.0) * np.sin(rad_j2_flex)
                # Torque on J2 due to lower arm (m2)
                # Need horizontal position of CoM of L2 in side view.
                # 1. Horizontal projection of L1 (L1_x_side)
                #    Angle of L1 from horizontal in side view is (rad_j2_flex - pi/2)
                #    L1_x_side = L1 * cos(rad_j2_flex - pi/2) = L1 * sin(rad_j2_flex)
                L1_x_side = L1 * np.sin(rad_j2_flex)
                
                # 2. Vertical position of elbow relative to shoulder
                elbow_z_rel_shoulder = -L1 * np.cos(rad_j2_flex) 

                # 3. Vertical position of elbow globally (hand at z=0)
                elbow_z_global = z_s + elbow_z_rel_shoulder
                
                # 4. Horizontal projection of L2 (L2_x_side_projected)
                #    L2_x_side_projected^2 + elbow_z_global^2 = L2^2
                squared_vertical_span_L2 = elbow_z_global**2
                val_for_sqrt_L2_x_proj = L2**2 - squared_vertical_span_L2
                
                if val_for_sqrt_L2_x_proj < -1e-9: # Unreachable configuration (allow small tolerance for float precision)
                    return np.nan, np.nan # Cannot calculate gravitational torque if arm can't reach
                L2_x_side_projected = np.sqrt(max(0, val_for_sqrt_L2_x_proj))

                rad_j3_global = np.arcsin(elbow_z_global / L2)
                self.torque_j3 = m2 * g * (L2 / 2.0) * np.cos(rad_j3_global)

                # Lever arm for CoM of m2 in side view
                # Horizontal pos of CoM_L2 = L1_x_side + L2_x_side_projected / 2.0
                lever_arm_m2 = L1_x_side + L2_x_side_projected / 2.0
                torque_g_m2_on_j2 = -m2 * g * lever_arm_m2
                
                total_torque_gravity_j2 = torque_g_m1_on_j2 + torque_g_m2_on_j2
                torque_mu = total_torque_gravity_j2
            else:
                # Should not happen if joint_index is 0 or 1
                return np.nan, np.nan
        else: # elbow_out configuration (planar movement in XY plane)
            # Gravity acts along Z, no torque on J1 or J2 which rotate in/about XY.
            # current_rad_j1 and current_rad_j2 are not used here.
            torque_mu = 0.0
            
        if np.isnan(torque_mu): # Safety check if any intermediate calculation resulted in NaN
            torque_sigma = np.nan
            
        return torque_mu, torque_sigma

    def apply_torques_to_joints(self, torque_j1, torque_j2):
        """
        Applies torques to the joints and updates the arm's state using simplified dynamics.

        Updates angular acceleration, velocity, and position based on input torques
        and calculated moments of inertia. Uses Euler integration.

        Args:
            torque_j1 (float): Net torque applied at the j1 joint (Nm).
            torque_j2 (float): Net torque applied at the j2 joint (Nm).

        """

        torque_j1 += self.tau_ext_j1 # add externally imposed torques, e.g. force field
        torque_j2 += self.tau_ext_j2
        self.rad_j1, self.rad_j2, self.omega_j1, self.omega_j2, self.alpha_j1, self.alpha_j2 = self.update_joint_kinematics(
            self.rad_j1, self.rad_j2, self.omega_j1, self.omega_j2, torque_j1, torque_j2, true_physics=True
        )

    def update_velocity_hand(self, true_physics=True):
        """
        Updates the velocity of the hand in Cartesian space based on the current joint velocities.
        """
        J = self.calculate_jacobian(self.rad_j1, self.rad_j2, true_physics=true_physics)
        q_dot = np.array([self.omega_j1, self.omega_j2], dtype=float)
        v_hand = J @ q_dot
        return v_hand

    def forward_kinematics(self, rad_j1, rad_j2, true_physics=False):
        """
        Calculates and updates elbow and hand cartesian positions based on current joint angles.
        Returns the elbow and hand positions in cartesian space.
        Args:
            rad_j1 (float): Joint 1 angle in radians.
            rad_j2 (float): Joint 2 angle in radians.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - p_elbow (np.ndarray): [x,y] coordinates of the elbow position in meters
                - p_hand (np.ndarray): [x,y] coordinates of the hand position in meters

        Raises:
            ValueError: If calculated hand or elbow positions are NaN.
        """
        if self.elbow_down:
            return self.forward_kinematics_elbow_down(rad_j1, rad_j2, true_physics=true_physics)
        else:
            return self.forward_kinematics_elbow_out(rad_j1, rad_j2, true_physics=true_physics)

    def forward_kinematics_elbow_out(self, rad_j1, rad_j2, true_physics=False):
        """Calculates and updates j2 and hand cartesian positions based on current joint angles.

        Args:
            rad_j1 (float): j1 angle in radians.
            rad_j2 (float): j2 angle in radians (relative to upper arm).

        Returns:
            tuple[np.ndarray, np.ndarray]: (p_elbow, p_hand) Cartesian coordinates.

        Raises:
            ValueError: If calculated hand or j2 positions are NaN.
        """
        # Using lengths in meters for calculation, assuming p_elbow/p_hand should be in meters
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm

        # j2 position (relative to j1 at [0,0])
        j2_x = L1 * np.cos(rad_j1)
        j2_y = L1 * np.sin(rad_j1)
        p_elbow = np.array([j2_x, j2_y]) + self.p_shoulder

        # Hand position (relative to j1 at [0,0])
        # Angle of lower arm in world frame = j1 angle + j2 angle
        # Assumes j2 angle (rad_j2) definition: 0 = straight, positive CCW relative to upper arm
        world_lower_arm_angle = rad_j1 + rad_j2
        hand_x = j2_x + L2 * np.cos(world_lower_arm_angle)
        hand_y = j2_y + L2 * np.sin(world_lower_arm_angle)
        p_hand = np.array([hand_x, hand_y]) + self.p_shoulder
        
        if np.isnan(p_hand).any() or np.isnan(p_elbow).any():
            raise ValueError(f"Hand position is NaN: {p_hand}, j2 position is NaN: {p_elbow}")

        return p_elbow, p_hand
    
    def forward_kinematics_elbow_down(self, rad_j1, rad_j2, true_physics=False):    
        """Calculates and updates elbow and hand cartesian positions based on current joint angles for the elbow down configuration.
        j1 is internal/external rotation of the shoulder (around Z-axis of shoulder frame), 0 rad = arm projected sidewards along +X axis of p_shoulder.
        j2 is shoulder flexion/extension angle in the vertical plane of the arm, 0 rad = upper arm straight down (-Z), positive rad = flexion (upper arm moves towards +X in side view).
        The hand is constrained to the table (z=0 relative to shoulder's XY plane).

        Args:
            rad_j1 (float): j1 angle in radians (internal/external rotation of the shoulder).
            rad_j2 (float): j2 angle in radians (shoulder flexion/extension in the vertical plane of the arm).
        """
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm

        # Side view calculations (local XZ plane of the arm, before j1 rotation)
        # shoulder_x_local = 0 for this side view projection.
        # rad_j2 = 0 means upper arm is vertical, pointing down.
        # Positive rad_j2 flexes the shoulder, moving elbow towards positive X in this local side view.
        # The angle for L1 in this local XZ plane, measured CCW from positive X-axis: rad_j2 - np.pi/2
        # If rad_j2 = 0 (down), angle is -pi/2. cos(-pi/2)=0, sin(-pi/2)=-1. elbow_x_local=0, elbow_z_local = -L1.
        # If rad_j2 = pi/2 (horizontal), angle is 0. cos(0)=1, sin(0)=0. elbow_x_local=L1, elbow_z_local = 0.
        
        elbow_x_local_side = L1 * np.cos(rad_j2 - np.pi/2) # This is L1 * sin(rad_j2)
        elbow_z_local_side = L1 * np.sin(rad_j2 - np.pi/2) # This is -L1 * cos(rad_j2)

        # elbow_z_global is relative to the table plane (z=0 for the hand)
        # self.p_shoulder_z is the height of the shoulder above the table.
        elbow_z_global = self.p_shoulder_z + elbow_z_local_side

        # Projected lengths in the XY plane (top-down view)
        L1_projected = elbow_x_local_side # This is the horizontal reach of L1 in the side view

        # Check reachability: Is the elbow too high/low for the hand to be on the table?
        # The vertical distance the lower arm (L2) must span is |elbow_z_global - 0|
        squared_vertical_span_L2 = elbow_z_global**2
        if L2**2 < squared_vertical_span_L2 - 1e-9: # Added tolerance
            # print(f"Warning: Elbow configuration unreachable. Elbow z_global: {elbow_z_global:.3f}, L2: {L2:.3f}. Required L2_projected would be imaginary.")
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
        
        # If L2**2 == squared_vertical_span_L2, L2_projected is 0 (elbow directly above/below hand).
        # This can happen if L2 == abs(elbow_z_global)
        L2_projected = np.sqrt(max(0, L2**2 - squared_vertical_span_L2)) # max(0,...) handles potential float inaccuracies near zero

        # Top-down view: Apply j1 rotation (internal/external rotation of shoulder)
        # rad_j1 = 0 means arm is projected along +X axis of p_shoulder.
        # Positive rad_j1 is CCW rotation from +X.
        phi_xy = rad_j1

        # Elbow position relative to shoulder's XY position
        p_elbow_x_rel = np.cos(phi_xy) * L1_projected
        p_elbow_y_rel = np.sin(phi_xy) * L1_projected
        
        # Hand position component from L2_projected, relative to elbow's projected XY
        # This component is also rotated by phi_xy to keep the projected arm straight
        p_hand_add_x = np.cos(phi_xy) * L2_projected
        p_hand_add_y = np.sin(phi_xy) * L2_projected

        p_elbow_xy_relative_to_shoulder = np.array([p_elbow_x_rel, p_elbow_y_rel])
        p_hand_xy_relative_to_shoulder = p_elbow_xy_relative_to_shoulder + np.array([p_hand_add_x, p_hand_add_y])
        
        # Add global shoulder XY position
        p_elbow_global_xy = self.p_shoulder + p_elbow_xy_relative_to_shoulder
        p_hand_global_xy = self.p_shoulder + p_hand_xy_relative_to_shoulder
        
        # The Z coordinates are fixed by the setup: elbow is at elbow_z_global, hand is at 0.
        # For a 2D return matching the other FK function, we return only XY.
        # If 3D points were needed:
        # p_elbow_global_3d = np.array([p_elbow_global_xy[0], p_elbow_global_xy[1], elbow_z_global])
        # p_hand_global_3d = np.array([p_hand_global_xy[0], p_hand_global_xy[1], 0.0])

        if np.isnan(p_elbow_global_xy).any() or np.isnan(p_hand_global_xy).any():
            # This might happen if L1_projected or L2_projected became NaN due to sqrt issues not caught,
            # or if rad_j1/rad_j2 are NaN.
            raise ValueError(f"NaN in FK elbow_down: p_elbow_xy={p_elbow_global_xy}, p_hand_xy={p_hand_global_xy}")

        return p_elbow_global_xy, p_hand_global_xy

    def update_time(self):
        """Updates the current simulation time based on the step count and dt."""
        self.time = round(self.step * self.dt, 3)
        self.time_run = round(self.step_run * self.dt, 3)

    def inverse_kinematics(self, cartesian_position, clip_limits=False, p_shoulder_arg=None, true_physics=False):
        if self.elbow_down:
            return self._inverse_kinematics_elbow_down(cartesian_position, clip_limits, p_shoulder_arg=p_shoulder_arg, true_physics=true_physics)
        else:
            return self._inverse_kinematics_elbow_out(cartesian_position, clip_limits, p_shoulder_arg=p_shoulder_arg, true_physics=true_physics)

    def _inverse_kinematics_elbow_out(self, cartesian_position, clip_limits=False, p_shoulder_arg=None, true_physics=True):
        """Calculate inverse kinematics for 2-joint planar arm.

        Given a target position in 2D cartesian coordinates, calculates the required j1 and j2 angles
        to reach that position. Uses geometric solution based on law of cosines.

        If target is beyond maximum reach, projects it onto reachable workspace boundary before calculating angles.
        Final angles are clipped to joint limits defined in config.

        Args:
            cartesian_position (np.ndarray): 2D array [x,y] of target position in cartesian coordinates
            clip_limits (bool): Whether to clip the calculated angles to joint limits.
            p_shoulder_arg (np.ndarray, optional): Specific shoulder position to use. Defaults to self.p_shoulder.


        Returns:
            tuple: (j1_angle, j2_angle, outside_limits_flag) in radians
                  j1_angle: angle of upper arm relative to x-axis (CCW positive)
                  j2_angle: angle of lower arm relative to upper arm (CCW positive)
        """
        p_shoulder_to_use = p_shoulder_arg if p_shoulder_arg is not None else self.p_shoulder
        if np.isnan(p_shoulder_to_use).any():
            raise ValueError(f"Shoulder position is NaN in _inverse_kinematics_elbow_out: {p_shoulder_to_use}")
        if np.isnan(cartesian_position).any():
            raise ValueError(f"Target position is NaN: {cartesian_position}")
        
        # Select arm lengths based on true_physics flag
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm
        
        cartesian_position_relative = cartesian_position - p_shoulder_to_use
        x, y = cartesian_position_relative
        r = np.linalg.norm(cartesian_position_relative)
        theta = np.arctan2(y, x) # theta is the angle between the x-axis and the vector to the target


        if r < 1e-9: # Use a small tolerance instead of r == 0
            print("Warning: IK target is at or very close to the j1.")
            return np.nan, np.nan, True # Indicate issue with outside_limits=True


        # Calculate j2 angle using law of cosines
        # Use selected arm lengths
        L1_sq = L1**2
        L2_sq = L2**2
        r_sq = r**2
        # Argument for arccos to find the internal angle at the j2
        cos_j2_arg = (L1_sq + L2_sq - r_sq) / (2 * L1 * L2)

        min_reach_sq = (L1 - L2)**2
        if r_sq < min_reach_sq and not np.isclose(r_sq, min_reach_sq):
             print(f"Warning: Target distance r={r:.3f} is inside the minimum reach ({abs(L1-L2):.3f}), p_shoulder={p_shoulder_to_use}, p_target={cartesian_position}. Projecting outwards.")
             # Project target onto the inner boundary circle
             scale = np.sqrt(min_reach_sq) / r
             cartesian_position_relative *= scale
             x, y = cartesian_position_relative
             r = np.linalg.norm(cartesian_position_relative) # r is now sqrt(min_reach_sq)
             r_sq = r**2
             theta = np.arctan2(y, x)
             # Recalculate cos_j2_arg for the projected point
             cos_j2_arg = (L1_sq + L2_sq - r_sq) / (2 * L1 * L2)

        # Clamp to valid range to handle numerical imprecision near boundaries
        cos_j2_arg = np.clip(cos_j2_arg, -1.0, 1.0)

        # Calculate the internal angle at the j2 joint (always positive [0, pi])
        j2_internal_angle = np.arccos(cos_j2_arg)

        # Get j2 joint angle relative to the upper arm.
        j2 = np.pi - j2_internal_angle

        # Calculate j1 angle
        cos_beta_arg = (r_sq + L1_sq - L2_sq) / (2 * L1 * r)

        cos_beta_arg = np.clip(cos_beta_arg, -1.0, 1.0)
        beta = np.arccos(cos_beta_arg)

        j1 = theta - beta

        # Check if calculated angles are outside limits 
        j1_clipped = False
        j2_clipped = False
        if (j1 < self.lim_j1_min) or (j1 > self.lim_j1_max):
            j1_clipped = True
        if (j2 < self.lim_j2_min) or (j2 > self.lim_j2_max):
            j2_clipped = True
        # An angle being outside limits doesn't necessarily mean the original
        # target point was unreachable, just that this specific IK solution is out of bounds.
        # outside_limits = j1_clipped or j2_clipped
        outside_limits = j1_clipped or j2_clipped

        # Clip final angles to be within limits if requested
        if clip_limits:
            j1 = np.clip(j1, self.lim_j1_min, self.lim_j1_max)
            j2 = np.clip(j2, self.lim_j2_min, self.lim_j2_max)

        # Check if the final calculated angles are NaN and raise an error if so
        if np.isnan(j1) or np.isnan(j2):
            # Instead of printing, raise a ValueError
            raise ValueError(f"Inverse kinematics resulted in NaN angles: j1={j1}, j2={j2} for target={cartesian_position}")

        # Return angles and the flag indicating if the *calculated* angles were outside limits
        return j1, j2, outside_limits

    def _inverse_kinematics_elbow_down(self, cartesian_position, clip_limits=False, p_shoulder_arg=None, true_physics=False):
        """
        Calculates inverse kinematics for the "elbow_down" configuration.
        Given a target hand position in global XY coordinates (on the table, z=0),
        calculates the required rad_j1 (shoulder rotation) and rad_j2 (shoulder flexion/extension).

        Args:
            cartesian_position (np.ndarray): 2D array [x,y] of target hand position.
            clip_limits (bool): Whether to clip the calculated angles to joint limits.
            p_shoulder_arg (np.ndarray, optional): Specific shoulder position to use. Defaults to self.p_shoulder.

        Returns:
            tuple: (rad_j1, rad_j2, outside_limits_flag)
                   Angles in radians. outside_limits_flag is True if target is unreachable
                   or calculated angles are outside joint limits (before clipping).
                   Returns (np.nan, np.nan, True) on failure/unreachability.
        """
        p_shoulder_to_use = p_shoulder_arg if p_shoulder_arg is not None else self.p_shoulder
        if np.isnan(p_shoulder_to_use).any():
            # This check is good, but to make it more verbose for debugging:
            print(f"DEBUG IK_elbow_down: p_shoulder_to_use is NaN. p_shoulder_arg={p_shoulder_arg}, self.p_shoulder={self.p_shoulder}")
            raise ValueError(f"Shoulder position is NaN in _inverse_kinematics_elbow_down: {p_shoulder_to_use}")
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm

        p_shoulder_xy = p_shoulder_to_use # Use the determined shoulder position
        z_s = self.p_shoulder_z # Height of shoulder above the table

        target_hand_xy_rel = cartesian_position - p_shoulder_xy
        x_rel = target_hand_xy_rel[0]
        y_rel = target_hand_xy_rel[1]

        # Calculate rad_j1 (top-down rotation)
        rad_j1 = np.arctan2(y_rel, x_rel)

        # Calculate L_projected_total (horizontal distance from shoulder's Z-axis to hand)
        L_projected_total = np.linalg.norm(target_hand_xy_rel)

        # Handle singularity: target at shoulder XY and shoulder on table
        if L_projected_total < 1e-9 and abs(z_s) < 1e-9:
            print(f"DEBUG IK_elbow_down: Singularity at shoulder base on table. L_projected_total={L_projected_total:.4g}, z_s={z_s:.4g}")
            print(f"  Inputs: cartesian_position={cartesian_position}, p_shoulder_xy_used={p_shoulder_xy}, z_s={z_s}")
            # For this singularity, rad_j1 is ill-defined (atan2(0,0)=0).
            # rad_j2 has solutions 0 or pi if L1=L2. Otherwise unreachable.
            if abs(L1 - L2) < 1e-9 and L1 > 1e-9: # If L1=L2 and not zero length
                # This case is complex, returning NaN for now. Could define a specific posture.
                return np.nan, np.nan, True
            return np.nan, np.nan, True # Unreachable or ill-defined

        # Distance squared from shoulder to hand in the side-view plane
        d_sh_sq = L_projected_total**2 + z_s**2

        # Denominator for Law of Cosines argument
        denom_cos_gamma_s = 2 * L1 * np.sqrt(d_sh_sq)

        if abs(denom_cos_gamma_s) < 1e-9: # Avoid division by zero (e.g. L1=0 or target at shoulder if z_s!=0)
            print(f"DEBUG IK_elbow_down: Denom for cos_gamma_s near zero. denom_cos_gamma_s={denom_cos_gamma_s:.4g}, L1={L1:.3f}, sqrt(d_sh_sq)={np.sqrt(d_sh_sq):.3f}")
            print(f"  Inputs: cartesian_position={cartesian_position}, p_shoulder_xy_used={p_shoulder_xy}, z_s={z_s}, L_projected_total={L_projected_total}")
            return np.nan, np.nan, True

        cos_gamma_s_arg = (L1**2 + d_sh_sq - L2**2) / denom_cos_gamma_s

        if abs(cos_gamma_s_arg) > 1.0 + 1e-6: # Allow small tolerance for float precision
            print(f"DEBUG IK_elbow_down: Unreachable, cos_gamma_s_arg ({cos_gamma_s_arg:.6f}) out of [-1, 1] range.")
            print(f"  Details: L1={L1:.3f}, L2={L2:.3f}, d_sh_sq={d_sh_sq:.3f} (sqrt(d_sh_sq)={np.sqrt(d_sh_sq):.3f}), denom_cos_gamma_s={denom_cos_gamma_s:.3f}")
            print(f"  L1^2={L1**2:.3f}, d_sh_sq={d_sh_sq:.3f}, L2^2={L2**2:.3f}, Numerator={(L1**2 + d_sh_sq - L2**2):.3f}")
            print(f"  Inputs: cartesian_position={cartesian_position}, p_shoulder_xy_used={p_shoulder_xy}, z_s={z_s}, L_projected_total={L_projected_total}")
            return np.nan, np.nan, True
        
        cos_gamma_s_arg_clipped = np.clip(cos_gamma_s_arg, -1.0, 1.0)
        gamma_s = np.arccos(cos_gamma_s_arg_clipped) # Angle at shoulder in side-view triangle S-E-H

        # Angle of line SH_side w.r.t. positive x-axis in side view
        # (Hand is at z=0, Shoulder at z_s, so z_hand_rel_shoulder_side = -z_s)
        delta_sh = np.arctan2(-z_s, L_projected_total)

        # Two solutions for alpha (angle of L1 w.r.t. +X_side in side view)
        # alpha1 = delta_sh + gamma_s # Corresponds to one elbow configuration
        alpha2 = delta_sh - gamma_s # Corresponds to the other (often "elbow down/inward")
        
        alpha_chosen = alpha2

        rad_j2 = alpha_chosen + np.pi/2

        # Check if calculated angles are outside limits 
        limits_violated = False
        if (rad_j1 < self.lim_j1_min) or (rad_j1 > self.lim_j1_max):
            limits_violated = True
        if (rad_j2 < self.lim_j2_min) or (rad_j2 > self.lim_j2_max):
            limits_violated = True

        if clip_limits:
            rad_j1 = np.clip(rad_j1, self.lim_j1_min, self.lim_j1_max)
            rad_j2 = np.clip(rad_j2, self.lim_j2_min, self.lim_j2_max)
        
        if np.isnan(rad_j1) or np.isnan(rad_j2):
            return np.nan, np.nan, True # Should be caught by earlier checks, but as a safeguard

        return rad_j1, rad_j2, limits_violated

    def calculate_moments_of_inertia(self, rad_j2_input_mu, rad_j2_input_sigma=0.0):
        """
        Calculates the moment of inertia for the J1 and J2 joints based on configuration.
        This is a dispatcher method.

        Args:
            rad_j2_input_mu (float): The mean of the current J2 angle.
            rad_j2_input_sigma (float, optional): The std dev of the current J2 angle. Defaults to 0.0.
        Returns:
            tuple: (I_j1_mu, I_j1_sigma, I_j2_mu, I_j2_sigma) in kg*m^2
        """
        if np.isnan(rad_j2_input_mu): # If mean is NaN, all results are NaN
            raise ValueError("Error: NaN input mean for calculate_moments_of_inertia.")

        current_rad_j2_sigma = rad_j2_input_sigma
        if np.isnan(rad_j2_input_sigma): # If input sigma is NaN, treat as zero for calculations within sub-functions
            current_rad_j2_sigma = 0.0

        if self.elbow_down:
            return self._calculate_moments_of_inertia_elbow_down(rad_j2_input_mu, current_rad_j2_sigma)
        else:
            return self._calculate_moments_of_inertia_elbow_out(rad_j2_input_mu, current_rad_j2_sigma)

    def _calculate_moments_of_inertia_elbow_down(self, rad_j2_shoulder_flexion_mu, rad_j2_shoulder_flexion_sigma=0.0):
        """
        Calculates moments of inertia for J1 and J2 for ELBOW_DOWN configuration.
        J1: Shoulder rotation in XY plane (around Z-axis).
        J2: Shoulder flexion/extension in the vertical plane of the arm.
        Uses numerical differentiation (central differences) for sigma propagation.

        Args:
            rad_j2_shoulder_flexion_mu (float): Mean of shoulder flexion/extension angle in radians.
            rad_j2_shoulder_flexion_sigma (float, optional): Std dev of shoulder flexion/extension angle. Defaults to 0.0.
        Returns:
            tuple: (I_j1_mu, I_j1_sigma, I_j2_mu, I_j2_sigma) in kg*m^2.
                   Returns (np.nan, np.nan, np.nan, np.nan) if mean calculation fails or input mu is NaN.
        """
        
        # Determine effective sigma for J2 to propagate for I_j2's uncertainty.
        # For I_j1's uncertainty, if uncoupled, it will be 0 later.
        sigma_for_Ij2_prop = 0.0
        if not np.isnan(rad_j2_shoulder_flexion_sigma) and rad_j2_shoulder_flexion_sigma > 1e-9:
            sigma_for_Ij2_prop = rad_j2_shoulder_flexion_sigma

        # --- Coupled MOI Calculation --- 
        # I_j1 and I_j2 both depend on rad_j2_shoulder_flexion_mu and its uncertainty.
        I_j1_mu, I_j2_mu = self._core_calculate_moi_elbow_down(rad_j2_shoulder_flexion_mu)

        # Handle NaN from core calculation
        if np.isnan(I_j1_mu) and np.isnan(I_j2_mu): 
            return np.nan, np.nan, np.nan, np.nan
        # Individual NaNs in I_j1_mu or I_j2_mu will lead to NaN sigmas later.

        I_j1_sigma, I_j2_sigma = 0.0, 0.0
        if sigma_for_Ij2_prop > 0.0 and not np.isnan(rad_j2_shoulder_flexion_mu):
            h = 1e-6 
            q_plus = rad_j2_shoulder_flexion_mu + h
            q_minus = rad_j2_shoulder_flexion_mu - h

            I_j1_plus, I_j2_plus = self._core_calculate_moi_elbow_down(q_plus)
            I_j1_minus, I_j2_minus = self._core_calculate_moi_elbow_down(q_minus)

            # Calculate I_j1_sigma
            if not (np.isnan(I_j1_plus) or np.isnan(I_j1_minus)):
                deriv_I_j1 = (I_j1_plus - I_j1_minus) / (2 * h)
                I_j1_sigma = np.abs(deriv_I_j1) * sigma_for_Ij2_prop
            else:
                I_j1_sigma = np.nan 
            
            # Calculate I_j2_sigma
            if not (np.isnan(I_j2_plus) or np.isnan(I_j2_minus)):
                deriv_I_j2 = (I_j2_plus - I_j2_minus) / (2 * h)
                I_j2_sigma = np.abs(deriv_I_j2) * sigma_for_Ij2_prop
            else:
                I_j2_sigma = np.nan
        
        if np.isnan(I_j1_mu): I_j1_sigma = np.nan
        if np.isnan(I_j2_mu): I_j2_sigma = np.nan
        
        return I_j1_mu, I_j1_sigma, I_j2_mu, I_j2_sigma

    def _core_calculate_moi_elbow_down(self, rad_j2_shoulder_flexion_angle):
        """Core logic to calculate I_j1 and I_j2 for elbow_down, given a specific j2 angle.
        Args:
            rad_j2_shoulder_flexion_angle (float): Specific shoulder flexion/extension angle.
        Returns:
            tuple: (I_j1, I_j2) or (np.nan, np.nan) if unreachable.
        """
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm
        m1 = self.true_m_upper_arm
        m2 = self.true_m_lower_arm
        z_s = self.p_shoulder_z

        if np.isnan(rad_j2_shoulder_flexion_angle):
            return np.nan, np.nan

        angle_L1_in_side_view_from_horizontal = rad_j2_shoulder_flexion_angle - np.pi/2
        L1_p = L1 * np.cos(angle_L1_in_side_view_from_horizontal)
        elbow_z_relative_to_shoulder = L1 * np.sin(angle_L1_in_side_view_from_horizontal)
        elbow_z_global = z_s + elbow_z_relative_to_shoulder
        squared_vertical_span_L2 = elbow_z_global**2
        val_for_sqrt_L2p = L2**2 - squared_vertical_span_L2

        if val_for_sqrt_L2p < -1e-9:
            return np.nan, np.nan
        L2_p = np.sqrt(max(0, val_for_sqrt_L2p))

        I_L1p_about_shoulder = (1/3) * m1 * L1_p**2
        I_L2p_about_its_CoM = (1/12) * m2 * L2_p**2
        d_CoM_L2p_from_shoulder = L1_p + L2_p / 2
        I_L2p_about_shoulder_via_PAT = I_L2p_about_its_CoM + m2 * d_CoM_L2p_from_shoulder**2
        current_I_j1 = I_L1p_about_shoulder + I_L2p_about_shoulder_via_PAT

        x_hand_side_projection = L1_p + L2_p
        d_sh_side_sq = x_hand_side_projection**2 + z_s**2

        if d_sh_side_sq > (L1 + L2)**2 + 1e-9 or d_sh_side_sq < (abs(L1 - L2))**2 - 1e-9:
            return np.nan, np.nan

        cos_theta_elbow_internal_side = (L1**2 + L2**2 - d_sh_side_sq) / (2 * L1 * L2)
        if abs(cos_theta_elbow_internal_side) > 1.0 + 1e-9:
            return np.nan, np.nan
        cos_theta_elbow_internal_side = np.clip(cos_theta_elbow_internal_side, -1.0, 1.0)
        cos_q2_for_formula = -cos_theta_elbow_internal_side

        current_I_j2 = ((1/3)*m1*L1**2 + m2*(L1**2 + (1/3)*L2**2 + L1*L2*cos_q2_for_formula))

        if np.isnan(current_I_j1) or np.isnan(current_I_j2):
            return np.nan, np.nan
            
        return current_I_j1, current_I_j2

    def _calculate_moments_of_inertia_elbow_out(self, rad_j2_elbow_angle_mu, rad_j2_elbow_angle_sigma=0.0):
        """
        Calculates the moment of inertia for J1 and J2 joints for the ELBOW_OUT configuration.
        J1: Shoulder rotation in XY plane.
        J2: Elbow flexion/extension in XY plane.

        Args:
            rad_j2_elbow_angle_mu (float): Mean of the elbow angle in radians (relative to upper arm, 0 = straight).
            rad_j2_elbow_angle_sigma (float, optional): Std dev of the elbow angle. Defaults to 0.0.

        Returns:
            tuple: (I_j1_mu, I_j1_sigma, I_j2_mu, I_j2_sigma) in kg*m^2
        """
        m1 = self.true_m_upper_arm
        m2 = self.true_m_lower_arm
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm
        
        # I_j2 is constant for elbow_out configuration as it's defined as rotation of forearm about elbow.
        I_j2_mu = (1/3) * m2 * L2**2
        I_j2_sigma = 0.0 # I_j2 is constant, so no uncertainty propagated from J2 angle.

        # Determine the J2 angle and sigma to use for I_j1 calculation
        j2_angle_for_Ij1_calc = rad_j2_elbow_angle_mu
        j2_sigma_to_propagate_to_Ij1 = rad_j2_elbow_angle_sigma

        if np.isnan(j2_angle_for_Ij1_calc):
            I_j1_mu = np.nan
            I_j1_sigma = np.nan
            return I_j1_mu, I_j1_sigma, I_j2_mu, I_j2_sigma

        cos_val = np.cos(j2_angle_for_Ij1_calc)
        
        # Distance squared from J1 to CoM of lower arm 
        I_j1_mu = ( (1/3)*m1*L1**2 + 
                    m2*(L1**2 + (1/4)*L2**2 + L1*L2*cos_val) + 
                    (1/12)*m2*L2**2 )


        # Sigma Calculation for I_j1

        effective_j2_sigma_for_Ij1_prop = 0.0
        if not np.isnan(j2_sigma_to_propagate_to_Ij1) and j2_sigma_to_propagate_to_Ij1 > 1e-9:
            effective_j2_sigma_for_Ij1_prop = j2_sigma_to_propagate_to_Ij1

        if np.isclose(effective_j2_sigma_for_Ij1_prop, 0.0, rtol=1e-09, atol=1e-09): # No uncertainty to propagate
            I_j1_sigma = 0.0
        else:
            # Derivative of I_j1 w.r.t j2_angle_for_Ij1_calc
            # from m2*L1*L2*cos(j2_angle_for_Ij1_calc) part of I_j1
            B_coeff = m2 * L1 * L2 
            derivative_I_j1_wrt_j2 = -B_coeff * np.sin(j2_angle_for_Ij1_calc)
            I_j1_sigma = np.abs(derivative_I_j1_wrt_j2) * effective_j2_sigma_for_Ij1_prop
        
        # If I_j1_mu is NaN (due to j2_angle_for_Ij1_calc being NaN), I_j1_sigma should also be NaN.
        if np.isnan(I_j1_mu):
             I_j1_sigma = np.nan

        return I_j1_mu, I_j1_sigma, I_j2_mu, I_j2_sigma

    def update_joint_kinematics(self, rad_j1_k, rad_j2_k, omega_j1_k, omega_j2_k, torque_j1_efferent, torque_j2_efferent, true_physics=False):
        """
        Updates the joint kinematics based on the current state and motor commands.
        """
        return self._update_joint_kinematics_full(
            rad_j1_k, rad_j2_k, omega_j1_k, omega_j2_k, torque_j1_efferent, torque_j2_efferent, true_physics=true_physics
        )

    def _calculate_mass_matrix(self, rad_j2, true_physics=False):
        """
        Calculates the 2x2 mass-inertia matrix M(q) for the elbow_out configuration.
        
        Args:
            rad_j2 (float): The current elbow angle (q2).

        Returns:
            np.ndarray: The 2x2 mass matrix M. Returns a matrix of NaNs if input is NaN.
        """
        if np.isnan(rad_j2):
            return np.full((2, 2), np.nan)
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm
        m1 = self.true_m_upper_arm
        m2 = self.true_m_lower_arm
        
        lc1 = L1 / 2.0
        lc2 = L2 / 2.0
        I1_com = (1/12.0) * m1 * L1**2
        I2_com = (1/12.0) * m2 * L2**2
        c2 = np.cos(rad_j2)

        m11 = m1*lc1**2 + m2*(L1**2 + lc2**2 + 2*L1*lc2*c2) + I1_com + I2_com
        m12 = m2*(lc2**2 + L1*lc2*c2) + I2_com
        m22 = m2*lc2**2 + I2_com
        
        M = np.array([
            [m11, m12],
            [m12, m22]
        ])
        return M

    def _calculate_coriolis_vector(self, rad_j2, omega_j1, omega_j2, true_physics=False):
        """
        Calculates the 2x1 vector of Coriolis and centrifugal torques C(q, q_dot) * q_dot.

        Args:
            rad_j2 (float): Current elbow angle (q2).
            omega_j1 (float): Current shoulder angular velocity (q_dot1).
            omega_j2 (float): Current elbow angular velocity (q_dot2).

        Returns:
            np.ndarray: The 2x1 Coriolis/centrifugal torque vector. Returns NaNs if inputs are NaN.
        """
        if np.isnan(rad_j2) or np.isnan(omega_j1) or np.isnan(omega_j2):
            return np.full(2, np.nan)
        m2 = self.true_m_lower_arm
        L1 = self.true_len_upper_arm
        lc2 = self.true_len_lower_arm / 2.0

        s2 = np.sin(rad_j2)

        h = -m2 * L1 * lc2 * s2

        c1 = h * (2 * omega_j1 * omega_j2 + omega_j2**2)
        c2 = h * (-omega_j1**2)

        return np.array([c1, c2])

    def _calculate_gravity_vector(self, rad_j1, rad_j2, true_physics=False):
        """
        Calculates the 2x1 vector of gravitational torques G(q).
        """
        g1, _ = self.calculate_torque_due_to_gravity(0, rad_j1, rad_j2, true_physics=true_physics)
        g2, _ = self.calculate_torque_due_to_gravity(1, rad_j1, rad_j2, true_physics=true_physics)
        return np.array([g1 if not np.isnan(g1) else 0.0, 
                         g2 if not np.isnan(g2) else 0.0])

    def _update_joint_kinematics_full(self, rad_j1_k, rad_j2_k, omega_j1_k, omega_j2_k, torque_j1_efferent, torque_j2_efferent, true_physics = False):
        """
        Updates joint kinematics using the full, coupled dynamic model for the elbow_out configuration.
        Solves M(q)q_ddot + C(q, q_dot)q_dot + G(q) = Tau.
        This function will always include damping.
        """
        q = np.array([rad_j1_k, rad_j2_k])
        q_dot = np.array([omega_j1_k, omega_j2_k])
        
        # 1. Get dynamic model components
        M = self._calculate_mass_matrix(q[1], true_physics=true_physics) # Mass matrix depends on rad_j2
        C_vec = self._calculate_coriolis_vector(q[1], q_dot[0], q_dot[1], true_physics=true_physics) # Coriolis vector
        G_vec = self._calculate_gravity_vector(q[0], q[1], true_physics=true_physics) # Gravity vector (zero if gravity disabled)

        if np.isnan(M).any() or np.isnan(C_vec).any():
            raise ValueError(f"NaN in dynamics matrices for state {q}, {q_dot}. M={M}, C_vec={C_vec}")

        # 2. Calculate applied and external torques
        motor_torques = np.array([torque_j1_efferent, torque_j2_efferent])
        
        damping_torques = np.array([
            -self.damping_factor_j1 * q_dot[0],
            -self.damping_factor_j2 * q_dot[1]
        ])


        # 3. Solve for angular acceleration (q_ddot)
        # M * q_ddot = motor_torques + damping_torques - C_vec - G_vec
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            raise ValueError(f"Mass matrix M is singular for state q={q}. M={M}")

        
        force_field_torques = self._get_force_field_torques(true_physics=true_physics) * true_physics
        
        # Optional hand friction in task space: F_fric = -C v_hand ; tau_fric = J^T F
        hand_friction_torques = np.array([0.0, 0.0])
        if self.apply_hand_friction:
            # Update hand velocity and Jacobian at current state
            J = self.calculate_jacobian(q[0], q[1], true_physics=true_physics)
            v_hand = J @ q_dot
            F_fric = - self.hand_friction_matrix @ v_hand
            hand_friction_torques = J.T @ F_fric

        rhs = motor_torques + damping_torques - C_vec - G_vec + force_field_torques + hand_friction_torques  # per Eq. \ref{eq:ddot}
        q_ddot = M_inv @ rhs

        next_alpha_j1, next_alpha_j2 = q_ddot

        # 4. Integrate to get next state (Euler integration)
        next_omega_j1 = omega_j1_k + next_alpha_j1 * self.dt
        next_omega_j2 = omega_j2_k + next_alpha_j2 * self.dt

        next_rad_j1 = rad_j1_k + next_omega_j1 * self.dt
        next_rad_j2 = rad_j2_k + next_omega_j2 * self.dt

        # 5. Apply Joint Limits (and stop motion if limit is hit)
        pre_clip_rad_j1 = next_rad_j1
        pre_clip_rad_j2 = next_rad_j2

        next_rad_j1 = np.clip(next_rad_j1, self.lim_j1_min, self.lim_j1_max)
        next_rad_j2 = np.clip(next_rad_j2, self.lim_j2_min, self.lim_j2_max)

        if next_rad_j1 != pre_clip_rad_j1:
            next_omega_j1 = 0.0
        if next_rad_j2 != pre_clip_rad_j2:
            next_omega_j2 = 0.0

        return np.array([next_rad_j1, next_rad_j2, next_omega_j1, next_omega_j2, next_alpha_j1, next_alpha_j2])

    def _calculate_force_field(self):
        """
        Calculates the external force to be applied to the hand based on the force field settings.
        """
        force = np.array([0.0, 0.0])
        if self.force_field_type == 'constant':
            force = self.force_field_vector * self.force_field_magnitude
        elif self.force_field_type == 'curl_cw':
            rotation_matrix = np.array([[0, 1], [-1, 0]])
            force = self.force_field_magnitude * (rotation_matrix @ self.v_hand)
        elif self.force_field_type == 'curl_ccw':
            rotation_matrix = np.array([[0, -1], [1, 0]])
            force = self.force_field_magnitude * (rotation_matrix @ self.v_hand)
        
        # Check if the force field is active
        if self.time < self.force_field_on[0] or self.time > self.force_field_on[1]:
            force = np.array([0.0, 0.0])

        return force

    def _get_force_field_torques(self, true_physics=False):
        """
        Calculates the joint torques resulting from the external force field.
        """
        if self.force_field_type == 'none':
            return np.array([0.0, 0.0])
        else:
            # 1. Calculate the external force
            force = self._calculate_force_field()
            if np.all(force == 0):
                return np.array([0.0, 0.0])

            # 2. Calculate the Jacobian
            J = self.calculate_jacobian(self.rad_j1, self.rad_j2, true_physics=true_physics)

            # 3. Calculate joint torques: tau = J^T * F
            torques = J.T @ force
            return torques

    def calculate_jacobian(self, rad_j1, rad_j2, true_physics=False):
        """Calculates the Jacobian matrix for the forward kinematics.

        Args:
            rad_j1 (float): Current j1 angle in radians.
            rad_j2 (float): Current j2 angle in radians.

        Returns:
            np.ndarray: The 2x2 Jacobian matrix [[dx/dq1, dx/dq2], [dy/dq1, dy/dq2]],
                        or raises ValueError if calculation fails.
        """
        if np.isnan(rad_j1) or np.isnan(rad_j2):
            raise ValueError(f"NaN input angles to calculate_jacobian: rad_j1={rad_j1}, rad_j2={rad_j2}")

        if self.elbow_down:
            return self._calculate_jacobian_elbow_down(rad_j1, rad_j2, true_physics=true_physics)
        else:
            return self._calculate_jacobian_elbow_out(rad_j1, rad_j2, true_physics=true_physics)

    def _calculate_jacobian_elbow_out(self, rad_j1, rad_j2, true_physics=False):
        """Calculates Jacobian for the elbow_out configuration.
        x_hand = p_shoulder_x + L1*cos(q1) + L2*cos(q1+q2)
        y_hand = p_shoulder_y + L1*sin(q1) + L2*sin(q1+q2)
        """
        L1 = self.true_len_upper_arm
        L2 = self.true_len_lower_arm

        s1 = np.sin(rad_j1)
        c1 = np.cos(rad_j1)
        s12 = np.sin(rad_j1 + rad_j2)
        c12 = np.cos(rad_j1 + rad_j2)

        J = np.zeros((2, 2))
        J[0, 0] = -L1 * s1 - L2 * s12  # dx/dq1
        J[0, 1] = -L2 * s12            # dx/dq2
        J[1, 0] = L1 * c1 + L2 * c12   # dy/dq1
        J[1, 1] = L2 * c12             # dy/dq2
        return J

    def _calculate_jacobian_elbow_down(self, rad_j1, rad_j2, h=1e-7, true_physics=False):
        """Calculates Jacobian for the elbow_down configuration using numerical differentiation (central differences).
        This is an approximation.
        Args:
            rad_j1 (float): Current j1 angle in radians.
            rad_j2 (float): Current j2 angle in radians.
            h (float): Step size for numerical differentiation.
        Returns:
            np.ndarray: The 2x2 Jacobian matrix.
        Raises:
            ValueError: If FK fails for original or perturbed states, or if Jacobian contains NaNs.
        """
        J_num = np.zeros((2, 2))

        try:
            _, p_hand_orig = self.forward_kinematics_elbow_down(rad_j1, rad_j2, true_physics=true_physics)
            if np.isnan(p_hand_orig).any():
                raise ValueError(f"FK failed (NaN) at original point in numerical Jacobian for elbow_down ({rad_j1}, {rad_j2})")
        except ValueError as e:
            raise ValueError(f"Error in FK at original point for numerical Jacobian (elbow_down, state: {rad_j1},{rad_j2}): {e}")

        # Partial derivative w.r.t. rad_j1
        try:
            _, p_hand_j1_plus = self.forward_kinematics_elbow_down(rad_j1 + h, rad_j2, true_physics=true_physics)
            _, p_hand_j1_minus = self.forward_kinematics_elbow_down(rad_j1 - h, rad_j2, true_physics=true_physics)
            if np.isnan(p_hand_j1_plus).any() or np.isnan(p_hand_j1_minus).any():
                raise ValueError(f"FK failed (NaN) for rad_j1 perturbation in numerical Jacobian elbow_down. Plus: {p_hand_j1_plus}, Minus: {p_hand_j1_minus}")
            J_num[:, 0] = (p_hand_j1_plus - p_hand_j1_minus) / (2 * h)
        except ValueError as e:
            raise ValueError(f"Error in FK for rad_j1 perturbation in numerical Jacobian (elbow_down): {e}")

        # Partial derivative w.r.t. rad_j2
        try:
            _, p_hand_j2_plus = self.forward_kinematics_elbow_down(rad_j1, rad_j2 + h, true_physics=true_physics)
            _, p_hand_j2_minus = self.forward_kinematics_elbow_down(rad_j1, rad_j2 - h, true_physics=true_physics)
            if np.isnan(p_hand_j2_plus).any() or np.isnan(p_hand_j2_minus).any():
                raise ValueError(f"FK failed (NaN) for rad_j2 perturbation in numerical Jacobian elbow_down. Plus: {p_hand_j2_plus}, Minus: {p_hand_j2_minus}")
            J_num[:, 1] = (p_hand_j2_plus - p_hand_j2_minus) / (2 * h)
        except ValueError as e:
            raise ValueError(f"Error in FK for rad_j2 perturbation in numerical Jacobian (elbow_down): {e}")
            
        if np.isnan(J_num).any():
            raise ValueError(f"Numerical Jacobian for elbow_down resulted in NaNs for state ({rad_j1},{rad_j2}). J_num: {J_num}")
        return J_num


    def get_trajectory_targets_oct_receeding_horizon(self, current_state):
        """
        Calculates the current target position and velocity based on a pre-computed optimal plan.
        If a plan does not exist, it generates one using an LQR-based optimal control solver.
        This function serves as a drop-in replacement for get_trajectory_targets.
        """
        # Uses planner in Eq. \ref{eq:J}; first action u_0^* becomes feedforward per Eq. \ref{eq:tauff}
        # Return current state and no feedforward torque if movement should not start yet
        if self.time < self.min_time_before_movement:
            self.planned_state = self.x_est_ukf
            self.torque_j1_ff, self.torque_j2_ff = 0.0, 0.0
            return

        if self.constant_remaining_time:
            remaining_time = self.planned_max_time_target
        else:
            remaining_time = self.planned_max_time_target - (self.time - self.min_time_before_movement)

        # target_state = np.array([self.rad_j1_target, self.rad_j2_target, 0.0, 0.0])
        target_state = np.array([self.rad_j1_target, self.rad_j2_target, self.omega_j1_target, self.omega_j2_target])
    
        if remaining_time >= self.dt:
            self.planned_optimal_states, self.planned_optimal_torques = self._plan_optimal_trajectory(
                initial_state = current_state,
                final_state = target_state,
                duration = remaining_time,
                dt=self.dt,
                return_all_steps=False, # Dont compute the entire forward pass, just the first step,
                forward_steps=1
                )

            self.trajectory_planned = True
            # Set the initial planned state and forward torque to 1nd index of the plan; torques control
            # where the agents wants to be at the next time step
            self.planned_state = self.planned_optimal_states[1] # next planned state
            torque_j1, torque_j2 = self.planned_optimal_torques[0]
            return torque_j1, torque_j2
        else:
            return 0.0, 0.0

    def get_trajectory_targets_oct_open_loop(self):
        """
        Calculates the current target position and velocity based on a pre-computed optimal plan.
        If a plan does not exist, it generates one using an LQR-based optimal control solver.
        This function serves as a drop-in replacement for get_trajectory_targets.
        """
        # plan the trajectory
        if not self.trajectory_planned and self.time >= self.min_time_before_movement:
            self.trajectory_start_time = self.time

            # Define initial and final states for the planner
            initial_state = np.array([self.x_est_ukf[0], self.x_est_ukf[1], self.x_est_ukf[2], self.x_est_ukf[3]])
            final_state = np.array([self.rad_j1_target, self.rad_j2_target, 0.0, 0.0])

            # Call the new planner to compute the entire optimal trajectory
            self.planned_optimal_states, self.planned_optimal_torques = self._plan_optimal_trajectory(
                initial_state=initial_state,
                final_state=final_state,
                duration=self.planned_max_time_target,
                dt=self.dt,
                return_all_steps=True
            )
            
            self.trajectory_planned = True

        elif not self.trajectory_planned and self.time < self.min_time_before_movement:
            return self.x_est_ukf[0], self.x_est_ukf[1], 0.0, 0.0

        # retrieve planned state and torques
        if self.trajectory_planned:
            time_since_start = self.time - self.trajectory_start_time
            time_since_start_idx = max(0, int(time_since_start / self.dt))

            # Check if the planned movement duration has been exceeded.
            # The torques array has a length equal to the number of steps in the plan.
            if time_since_start_idx >= len(self.planned_optimal_torques):
                # If the plan is complete, the feedforward torque is zero.
                self.torque_j1_ff, self.torque_j2_ff = 0.0, 0.0
                # The reference state for the feedback controller is the final planned state.
                self.planned_state = self.planned_optimal_states[-1]
            else:
                # If we are still within the plan, extract the current reference state and torque.
                # The state index corresponds to the start of the time step.
                self.planned_state = self.planned_optimal_states[time_since_start_idx]
                self.torque_j1_ff, self.torque_j2_ff = self.planned_optimal_torques[time_since_start_idx]

            # The function still returns kinematic targets for consistency with the old planner
            target_pos = self.planned_state[:2]
            target_vel_final = self.planned_state[2:]

        else: # (Not planned yet) or (Planning failed)
            # If no plan is active, the target is simply the current estimated state (hold position).
            target_pos = np.array([self.x_est_ukf[0], self.x_est_ukf[1]])
            target_vel_final = np.array([self.x_est_ukf[2], self.x_est_ukf[3]])
            
            # Ensure planned values are NaNs so the controller knows no plan is active
            self.planned_state = np.array([np.nan, np.nan, np.nan, np.nan])
            self.torque_j1_ff, self.torque_j2_ff = np.array([np.nan, np.nan])

        # store planned trajectory position and velocity (Cartesian for plotting/analysis)
        try:
            _, p_planned_cartesian = self.forward_kinematics(rad_j1=target_pos[0], rad_j2=target_pos[1])
            self.p_planned_trajectory = p_planned_cartesian
            
            J_target_pos = self.calculate_jacobian(target_pos[0], target_pos[1]) 
            vx_hand, vy_hand = J_target_pos @ target_vel_final
            self.v_planned_trajectory = np.array([vx_hand, vy_hand])
        except ValueError as e:
            # print(f"Warning: FK or Jacobian failed in get_trajectory_targets for planned path. {e}")
            self.p_planned_trajectory = np.array([np.nan, np.nan])
            self.v_planned_trajectory = np.array([np.nan, np.nan])

    def _dynamics_full(self, state, u):
        rad_j1_k, rad_j2_k, omega_j1_k, omega_j2_k = state[:4]
        torque_j1, torque_j2 = u
        # Only return position and vel, not torque
        return self._update_joint_kinematics_full(rad_j1_k, rad_j2_k, omega_j1_k, omega_j2_k, torque_j1, torque_j2)[:4]

    def _linearize_dynamics_analytic(self, x, u, dt=None, cancel_nonlinearity=True):
        """
        Analytic first-order linearization of the 2R planar arm around (x,u).
        State x = [q1, q2, w1, w2], control u = [tau1, tau2].
        Uses elbow_out model: M(q2) qddot + h(q2,w) + D w (+ G(q) if enabled) = tau.

        Returns discrete-time (A_d, B_d) via forward Euler: A_d = I + A_c*dt, B_d = B_c*dt.
        """

        if dt is None:
            dt = self.dt

        # Unpack state
        q1, q2, w1, w2 = x[:4]
        tau = np.array(u, dtype=float)

        # Parameters
        # Use the same "belief" parameters as planning/dynamics
        m2 = self.true_m_lower_arm
        L1 = self.true_len_upper_arm
        lc2 = self.true_len_lower_arm / 2.0

        # Damping (always included in full dynamics)
        D = np.diag([(self.damping_factor_j1), (self.damping_factor_j2)])

        # Mass matrix and its inverse
        M = self._calculate_mass_matrix(q2, true_physics=False)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # Fallback: small regularization
            M_inv = np.linalg.inv(M + 1e-8 * np.eye(2))

        # Coriolis/centrifugal torque vector h(q2, w)
        s2 = np.sin(q2); c2 = np.cos(q2)
        H = -m2 * L1 * lc2 * s2
        dH_dq2 = -m2 * L1 * lc2 * c2

        h1 = H * (2.0 * w1 * w2 + w2**2)
        h2 = H * (-w1**2)
        h = np.array([h1, h2])

        # Jacobians of h
        # dh/dq = [ [0, d/dq2 h1],
        #           [0, d/dq2 h2] ]
        dh_dq = np.zeros((2, 2))
        dh_dq[0, 1] = dH_dq2 * (2.0 * w1 * w2 + w2**2)
        dh_dq[1, 1] = dH_dq2 * (-w1**2)

        # dh/dw
        dh_dw = np.array([
            [ 2.0 * H * w2,  H * (2.0 * w1 + 2.0 * w2)],
            [-2.0 * H * w1,  0.0]
        ])

        # G = self._calculate_gravity_vector(q1, q2, true_physics=False)

        r = tau - h - D @ np.array([w1, w2])# - G
        if cancel_nonlinearity:
            # Choose nominal torques that cancel nonlinearities at (x): r=0
            r = np.zeros(2)

        # dM/dq2 and d(M^{-1})/dq2
        dm11_dq2 = -2.0 * m2 * L1 * lc2 * s2
        dm12_dq2 = -1.0 * m2 * L1 * lc2 * s2
        dm22_dq2 = 0.0
        dM_dq2 = np.array([[dm11_dq2, dm12_dq2],
                        [dm12_dq2, dm22_dq2]])
        dM_inv_dq2 = - M_inv @ dM_dq2 @ M_inv


        # A21 = d(qddot)/dq = (dM_inv/dq2)*r + M_inv * ( - dh/dq )
        A21 = np.zeros((2, 2))
        A21[:, 1] = dM_inv_dq2 @ r + M_inv @ (-dh_dq[:, 1])

        # A22 = d(qddot)/dw = M_inv * ( - dh/dw - D )
        A22 = M_inv @ ( - dh_dw - D )

        # Assemble continuous-time A_c, B_c
        A_c = np.zeros((4, 4))
        A_c[0, 2] = 1.0; A_c[1, 3] = 1.0
        A_c[2:4, 0:2] = A21
        A_c[2:4, 2:4] = A22

        B_c = np.zeros((4, 2))
        B_c[2:4, :] = M_inv

        # Discretize (forward Euler)
        A_d = np.eye(4) + A_c * dt
        B_d = B_c * dt

        return A_d, B_d


    def _ensure_lqr_regulator(self, final_state):
        q1f, q2f = final_state[0], final_state[1]
        x_star = np.array([q1f, q2f, 0.0, 0.0])
        u_star = np.array([0.0, 0.0])  # we use cancellation separately

        # Recompute only if target changed or K not set
        if getattr(self, '_reg_target_state', None) is not None:
            if np.allclose(self._reg_target_state[:2], x_star[:2]):
                return

        # Linearize at the goal and solve DARE (state-only)
        A_d, B_d = self._linearize_dynamics_analytic(
            x=x_star, u=u_star, dt=self.dt, cancel_nonlinearity=True
        )
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A_d, B_d, self.Q_lqr, self.R_lqr)
        self.K_reg = np.linalg.solve(self.R_lqr + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
        self._reg_target_state = x_star
        self._reg_last_u = np.array([0.0, 0.0], dtype=float)  # for optional RFD limiting

    def _lqr_classic_control(self, current_state, final_state):
        """
        Classical infinite-horizon discrete-time LQR control law:
        - Quadratic stage cost (x_k - x*)^T Q (x_k - x*) + u_k^T R u_k
        - Constant feedback gain from the DARE (no time-varying gains)
        - No terminal cost shaping, no augmented state, no rate-of-force smoothing
        - No fixed arrival time; convergence to final_state emerges from Q/R and dynamics

        Given the current state and desired final state, returns the instantaneous
        LQR control torque u_k = -K (x_k - x*).
        """
        n = len(current_state)  # state dimension (4)
        m = 2                   # control dimension (2)
        
        x_star = np.array(final_state, dtype=float)
        u_star = np.zeros(m, dtype=float)
        A_d, B_d = self._linearize_dynamics_analytic(
            x=x_star,
            u=u_star,
            dt=self.dt,
            cancel_nonlinearity=False
        )

        # Solve discrete-time algebraic Riccati equation for infinite-horizon LQR
        P = solve_discrete_are(A_d, B_d, self.Q_lqr, self.R_lqr)
        K = np.linalg.solve(self.R_lqr + B_d.T @ P @ B_d, B_d.T @ P @ A_d)

        # Instantaneous control for the current state
        xk = np.array(current_state, dtype=float)
        e = xk - x_star
        u = -K @ e

        return u

    def _plan_optimal_trajectory(self, initial_state, final_state, duration, dt, return_all_steps = True, forward_steps = 1):
        """
        Generates an optimal trajectory using a finite-horizon linear quadratic regulator (LQR) solver.
        """
        # Eq. \ref{eq:J}: finite-horizon quadratic cost; forward pass yields u_0^* used as feedforward (Eq. \ref{eq:tauff})
        if duration <= 0:
            return np.array([initial_state]), np.array([[0.0, 0.0]])

        num_steps = int(duration / dt)
        if num_steps <= 0:
            return np.array([initial_state]), np.array([[0.0, 0.0]])
        
        n = len(initial_state)  # State dimensions (4)
        m = 2  # Control dimensions (2)
        dynamics_func = self._dynamics_full

        # Linearize once at the initial operating point
        x_nominal = initial_state 
        u_nominal = np.array([0.0, 0.0])

        # Terminal cost on state deviation (no terminal cost on previous torque by default)

        Q_final = np.diag([
            self.Q_lqr[0,0] * self.Q_lqr_final_multiplier_position, 
            self.Q_lqr[1,1] * self.Q_lqr_final_multiplier_position, 
            self.Q_lqr[2,2] * self.Q_lqr_final_multiplier_velocity, 
            self.Q_lqr[3,3] * self.Q_lqr_final_multiplier_velocity])
        # Terminal cost on state only (no cost on u_prev at terminal)
        Q_final_aug = np.block([
            [Q_final,           np.zeros((n, m))],
            [np.zeros((m, n)),  np.zeros((m, m))]
        ])



        A, B = self._linearize_dynamics_analytic(
            x=x_nominal,
            u=u_nominal,
            dt=self.dt,
            cancel_nonlinearity=False  # use full plant linearization; no explicit cancellation in execution
        )

        A_aug = np.block([
            [A,                B],
            [np.zeros((m, n)), np.eye(m)]
        ])
        B_aug = np.vstack([B, np.eye(m)])

        # Stage costs
        Q_aug = np.block([
            [self.Q_lqr,              np.zeros((n, m))],
            [np.zeros((m, n)),        self.R_lqr]
        ])
        # Exact equivalence to u_k^T R u_k with smoothing v^T R_delta v:
        # R_eff = R + R_delta, and cross-term S between z=[x;u_prev] and v is S = [0; R]
        R_eff = self.R_lqr + self.R_lqr_delta
        S_aug = np.vstack([np.zeros((n, m)), self.R_lqr])


        # Backward Riccati sweep
        P = Q_final_aug.copy()
        K_gains = []
        for solver_step in range(num_steps):
            try:
                inv_term = np.linalg.inv(R_eff + B_aug.T @ P @ B_aug)
                K = inv_term @ (B_aug.T @ P @ A_aug + S_aug.T)
                P = Q_aug + A_aug.T @ P @ A_aug - (A_aug.T @ P @ B_aug + S_aug) @ inv_term @ (B_aug.T @ P @ A_aug + S_aug.T)
            except np.linalg.LinAlgError:
                K = np.zeros((m, n + m))
                print(f"WARNING: Augmented LQR solver failed. Using zero gains for step {solver_step}.")
            K_gains.insert(0, K)

        # Forward rollout
        z_traj = [np.hstack([initial_state, np.array([self.torque_j1_ff, self.torque_j2_ff])])]
        u_traj = []
        fs = num_steps if return_all_steps else min(forward_steps, num_steps)

        z_final = np.hstack([final_state, np.zeros(m)])
        for k in range(fs):
            z_curr = z_traj[-1]
            K = K_gains[k]
            v_k = -K @ (z_curr - z_final)  # optimal delta-u

            # Apply rate-of-force development limits directly on delta-u
            if self.limit_rfd:
                v_k[0] = np.clip(v_k[0], -self.rfd_j1_max, self.rfd_j1_max)
                v_k[1] = np.clip(v_k[1], -self.rfd_j2_max, self.rfd_j2_max)

            u_prev = z_curr[n:]
            u_k = u_prev + v_k
            u_k[0] = np.clip(u_k[0], -self.torque_j1_max, self.torque_j1_max)
            u_k[1] = np.clip(u_k[1], -self.torque_j2_max, self.torque_j2_max)

            # Propagate true dynamics for x, and update u_prev portion explicitly
            x_next = dynamics_func(z_curr[:n], u_k)
            z_next = np.hstack([x_next, u_k])
            z_traj.append(z_next)
            u_traj.append(u_k)

        x_traj = [z[:n] for z in z_traj]
        return np.array(x_traj), np.array(u_traj)

    def unpack_ukf_results_vectorized(self, results_df):
        """
        Unpacks various state vectors from the UKF results DataFrame into individual columns
        for easier analysis and plotting, using more vectorized operations.

        Args:
            results_df (pd.DataFrame): The DataFrame generated by the Agent's save_step method.

        Returns:
            pd.DataFrame: A new DataFrame with unpacked columns.
        """
        # Initialize a dictionary to hold all new columns
        data_for_df = {}
        # Time and step (direct assignment)
        cols_to_extract = ['seed', 'time', 'time_run', 'dt', 'step', 'trial', 'run', 'visual_feedback', 'proprioceptive_feedback_rad', 'proprioceptive_feedback_omega',
                           'visual_feedback_rotation', 'proprioceptive_offset_rad_j1', 'proprioceptive_offset_omega_j1',
                           'proprioceptive_offset_rad_j2', 'proprioceptive_offset_omega_j2',
                           'r_target', 'torque_j2_sigma_const', 'torque_j2_sigma_prop',
                           'damping_factor_j1', 'damping_factor_j2',
                           'torque_j1', 'torque_j2', 'torque_j1_sigma_scaled', 'torque_j2_sigma_scaled', 'P_est_cartesian_ukf',
                           'planned_max_time_target',
                           'rad_j1_target', 'rad_j2_target',
                           'rad_j1_target_intermediate_this_step', 'rad_j2_target_intermediate_this_step',
                           'omega_j1_target_intermediate_this_step', 'omega_j2_target_intermediate_this_step',
                           'torque_j1_efferent', 'torque_j2_efferent', 'r_target_out', 'r_target_home', 'rad_j1_target_radius', 'rad_j2_target_radius',
                           'torque_j1_ff', 'torque_j2_ff', 'proprioceptive_intervention_on_angle',
                           'vis_hand_j1', 'vis_hand_j2'
                           ]
        for col in cols_to_extract:
            if col in results_df.columns and results_df[col] is not None:
                data_for_df[col] = results_df[col]
            else:
                print(f"WARNING: {col} not available to save.")

        # --- Helper function to safely extract from array columns ---
        def extract_vector_column(series, idx, expected_size, default_val=np.nan):
            """
            Extracts a component from a Series of vectors (np.arrays).
            Handles cases where elements might not be arrays or have wrong size.
            """
            def safe_get(arr):
                if isinstance(arr, (list, np.ndarray)) and len(arr) > idx:
                    # Accept arrays longer than expected_size; take the requested component
                    return arr[idx]
                return default_val
            return series.apply(safe_get)

        # True arm states (Cartesian and Joint)
        if 'p_hand' in results_df.columns:
            data_for_df['true_hand_x'] = extract_vector_column(results_df['p_hand'], 0, 2)
            data_for_df['true_hand_y'] = extract_vector_column(results_df['p_hand'], 1, 2)

        if 'v_hand' in results_df.columns:
            data_for_df['true_hand_vx'] = extract_vector_column(results_df['v_hand'], 0, 2)
            data_for_df['true_hand_vy'] = extract_vector_column(results_df['v_hand'], 1, 2)
        if 'p_elbow' in results_df.columns:
            data_for_df['true_elbow_x'] = extract_vector_column(results_df['p_elbow'], 0, 2)
            data_for_df['true_elbow_y'] = extract_vector_column(results_df['p_elbow'], 1, 2)
        if 'vis_p_hand' in results_df.columns:
            data_for_df['vis_hand_x'] = extract_vector_column(results_df['vis_p_hand'], 0, 2)
            data_for_df['vis_hand_y'] = extract_vector_column(results_df['vis_p_hand'], 1, 2)
        if 'p_shoulder' in results_df.columns:
            data_for_df['true_shoulder_x'] = extract_vector_column(results_df['p_shoulder'], 0, 2)
            data_for_df['true_shoulder_y'] = extract_vector_column(results_df['p_shoulder'], 1, 2)

        for col_name in ['rad_j1', 'rad_j2', 'omega_j1', 'omega_j2', 'alpha_j1', 'alpha_j2']:
            if col_name in results_df.columns:
                data_for_df[f'true_{col_name}'] = results_df[col_name]
        # Target states
        if 'p_target' in results_df.columns:
            data_for_df['target_x'] = extract_vector_column(results_df['p_target'], 0, 2)
            data_for_df['target_y'] = extract_vector_column(results_df['p_target'], 1, 2)
        if 'p_target_home' in results_df.columns and not None:
            data_for_df['p_target_home_x'] = extract_vector_column(results_df['p_target_home'], 0, 2)
            data_for_df['p_target_home_y'] = extract_vector_column(results_df['p_target_home'], 1, 2)
        if 'p_target_out' in results_df.columns and not None:
            data_for_df['p_target_out_x'] = extract_vector_column(results_df['p_target_out'], 0, 2)
            data_for_df['p_target_out_y'] = extract_vector_column(results_df['p_target_out'], 1, 2)
        if 'p_target_final' in results_df.columns and not None:
            data_for_df['p_target_final_x'] = extract_vector_column(results_df['p_target_final'], 0, 2)
            data_for_df['p_target_final_y'] = extract_vector_column(results_df['p_target_final'], 1, 2)
        if 'circular_target_movement_r' in results_df.columns:
            data_for_df['circular_target_movement_r'] = results_df['circular_target_movement_r']
            data_for_df['circular_target_movement_c_x'] = extract_vector_column(results_df['circular_target_movement_c'], 0, 2)
            data_for_df['circular_target_movement_c_y'] = extract_vector_column(results_df['circular_target_movement_c'], 1, 2)
            data_for_df['circular_target_rads_cum'] = results_df['circular_target_rads_cum']

        if 'p_planned_trajectory' in results_df.columns:
            data_for_df['p_planned_trajectory_x'] = extract_vector_column(results_df['p_planned_trajectory'], 0, 2)
            data_for_df['p_planned_trajectory_y'] = extract_vector_column(results_df['p_planned_trajectory'], 1, 2)
        if 'v_planned_trajectory' in results_df.columns:
            data_for_df['v_planned_trajectory_x'] = extract_vector_column(results_df['v_planned_trajectory'], 0, 2)
            data_for_df['v_planned_trajectory_y'] = extract_vector_column(results_df['v_planned_trajectory'], 1, 2)

        # Create DataFrame from the dictionary, preserving original index
        out_df = pd.DataFrame(data_for_df, index=results_df.index)
        return out_df
