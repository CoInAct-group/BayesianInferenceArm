import os
import numpy as np
import pandas as pd
import config_ukf as c
import main_ukf as main
import time
import itertools
import gc 
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import inspect
import visualisation as vis
import winsound

# from batches import batches_cody1990_onset as b
# from batches import batches_fournerett1997 as b
# from batches import batches_kordingwolpert2004 as b

# from batches import batches_circulartask as b
# from batches import batches_seqreachingtask as b
# from batches import batches_roll1982 as b

# from batches import batches_kordingwolpert2004_vis as b
from batches import batches_circulartask_vis as b
# from batches import batches_seqreachingtask_vis as b


winsound.MessageBeep()  

# Function to run a single configuration in a separate process
def run_single_config(combo_info_tuple):
    """Process a single configuration combination"""
    # Unpack combo, base_config, repetition number (1-indexed), and unique seed
    combo, base_config, rep_num, run_seed = combo_info_tuple
    
    # Create run name
    # Filter combo items to include only those that vary across runs (more than one value in param_grid)
    filtered_combo_items = {
        k: v for k, v in combo.items()
        if k in b.param_grid and len(b.param_grid[k]) > 1
    }
    
    def _format_value_for_name(value):
        # Produce a compact, deterministic string for any value (incl. numpy arrays)
        # so that filenames do not depend on numpy's alignment/padding in str().
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            inner = "_".join(_format_value_for_name(x) for x in value)
            return inner
        s = str(value)
        # Sanitize characters that are invalid/undesirable in filenames.
        s = s.replace(".", "p")
        s = s.replace(" ", "")
        for ch in ("[", "]", "(", ")", "\\", "/", ",", "\n", "\r", "\t", "'", '"'):
            s = s.replace(ch, "")
        return s

    def _format_kv(k, v):
        return f"{k}_{_format_value_for_name(v)}"

    base_run_name_part = ""
    if not filtered_combo_items: # If no params vary (or param_grid was defined with single values for all)
        if b.param_grid: # Try to use the first key from param_grid if it exists
            first_param_key = list(b.param_grid.keys())[0]
            # Get the value for this key from the current combo
            combo_value = combo.get(first_param_key, 'default_value')
            base_run_name_part = _format_kv(first_param_key, combo_value)
        else: # Fallback if param_grid is empty
            base_run_name_part = "default_run"
    else: # If there are varying parameters
        base_run_name_part = "_".join(_format_kv(k, v) for k, v in filtered_combo_items.items())
    
    # Append repetition number (rep_num is 1-indexed)
    run_name = f"{base_run_name_part}_rep_{rep_num}"
    
    # Combine with base config
    full_config = {**base_config, **combo}
    
    # Run the simulation
    start_time = time.time()
    
    # Store original config values
    original_values = {}
    for key in full_config:
        if hasattr(c, key):
            original_values[key] = getattr(c, key)
    
    # Add the unique seed to the config
    if hasattr(c, 'simulation_seed'):
        original_values['simulation_seed'] = getattr(c, 'simulation_seed', None)
    setattr(c, 'simulation_seed', run_seed)

    # Apply modifications
    for key, value in full_config.items():
        setattr(c, key, value)

    # Run simulation
    results = main.main()
    
    # Save results
    results_dir = f"outputs/batches/{b.batch_name}"
    os.makedirs(results_dir, exist_ok=True)
    results["run_name"] = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results["timestamp"] = timestamp
    run_time = time.time() - start_time


    # Add all parameter values for this run to the results dataframe for saving.
    # IMPORTANT: skip keys that are already columns in `results`, because those
    # columns are populated per-step by the simulation (e.g. `visual_feedback`).
    # Overwriting them with the scalar parameter value would destroy per-step data.
    for key, value in combo.items():
        if key in results.columns:
            continue
        if isinstance(value, np.ndarray):
            results[key] = str(value.tolist())
        else:
            results[key] = value

    # Keys for params that have multiple values in the grid (i.e., they vary)
    varying_param_keys = list(filtered_combo_items.keys())
    
    # Keys for params that have only one value in the grid (i.e., they don't vary)
    non_varying_param_keys = {
        k for k, v in b.param_grid.items() if len(v) <= 1
    }

    # Start with base columns to save, but remove any non-varying params from it
    # save_cols_filtered = [
    #     col for col in b.save_cols if col not in non_varying_param_keys
    # ]

    # Add the varying parameters to the list of columns to save, ensuring no duplicates
    if hasattr(b, 'save_all_cols') and b.save_all_cols:
        save_results_cols = list(results.columns)
    else:
        save_results_cols = list(dict.fromkeys(b.save_cols + varying_param_keys))
    # print(varying_param_keys)
    results_filename = f"{results_dir}/results_batchrun_{run_name}.tsv"
    if b.save_results:
        # Create a list of columns that are actually present in the results DataFrame
        cols_to_save = [col for col in save_results_cols if col in results.columns]
        missing_cols = [col for col in save_results_cols if col not in results.columns]
        # Optionally, warn about missing columns
        # missing_cols = set(save_results_cols) - set(results.columns)
        if missing_cols:
            print(f"Warning: Columns {list(missing_cols)} not found in results for run {run_name}.")
            
        if cols_to_save:
            results_reduced = results[cols_to_save]
        else:
            # If no columns are specified for saving, save all results
            results_reduced = results
        # print(f'saving cols: {results_reduced.columns}')
            
        # Use timestamp and process ID to ensure unique filenames
        pid = os.getpid()
        results_reduced.to_csv(path_or_buf = results_filename, sep='\t', index=False)
    
    # Run plot functions if defined
    if hasattr(b, 'plot_functions') and b.plot_functions:
        plots_dir = f"{results_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
            
        # Format config info for plot text
        config_text_lines = []
        if hasattr(b, 'plot_extra_text'):
            for key in varying_param_keys:
                if key in full_config:
                    value = full_config[key]
                    if isinstance(value, float):
                        v_str = f"{value:.4f}"
                    elif isinstance(value, np.ndarray):
                        v_str = str(value.tolist())
                    else:
                        v_str = str(value)
                    config_text_lines.append(f"{key}: {v_str}")
        else: # Fallback to original behavior
            for k, v in filtered_combo_items.items():
                if isinstance(v, np.ndarray):
                    v_str = str(v.tolist())
                else:
                    v_str = str(v)
                config_text_lines.append(f"{k}: {v_str}")
        extra_text = "\n".join(config_text_lines)
            
        for plot_func in b.plot_functions:
            try:
                # Create a unique filename for each plot function
                plot_name = plot_func.__name__
                # Create subfolder per plotting function
                func_subdir = f"{plots_dir}/{plot_name}"
                os.makedirs(func_subdir, exist_ok=True)
                plot_filename = f"{func_subdir}/{run_name}_{plot_name}"
                
                # Call the plot function with the results and config info
                plot_func(results, file_type=b.plot_file_type, output_filename=plot_filename, extra_text=extra_text)
                # print(f"Generated plot: {plot_filename}")
            except Exception as e:
                print(f"Error generating plot {plot_func.__name__}: {str(e)}")

    # Return only metadata, not the full results
    metadata = {
        "run_name": run_name,
        "results_file": results_filename,
        "run_time": run_time,
        "config": {k: v for k, v in full_config.items() if not isinstance(v, np.ndarray)},
        "seed_used": run_seed,  # Optionally log the seed used for this run
        "plots_generated": [f.__name__ for f in b.plot_functions] if hasattr(b, 'plot_functions') and b.plot_functions else []
    }

    # Restore original config values
    for key, value in original_values.items():
        if value is not None: # Only restore if there was an original value
            setattr(c, key, value)
        elif key == 'simulation_seed': # If simulation_seed was added and had no prior original_value (was None or not present)
            if hasattr(c, key): # And it wasn't part of the explicit combo/base_config
                 if key not in combo and key not in base_config:
                    delattr(c, key) # Clean it up

    # Explicitly delete large data objects to free memory
    del results
    gc.collect()  # Force garbage collection
    
    return metadata

def create_parameter_grid(param_grid):
    """
    Create all combinations of parameters from a parameter grid
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    combinations = list(itertools.product(*param_values))
    result = []
    
    for combo in combinations:
        config_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
        result.append(config_dict)
    
    return result

def save_config_files(batch_name):
    """Save copies of config.py and batches.py to the batch output folder"""
    output_dir = f"outputs/batches/{batch_name}"
    os.makedirs(output_dir, exist_ok=True)
       
    # Copy the imported config_ukf (c) and batches (b) files
    config_src = c.__file__
    config_dest = f"{output_dir}/{batch_name}_config.py"
    shutil.copyfile(config_src, config_dest)

    batches_src = b.__file__
    batches_dest = f"{output_dir}/{batch_name}_batches.py"
    shutil.copyfile(batches_src, batches_dest)
    
    return config_dest, batches_dest

    
def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif callable(obj):  # Handle function objects
        return obj.__name__  # Return the function name as a string
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def save_config_as_json(batch_name, config_module, save_name="config"):
    """
    Save an imported configuration module as a JSON file.
    Only saves parameters defined directly in the provided module.
    
    Args:
        batch_name (str): Name of the batch/run for organizing outputs
        config_module: The imported configuration module
        save_name (str): Base name for the saved config file
    
    Returns:
        str: Path to the saved JSON file
    """
    # Create the output directory
    output_dir = f"outputs/batches/{batch_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to the output file
    output_file_path = f"{output_dir}/{batch_name}_{save_name}.json"
    
    # Get variables defined directly in the module
    config_dict = {}
    
    # Get the actual module's global variables
    module_globals = config_module.__dict__
    
    for name, value in module_globals.items():
        # Skip imported modules, private variables, functions, and built-ins
        if name.startswith('_') or callable(value) or inspect.ismodule(value):
            continue
        
        try:
            # Handle all NumPy types recursively
            serializable_value = convert_numpy_types(value)
            config_dict[name] = serializable_value
        except Exception as e:
            # Skip values that aren't JSON serializable
            print(f"Skipping non-serializable value for {name}: {str(e)}")
    
    # Save to JSON
    with open(output_file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # print(f"Configuration saved to {output_file_path}")
    return output_file_path


def combination_is_allowed_by_pairs(combo, allowed_pairs):
    """Return True if combo matches any dict in allowed_pairs.

    Each item in allowed_pairs is a dict of {param_name: value} that must all
    match in the combination for it to be considered allowed.
    """
    if not isinstance(allowed_pairs, (list, tuple)):
        return True
    for allowed in allowed_pairs:
        try:
            if all(combo.get(k) == v for k, v in allowed.items()):
                return True
        except AttributeError:
            # If an entry is not a dict, ignore it
            continue
    return False

if __name__ == "__main__":
    total_run_start = time.time()
    base_seed = int(total_run_start * 1000) # Use milliseconds for base_seed
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base seed for this batch: {base_seed}")
    
    # Define parameter grid for combinations to run
    param_grid = b.param_grid
    
    # Generate all parameter combinations
    all_combinations = create_parameter_grid(param_grid)
    
    # If allowed_pairs is defined, rebuild combinations so that allowed values are included
    # even if they are not present in the main grid. We take the product over all other params
    # (excluding keys mentioned in any allowed dict) and then merge with each allowed dict.
    if hasattr(b, "allowed_pairs") and b.allowed_pairs:
        # Keys that will be overridden by allowed pairs
        keys_overridden = set()
        for ap in b.allowed_pairs:
            if isinstance(ap, dict):
                keys_overridden.update(ap.keys())
        # Base grid excludes overridden keys
        base_param_grid = {k: v for k, v in param_grid.items() if k not in keys_overridden}
        base_combos = create_parameter_grid(base_param_grid)
        # If base_combos is empty (no base params), still produce combos from allowed pairs
        if not base_combos:
            base_combos = [{}]
        rebuilt = []
        for base in base_combos:
            for ap in b.allowed_pairs:
                if not isinstance(ap, dict):
                    continue
                merged = base.copy()
                merged.update(ap)
                rebuilt.append(merged)
        print(f"Rebuilt combinations from allowed_pairs over base grid: {len(base_combos)} x {len(b.allowed_pairs)} -> {len(rebuilt)}")
        all_combinations = rebuilt
    total_reps = b.reps_resample * b.reps_identical
    total_effective_runs = len(all_combinations) * total_reps
    
    # Add base configuration that applies to all runs
    base_config = {
        "batch_run": True,
    }
    
    # Determine number of workers (leave one core free for system operations)
    max_workers_avail = max(1, os.cpu_count() - 1)
    # Max workers should be based on total effective runs
    max_workers = min(max_workers_avail, total_effective_runs if total_effective_runs > 0 else 1)

    print(f"Generated {len(all_combinations)} parameter combinations, with {total_reps} reps each, for a total of {total_effective_runs} runs.")
    print(f"Using {b.reps_resample} resampled repetitions and {b.reps_identical} identical repetitions per parameter combination.")
    print(f"Max workers available: {max_workers_avail}, using up to {max_workers} parallel workers.")
    
    # Prepare input data for each worker, including repetition number and unique seed
    worker_inputs = []
    run_counter = 0
    for combo_spec in all_combinations:
        for resample_idx in range(b.reps_resample):
            # Seed for this resample iteration
            resample_seed = (base_seed + run_counter) % (2**32)
            np.random.seed(resample_seed)
            
            processed_combo = {}
            for key, value in combo_spec.items():
                if isinstance(value, (list, tuple)) and len(value) == 3 and value[0] == "normal":
                    mu, sigma = value[1], value[2]
                    processed_combo[key] = np.random.normal(loc=mu, scale=sigma)
                else:
                    processed_combo[key] = value
            
            for identical_idx in range(b.reps_identical):
                # Unique seed for each individual run
                current_run_seed = (base_seed + run_counter + identical_idx) % (2**32)
                rep_num = resample_idx * b.reps_identical + identical_idx + 1
                worker_inputs.append((processed_combo.copy(), base_config, rep_num, current_run_seed))
            
            # Increment run_counter after all identical reps for one resample are done
            run_counter += b.reps_identical
    
    # Run configurations in parallel with progress bar
    metadata_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(run_single_config, input_data) 
                  for input_data in worker_inputs]
        
        # Process results as they complete with progress bar
        for future in tqdm(as_completed(futures), 
                          total=len(futures), 
                          desc="Processing configurations"):
            metadata = future.result()
            metadata_results.append(metadata)
    
    # Calculate and display summary information
    total_run_time = time.time() - total_run_start
    
    print(f"\nAll simulations completed in {total_run_time:.2f} seconds")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save copies of config and batches files
    config_backup, batches_backup = save_config_files(b.batch_name)
    config_backup = save_config_as_json(b.batch_name, c, "config")
    batches_backup = save_config_as_json(b.batch_name, b, "batches")
    
   
    print(f"Configuration backup saved to: {config_backup}")
    print(f"Batches backup saved to: {batches_backup}")
    
    # Save intervention cols for easy plotting
    save_intervention_cols = pd.DataFrame(data = {"intervention": b.manipulated_vars})
    save_intervention_cols.to_csv(f"outputs/batches/{b.batch_name}/" + "intervention_cols.csv", index = None)

    # Save a small metadata summary file with run information
    # summary_file = f"outputs/batches/{b.batch_name}/{b.batch_name}_summary.csv"
    # summary_df = pd.DataFrame(metadata_results)
    # summary_df.to_csv(summary_file, index=False)
    # print(f"Summary metadata saved to {summary_file}")
    
    # Report average run time
    avg_time = sum(item["run_time"] for item in metadata_results if "run_time" in item) / len(metadata_results) if len(metadata_results) > 0 else 0
    print(f"Average run time per configuration: {avg_time:.2f} seconds")
    print(f"Total configurations: {len(metadata_results)}")
    winsound.Beep(1000, 200)
