import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import json
import argparse

from recourse import invertible_scms, noninvertible_scms
from recourse.applications import application_scms

DEBUGGING_ITERATIVE = False
if DEBUGGING_ITERATIVE:
    print('🚨🚨🚨 DEBUGGING MODE ACTIVE 🚨🚨🚨')


## paths
savedir = Path("results") / "recourse"
script_path = Path("scripts/run_simulation.py")

if not DEBUGGING_ITERATIVE:
    parser = argparse.ArgumentParser(description="Run recourse simulation")
    parser.add_argument("--seed", type=int, required=True, help="seed")
    parser.add_argument("--N_WORKERS", type=int, required=True, help="parallelization")
    parser.add_argument("--N_REC", type=int, required=True, help="number of recourse examples")

    args = parser.parse_args()

    seed = args.seed
    N_WORKERS = args.N_WORKERS
    N_Rec = args.N_REC
else:
    seed = 30
    N_WORKERS = 1
    N_Rec = 5

## hyperparameters that I configure before running the simulation
N_SAMPLES = 1000
pop_size = 25
ngen = 25
cxpb = 0.5
mutpb = 0.5

thresh_pred = 0.5
thresh_recourse = 0.9
N_fit = 10**5

application_filter = ['credit', 'satgpa']
model_dict = {
    'credit': 'rf',
    'satgpa': 'tree',
}

## don't change the rest of the code

params = {
    'N_Rec': N_Rec,
    'N_SAMPLES': N_SAMPLES,
    'pop_size': pop_size,
    'ngen': ngen,
    'cxpb': cxpb,
    'mutpb': mutpb,
    'seed': seed,
    'model': model_dict,
    'thresh_pred': thresh_pred,
    'thresh_recourse': thresh_recourse,
    'N_fit': N_fit,
}

def get_configs():
    """
    Generate all possible configurations for the simulation that I want to compute
    """
    settings = list(application_scms.keys())
    # Filter out the applications that are not in the application_filter
    settings = [scm for scm in settings if any(app in scm for app in application_filter)]
   
    configs = [] 
    for scm_name in settings:
        for goal in ["imp", "acc"]:
            for approach in ["ind", "sub", "ce"]:
                if approach == "ce" and goal == "imp":
                    continue
                else:
                    config = {
                        'scm_name': scm_name,
                        'goal': goal,
                        'approach': approach,
                        'model': model_dict[scm_name],
                        'savedir': savedir
                    }
                    configs.append(config) # type: ignore
    return configs

configs = get_configs()
    
def run_script(config):
    savedir_script = config['savedir'] / f"cfg_seed-{seed}_N-Rec-{N_Rec}_thresh_recourse-{thresh_recourse}_model-{config['model']}_N-SAMPLES-{N_SAMPLES}_N_fit-{N_fit}_pop_size-{pop_size}_ngen-{ngen}_cxpb-{cxpb}_mutpb-{mutpb}_thresh_pred-{thresh_pred}"
    savedir_script.mkdir(parents=True, exist_ok=True)
    
    # save params to file in savedir in json format
    params_file = savedir_script / "params.json"
    with params_file.open("w") as f:
        json.dump(params, f)

    # Build a folder name that combines the desired components
    run_folder_name = f"{config['scm_name']}_{config['goal']}_{config['approach']}"
    folder = savedir_script / run_folder_name
    folder.mkdir(parents=True, exist_ok=False)
    
    if 'noninvertible' in config['scm_name']:
        pass

    # Write the config to a text file in this folder
    config_file = folder / "config.txt"
    with config_file.open("w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Prepare paths for stdout/stderr logs
    # e.g., "logs.out", "logs.err" in the same folder
    stdout_path = folder / "logs.out"
    stderr_path = folder / "logs.err"

    # Construct the command
    cmd = [
        "python", script_path,
        "--scm_name", config['scm_name'],
        "--N_REC", str(N_Rec),
        "--goal", config["goal"],
        "--approach", config["approach"],
        "--N_SAMPLES", str(N_SAMPLES),
        "--pop_size", str(pop_size),
        "--ngen", str(ngen),
        "--cxpb", str(cxpb),
        "--mutpb", str(mutpb),
        "--folder_name", str(folder),  # Pass in the folder to the script
        "--model", str(config['model']),
        "--seed", str(seed),
        "--thresh_pred", str(thresh_pred),
        "--thresh_recourse", str(thresh_recourse),
        "--N_fit", str(N_fit),
        "--target_name", "y",
        "--continuous"  # Assuming discrete_vars is always True
    ]

    try:
        # Use stdout=out_file, stderr=err_file to stream output live
        with stdout_path.open("w") as out_file, stderr_path.open("w") as err_file:
            result = subprocess.run(
                cmd,
                stdout=out_file,
                stderr=err_file,
                text=True
            )
        success = (result.returncode == 0)
        return (config, success, None if success else f"Exit code: {result.returncode}")
    except Exception as e:
        # If the subprocess command itself fails (e.g., not found),
        # log the exception and return
        return (config, False, str(e))

# Parallel execution
if not DEBUGGING_ITERATIVE:
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_script, config): config for config in configs}
        for future in as_completed(futures):
            config = futures[future]
            try:
                config_name, success, err = future.result()
                if success:
                    print(f"✅ {config_name} completed successfully.")
                else:
                    print(f"❌ {config_name} failed. Check logs. Err: {err}")
            except Exception as e:
                print(f"❌ Exception running {config}: {e}")
else:
    for config in configs:
        config_name, success, err = run_script(config)
        if success:
            print(f"✅ {config_name} completed successfully.")
        else:
            print(f"❌ {config_name} failed. Check logs.")
        if err:
            print(f"Error: {err}")
        else:
            print(f"✅ {config_name} completed successfully.")