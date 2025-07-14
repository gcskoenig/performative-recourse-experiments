import argparse
import pandas as pd
import numpy as np
import torch.distributions as dist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from recourse import invertible_scms, noninvertible_scms, get_rec_setup
from recourse.applications import application_scms
from recourse import get_recourse, combined_optim_deap, true_pr_outcomes
import os
import datetime
import random
import torch
import cloudpickle as pickle
from recourse import SCM

CATCH_EXCEPTIONS = True
TRY_LOCAL = False

if __name__ == "__main__":
    # I want to use argparse to get the following arguments:
    # - scm_name: the name of the SCM to use
    # - N_REC: the number of recourse examples to generate
    # - approach: the approach to use for recourse
    # - goal: the goal to use for recourse
    # - N_SAMPLES: the number of samples to draw to estimate acc and imp probs
    # - pop_size: the population size for the DEAP algorithm
    # - ngen: the number of generations for the DEAP algorithm
    # - cxpb: the crossover probability for the DEAP algorithm
    # - mutpb: the mutation probability for the DEAP algorithm
    
    # based on the above, I want to load the appropriate SCM and run the recourse algorithm
    
    if not TRY_LOCAL:
        parser = argparse.ArgumentParser(description="Run recourse simulation")
        parser.add_argument("--scm_name", type=str, required=True, help="The name of the SCM to use")
        parser.add_argument("--N_REC", type=int, required=True, help="The number of recourse examples to generate")
        parser.add_argument("--goal", type=str, required=True, help="The goal to use for recourse")
        parser.add_argument("--approach", type=str, required=True, help="The approach to use for recourse")
        parser.add_argument("--N_SAMPLES", type=int, required=True, help="The number of samples to draw to estimate acc and imp probs")
        parser.add_argument("--pop_size", type=int, required=True, help="The population size for the DEAP algorithm")
        parser.add_argument("--ngen", type=int, required=True, help="The number of generations for the DEAP algorithm")
        parser.add_argument("--cxpb", type=float, required=True, help="The crossover probability for the DEAP algorithm")
        parser.add_argument("--mutpb", type=float, required=True, help="The mutation probability for the DEAP algorithm")
        parser.add_argument("--folder_name", type=str, required=False, help="folder name to save the results")
        parser.add_argument("--model", type=str, required=False, help="model to use for recourse")
        parser.add_argument("--seed", type=int, required=True, help="seed for the random number generator")
        parser.add_argument("--thresh_pred", type=float, required=False, default=0.5, help="threshold for the prediction")
        parser.add_argument("--thresh_recourse", type=float, required=False, default=0.5, help="threshold for the recourse")
        parser.add_argument("--N_fit", type=int, required=False, default=10**4, help="number of samples to fit the model")
        parser.add_argument("--target_name", type=str, required=False, default='y', help="target name")
        parser.add_argument('--continuous', action='store_true')
        args = parser.parse_args()
        
        seed = args.seed
        scm_name = args.scm_name
        folder_name = args.folder_name
        N_REC = args.N_REC
        approach = args.approach
        goal = args.goal
        N_SAMPLES = args.N_SAMPLES
        pop_size = args.pop_size
        ngen = args.ngen
        cxpb = args.cxpb
        mutpb = args.mutpb
        model = args.model
        thresh_pred = args.thresh_pred
        thresh_recourse = args.thresh_recourse
        N_fit = args.N_fit
        target_name = args.target_name
        discrete_vars = not args.continuous
    else:
        print("🚨🚨🚨 DEBUGGING MODE ACTIVE 🚨🚨🚨")
        seed = 22
        scm_name = 'credit'
        folder_name = None
        N_REC = 10
        approach = 'ind'
        goal = 'acc'
        N_SAMPLES = 1000
        pop_size = 25
        ngen = 25
        cxpb = 0.5
        mutpb = 0.5
        model = 'tree'
        thresh_pred = 0.5
        thresh_recourse = 0.9
        N_fit = 10**4
        target_name = 'y'
        discrete_vars = False
        
    print('Setting seed')
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if model == 'rf':
        model = RandomForestClassifier(random_state=seed)
    elif model == 'logreg':
        model = LogisticRegression(random_state=seed)
    elif model == 'tree':
        model = DecisionTreeClassifier(random_state=seed)
    else:
        raise ValueError(f"Model {model} not found")
        
    print('clone model')
    model_clone = clone(model) 

   
    # Load the appropriate SCM based on the scm_name argument
    if scm_name in invertible_scms:
        scm = invertible_scms[scm_name]
    elif scm_name in noninvertible_scms:
        scm = noninvertible_scms[scm_name]
    elif scm_name in application_scms:
        scm = application_scms[scm_name]
    else:
        raise ValueError(f"SCM {scm_name} not found")
    
    if folder_name is None:
        now = datetime.datetime.now()
        folder_name = f"results/recourse/{scm_name}_{approach}_{goal}"
    else:
        folder_name = folder_name

    print('Make dirs')
    # make sure that folder exists
    os.makedirs(folder_name, exist_ok=True)

     
    model_kwargs = {}
    args_rec_setup = (scm, target_name, N_REC, N_fit, thresh_pred, model, model_kwargs)
    scm, target_name, thresh_label, fn_l, cost_dict, predict, X_rej, (obss_rej, epss_rej) = get_rec_setup(*args_rec_setup)
    if discrete_vars:
        discrete_vars = scm.get_nodes_ordered()
    else:
        discrete_vars = []
        
    pars_to_save = {
        'scm_name': scm_name,
        'N_REC': N_REC,
        'approach': approach,
        'goal': goal,
        'N_SAMPLES': N_SAMPLES,
        'target_name': target_name,
        'thresh_label': float(thresh_label),
    }
    
    rec_kwargs = {
        'pop_size': pop_size,
        'ngen': ngen,
        'cxpb': cxpb,
        'mutpb': mutpb,
        'penalty_weight_constraint': 1e4,
        'penalty_weight_nr_int': 1e-2,
        'sigma_mut': 0.1,
        'indpb': 0.5
    }
    
    print('Starting recourse search.')
    recs, perfs, success = get_recourse(
        combined_optim_deap,
        X_rej,
        predict, # type: ignore
        scm,
        thresh_recourse,
        cost_dict,
        goal,
        approach,
        SAMPLING_SIZE=N_SAMPLES,
        discrete_vars=discrete_vars,
        catch_exceptions=CATCH_EXCEPTIONS,
        fn_l = fn_l,
        **rec_kwargs
    )
    
    print('Computing post-recourse outcomes')
    # get the new observations
    new_obss = true_pr_outcomes(scm, epss_rej, recs, 
                                fn_l=fn_l, target_var=target_name, predict=predict) # type: ignore
   
   
    print('saving all the stuff')
    # save the cloned model to file
    with open(f"{folder_name}/model_clone.pkl", "wb") as f:
        pickle.dump(model_clone, f)
        
    # save recs and perfs to a CSV file
    recs_df = pd.DataFrame(recs)
    perfs_df = pd.DataFrame(perfs)
    recs_df.to_csv(f"{folder_name}/recs.csv", index=False)
    perfs_df.to_csv(f"{folder_name}/perfs.csv", index=False)
    
    # save the new observations to a CSV file
    new_obss.to_csv(f"{folder_name}/new_observations.csv", index=False)
    
    # save obss_rej and epss_rej to a CSV file
    obss_rej.to_csv(f"{folder_name}/obss_rej.csv", index=False)
    epss_rej.to_csv(f"{folder_name}/epss_rej.csv", index=False)
    
    # save all the arguments to a JSON file
    import json
    with open(f"{folder_name}/args.json", "w") as f:
        json.dump(pars_to_save, f)
   
    # save the rec_kwargs to a JSON file
    with open(f"{folder_name}/rec_kwargs.json", "w") as f:
        json.dump(rec_kwargs, f)
    
    # save the cost_dict to a JSON file
    with open(f"{folder_name}/cost_dict.json", "w") as f:
        json.dump(cost_dict, f)
        
    # save the predict function
    with open(f"{folder_name}/predict.pkl", "wb") as f:
        pickle.dump(predict, f)
    
    # save the fn_l function
    with open(f"{folder_name}/fn_l.pkl", "wb") as f:
        pickle.dump(fn_l, f)
    
    # save the scm
    SCM.save(scm, f"{folder_name}/scm.pkl")
     
    # get a large sample from the original distribution
    obss, epss = scm.sample(sample_shape=(N_fit*10,))
    X = obss.drop(columns=[target_name])
    y = obss[target_name]    
    obss['l'] = fn_l(y)
    obss['y_pred'] = predict(X)

    # save the obss and epss to a CSV file
    obss.to_csv(f"{folder_name}/obss.csv", index=False)
    epss.to_csv(f"{folder_name}/epss.csv", index=False)    

    # everything is saved, so we can exit    
    if success:
        exit(0)
    else:
        exit(1)
    
    
