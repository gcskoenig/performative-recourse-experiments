from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from recourse.scm import SCM
from itertools import combinations

from typing import Callable, Optional, Iterable, Literal, Any
from deap import base, creator, tools, algorithms
import random
from scipy.optimize import minimize

import warnings

import tqdm
from functools import lru_cache

## todos to make this work:
## 1. implement an intervention operator, that takes a scm and returns a new intervened upon scm
## 2. implement a function that abducts the noise terms for a given observation
## for now I will do both these things for the linear gaussian case (Exactly thanks gpt)

def get_safe_obs(obs: pd.Series | dict) -> pd.Series:
    obs_safe = obs.copy()
    if type(obs) == dict:
        obs_safe = pd.Series(obs)
    assert type(obs_safe) == pd.Series
    return obs_safe

def get_acc_imp_probs_helper(predict: Callable[[pd.DataFrame], np.ndarray], scm: SCM, 
                             obs: pd.Series | dict, int_dict: dict[str, float],
                             goal: Optional[Literal['imp', 'acc']]=None,
                             fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                             SAMPLING_SIZE: int=10**4, enforce_binary_label: bool=True) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates the acceptance and improvement probabilities of an intervention
    for a given scm, predictor, observation and intervention dictionary.
    Therefore, the scm is intervened upon with the given intervention dictionary,
    new observations are sampled and the acceptance probability is calculated
    by predicting the target variable with the given predictor.
    The improvement probability is calculated by checking the mean of the target
    variable in the new observations. If a function fn_l is provided, it is applied
    to the target variable before calculating the improvement probability. Otherwise
    the target must be binary with values 0 and 1.
    
    Parameters
    ----------
    predict : function
        A function that takes a pandas DataFrame with the feature values and returns
        the predicted target values.
    scm : SCM
        The structural causal model that is intervened upon.
    obs : pandas DataFrame
        The observed feature values.
    int_dict : dict
        A dictionary with the intervention values for the scm.
    fn_l : function, optional
        A function that is applied to the target variable before calculating the
        improvement probability. The default is None.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default is 10**4.
        
    Returns
    -------
    imp_prob : float
        The improvement probability.
    acc_prob : float
        The acceptance probability.
    """
    assert scm.obss is not None
    assert scm.obss.shape[0] == SAMPLING_SIZE
    obs_safe = get_safe_obs(obs)
        
    scm_int = SCM.do(scm, int_dict)
    sample_obs, _ = scm_int.compute()
    sample_fs = sample_obs[obs_safe.index]
    sample_target = sample_obs.drop(obs_safe.index, axis=1).to_numpy()
    if fn_l is not None:
        sample_target = fn_l(sample_target)
    else:
        if not np.isin(sample_target, [0, 1]).all() and enforce_binary_label:
            raise ValueError('Target must be binary with values 0 and 1, or fn_l must be provided')
    imp_prob, acc_prob = (None, None)
    if goal == 'imp' or goal is None:
        imp_prob = float(np.mean(sample_target))
    if goal == 'acc' or goal is None:
        try:
            preds = predict(sample_fs)
        except Exception as e:
            print(e)
            raise e
        acc_prob = float(np.mean(preds))
    return imp_prob, acc_prob
    
def subpop_int_dict(scm: SCM, int_dict: dict[str, float], obs: pd.Series) -> dict[str, float]:
    """
    Combines the intervention dictionary with the observed values to create a
    dictionary with the total intervention values. More specifically, the 
    values for nondescendants of the observed variables are held fixed to the
    values of the observed variables. Thereby, we can sample from the conditional
    distribution of the descendants of the observed variables given the nondescendatns
    and the intervention values.

    Parameters
    ----------
    scm : SCM
        The structural causal model.
    int_dict : dict
        The intervention dictionary.
    obs : pandas DataFrame
        The observed values.
    """
    ndesc = scm.get_nondescendants(list(int_dict.keys()))
    ndesc = list(set(ndesc).intersection(set(obs.index)))
    if len(ndesc) == 0:
        obs_ndesc = {}
    else:
        obs_ndesc = obs[ndesc].to_dict()
    total_int = {**obs_ndesc, **int_dict}
    return total_int

def subpop_probs(predict: Callable[[pd.DataFrame], np.ndarray], scm: SCM, 
                 obs: pd.Series | dict, int_dict: dict[str, float],
                 goal: Optional[Literal['imp', 'acc']]=None,
                 fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                 SAMPLING_SIZE: int=10**4) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates the subpopulation-based acceptance and improvement probabilities.
    Therefore, the total intervention dictionary is calculated by combining the
    intervention dictionary with the observed values. Then, the structural causal
    model is intervened upon with the total intervention dictionary and new observations
    are sampled. The acceptance probability is calculated by predicting the target
    variable with the given predictor. The improvement probability is calculated by
    checking the mean of the target variable in the new observations. If a function
    fn_l is provided, it is applied to the target variable before calculating the
    improvement probability. Otherwise the target must be binary with values 0 and 1.
    
    Parameters
    ----------
    predict : function
        A function that takes a pandas DataFrame with the feature values and returns
        the predicted target values.
    scm : SCM
        The structural causal model that is intervened upon.
    obs : pandas DataFrame
        The observed feature values.
    int_dict : dict
        A dictionary with the intervention values for the scm.
    fn_l : function, optional
        A function that is applied to the target variable before calculating the
        improvement probability. The default is None.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default is 10**4.
    """
    obs_safe = get_safe_obs(obs)
    total_int = subpop_int_dict(scm, int_dict, obs_safe)
    # when the int_dict is empty, use acceptance probability as improvement probability (since observation dist)
    if len(int_dict) == 0:
        _, acc_prob = get_acc_imp_probs_helper(predict, scm, obs_safe, total_int,
                                               goal='acc', fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE)
        return acc_prob, acc_prob
    else:
        # make sure that either acceptance focused or an intervention on a cause
        target_var = [n for n in scm.nodes if n not in obs_safe.index][0]
        imp_prob, acc_prob = get_acc_imp_probs_helper(predict, scm, obs_safe, total_int,
                                                      goal=goal, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE)
        # if the target variable is a nondescendant, then also effects are
        # nondescendants and thus the approach of conditioning by intervening does not work
        if not (target_var in scm.get_descendants(list(int_dict.keys()))):
            imp_prob = None
        return imp_prob, acc_prob

def individual_probs(predict: Callable[[pd.DataFrame], np.ndarray], scm: SCM,
                     obs: pd.Series | dict[str, float], int_dict: dict[str, float],
                     goal: Optional[Literal['imp', 'acc']]=None,
                     fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                     SAMPLING_SIZE=10**4) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates the individual-based acceptance and improvement probabilities.
    Therefore, the structural causal model is intervened upon with the intervention
    dictionary and new observations are sampled. The acceptance probability is calculated
    by predicting the target variable with the given predictor. The improvement probability
    is calculated by checking the mean of the target variable in the new observations. If
    a function fn_l is provided, it is applied to the target variable before calculating
    the improvement probability. Otherwise the target must be binary with values 0 and 1.
    
    Parameters
    ----------
    predict : function
        A function that takes a pandas DataFrame with the feature values and returns
        the predicted target values.
    scm : SCM
        The structural causal model that is intervened upon.
    obs : pandas DataFrame
        The observed feature values.
    int_dict : dict
        A dictionary with the intervention values for the scm.
    fn_l : function, optional
        A function that is applied to the target variable before calculating the
        improvement probability. The default is None.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default
        is 10**4.
    """
    obs_safe = get_safe_obs(obs)
    if not scm.abducted:
        scm_abd = scm.abduct_one_unobserved(obs_safe)
        scm_abd.sample(sample_shape=(SAMPLING_SIZE,))
        warnings.warn('SCM was not abducted before calling individual_probs. For efficiency reasons this is advised against.')
    else:
        scm_abd = scm
    return get_acc_imp_probs_helper(predict, scm_abd, obs_safe, int_dict,
                                    goal=goal, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE)

def ce_int_dict(int_dict: dict[str, float], obs: pd.Series) -> dict[str, float]:
    """
    Creates an intervention dictionary where the values for all variables
    are either set to the changed values or held fixed to the observed values.
    This is used for the counterfactual explanation approach.

    Args:   
        int_dict (dict[str, float]): intervention dictionary
        obs (pd.Series): observation

    Returns:
        dict[str, float]: intervention dictionary with fixed values
    """
    int_dict_new = int_dict.copy()
    for key in obs.index:
        if key not in int_dict:
            int_dict_new[key] = float(obs[key]) # type: ignore
    return int_dict_new
 
def ce_probs(predict: Callable[[pd.DataFrame], np.ndarray], scm: SCM, 
             obs: pd.Series | dict, int_dict: dict[str, float],
             goal: Optional[Literal['imp', 'acc']]=None,
             fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
             SAMPLING_SIZE: int=10**4) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates the subpopulation-based acceptance and improvement probabilities.
    Therefore, the total intervention dictionary is calculated by combining the
    intervention dictionary with the observed values. Then, the structural causal
    model is intervened upon with the total intervention dictionary and new observations
    are sampled. The acceptance probability is calculated by predicting the target
    variable with the given predictor. The improvement probability is calculated by
    checking the mean of the target variable in the new observations. If a function
    fn_l is provided, it is applied to the target variable before calculating the
    improvement probability. Otherwise the target must be binary with values 0 and 1.
    
    Parameters
    ----------
    predict : function
        A function that takes a pandas DataFrame with the feature values and returns
        the predicted target values.
    scm : SCM
        The structural causal model that is intervened upon.
    obs : pandas DataFrame
        The observed feature values.
    int_dict : dict
        A dictionary with the intervention values for the scm.
    fn_l : function, optional
        A function that is applied to the target variable before calculating the
        improvement probability. The default is None.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default is 10**4.
    """
    obs_safe = get_safe_obs(obs)
    total_int = ce_int_dict(int_dict, obs_safe)
    imp_prob, acc_prob = get_acc_imp_probs_helper(predict, scm, obs_safe, total_int,
                                                  goal=goal, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE)
    return imp_prob, acc_prob

# def int_cost_fn(int_dict: dict[str, float], obs: pd.Series | dict) -> float:
#     """
#     Calculates the cost of an intervention by summing the squared differences
#     between the observed values and the intervention values.
    
#     Parameters
#     ----------
#     int_dict : dict
#         The intervention dictionary.
#     obs : pandas Series
#         The observed values.
#     """
#     obs_safe = get_safe_obs(obs)
#     int_ser = pd.Series(int_dict).to_numpy()
#     delta = obs_safe[list(int_dict.keys())].to_numpy() - int_ser
#     cost = float(np.sum(delta**2))
#     return cost

def all_subsets(S: list[str]) -> Iterable[str]:
    """Return all subsets of the iterable S, excluding the empty set."""
    for r in range(1, len(S)+1):
        for c in combinations(S, r):
            yield c # type: ignore
            
def create_cost_fn(costs: dict[str, float], obs: dict[str, float] | pd.Series,
                   type: Literal['squared']='squared') -> Callable[[dict[str, float]], float]:
    """
    Creates a cost function for the recourse problem by summing the costs
    of the interventions.
    
    Parameters
    ----------
    costs : dict
        A dictionary with the costs for each variable.
    """
    obs_safe = get_safe_obs(obs)
    costs_series = pd.Series(costs)
    if type == 'squared':
        scale = lambda x : x**2
    else:
        raise ValueError('type must be one of "squared"')
    
    @lru_cache(maxsize=None)
    def cost_fn(int_dict: frozenset) -> float:
        int_series = pd.Series(dict(int_dict))
        delta = obs_safe.loc[int_series.index] - int_series
        delta_weighted = delta * costs_series.loc[int_series.index]
        cost = scale(delta_weighted).sum()
        return float(cost)

    def cost_fn_wrapper(int_dict: dict[str, float]) -> float:
        items_rounded = [(it[0], round(it[1], 2)) for it in int_dict.items()]
        return cost_fn(frozenset(items_rounded))
    
    return cost_fn_wrapper

def target_fn(x: np.ndarray, int_set: list[str], 
              predict: Callable[[pd.DataFrame], np.ndarray],
              scm: SCM, obs: pd.Series | dict,
              goal: Literal['imp', 'acc']='imp', approach: Literal['ind', 'sub']='ind',
              fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
              SAMPLING_SIZE: int=10**4,
              return_raw: bool=False) -> float | tuple[float, float]:
    """
    Calculates the target function for the recourse problem. The target function
    is either the improvement or the acceptance probability of an intervention.
    The goal of the recourse problem is to maximize the improvement probability
    while satisfying a constraint on the acceptance probability. The approach
    can be either individual-based or subpopulation-based.
    
    Parameters
    ----------
    x : array
        The intervention values.
    int_set : list
        The intervention set.
    predict : function
        A function that takes a pandas DataFrame with the feature values and returns
        the predicted target values.
    scm : SCM
        The structural causal model that is intervened upon.
        If individualized recourse is used, the scm should be abducted.
        Also, the scm should be sampled before calling this function.
    obs : pandas DataFrame
        The observed feature values.
    goal : str, optional
        The goal of the recourse problem. The default is 'imp'.
    approach : str, optional
        The approach to calculate the probabilities. The default is 'ind'.
    fn_l : function, optional
        A function that is applied to the target variable before calculating the
        improvement probability. The default is None.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default is 10**4.
    return_raw : bool, optional
        Whether to return the raw probabilities. The default is False.
    """
    if return_raw:
        goal_pass_on = None # ensure that both imp and acc prob are computed
    else:
        goal_pass_on = goal
    int_dict = dict(zip(int_set, x))
    if approach == 'ind':
        imp_prob, acc_prob = individual_probs(predict, scm, obs, int_dict,
                                              goal=goal_pass_on, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE)
    elif approach == 'sub':
        imp_prob, acc_prob = subpop_probs(predict, scm, obs, int_dict,
                                          goal=goal_pass_on, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE)
    else:
        raise ValueError('approach must be one of "ind" or "subpop"')
    if return_raw:
        assert type(imp_prob) is float and type(acc_prob) is float
        return imp_prob, acc_prob
    if goal == 'imp':
        assert type(imp_prob) is float
        return imp_prob
    elif goal == 'acc':
        assert type(acc_prob) is float
        return acc_prob
    else:
        raise ValueError('goal must be one of "imp" or "acc"')
    
def prepare_scm(scm: SCM, obs: pd.Series | dict, approach: Literal['ind', 'sub', 'ce'], SAMPLING_SIZE: int):
    """
    Prepares the structural causal model for the recourse problem by sampling
    the noise terms and abducting the scm if individualized recourse is used.
    
    Parameters
    ----------
    scm : SCM
        The structural causal model.
    obs : pandas DataFrame
        The observed feature values.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default is 10**4.
    """
    # TODO hier nochmal überlegen ob man für CE was effizienter machen kann
    obs_safe = get_safe_obs(obs)
    scm_safe = scm.copy()
    if not scm.abducted and approach == 'ind':
        scm_safe = scm_safe.abduct_one_unobserved(obs_safe)
    if not scm_safe.epss_is_sampled() or not scm_safe.epss.shape[0] == SAMPLING_SIZE: # type: ignore
        scm_safe.sample(sample_shape=(SAMPLING_SIZE,))        
    return scm_safe
    
def inner_optim_scipy(int_set: list, predict: Callable[[pd.DataFrame], np.ndarray],
                      scm: SCM, obs: dict | pd.Series, thresh_recourse: float, costs: dict[str, float],
                      goal: Literal['imp', 'acc']='imp', approach: Literal['ind', 'sub']='ind',
                      fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                      SAMPLING_SIZE: int=10**3, return_all: bool=False,
                      initial_guess: Optional[pd.Series]=None, 
                      discrete_vars: Optional[Iterable[str]] = None,
                      method: str='COBYLA') -> tuple[dict[str, float], dict[str, float]] | dict[str, float]:
    """
    Solves the recourse problem using the scipy minimize function. The goal of the
    recourse problem is to minimize the cose while achieving a constraint on the
    improvement or acceptance probability. The approach can be either individual-based
    or subpopulation-based. We recommend to combine the function with a strategy to
    find the optimal intervention set as well, e.g. via an exhaustive search or combinatorial
    optimiaztion.
    
    Parameters
    ----------
    int_set : list
        The intervention set, that is the variables thare are intervened upon.
        Restricts the search space for the intervention values to the specified variables.
        Even if the optimal intervention does not change the value of the variable, it is
        still modeled as in intervention in the SCM.
    predict : function
        A function that takes a pandas DataFrame with the feature values and returns
        the predicted target values.
    scm : SCM
        The structural causal model that is intervened upon.
    obs : pandas DataFrame
        The observed feature values.
    thresh_recourse : float
        The threshold for the acceptance or improvement probability.
    goal : str, optional
        The goal of the recourse problem. The default is 'imp'.
    approach : str, optional
        The approach to calculate the probabilities. The default is 'ind'.
    fn_l : function, optional
        A function that is applied to the target variable before calculating the
        improvement probability. The default is None.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default is 10**4.
    return_all : bool, optional
        Whether to return the full result including the performance criteria. The default
        is False.
    """
    assert discrete_vars is None
    obs_safe = get_safe_obs(obs)
    
    if initial_guess is None:
        initial_guess = obs_safe[int_set]
    else:
        initial_guess = initial_guess[int_set]
    initial_guess_np = np.array(initial_guess, dtype=np.float64)
        
    scm_abd = prepare_scm(scm, obs_safe, approach, SAMPLING_SIZE)
    
    args = (int_set, predict, scm_abd, obs)
    kwargs = {'goal': goal, 'approach': approach,
              'fn_l': fn_l, 'SAMPLING_SIZE': SAMPLING_SIZE}
    
    cost_fn_dict = create_cost_fn(costs, obs_safe)
    
    def cost_fn(x):
        cost = cost_fn_dict(dict(zip(int_set, x)))
        return cost
    
    def constraint_fn(x):
        prob = target_fn(x, *args, **kwargs, return_raw=False)
        assert type(prob) is float
        return prob - thresh_recourse
    
    constraints = [{'type': 'ineq', 'fun': constraint_fn}]
    
    res = minimize(cost_fn, initial_guess_np,
                   constraints=constraints,
                   method=method)
    
    if not res.success:
        res = minimize(cost_fn, initial_guess_np, 
                       constraints=constraints,
                       method=method,
                       options={'maxiter': 1000, 'ftol': 1e-8, 'disp': True})
        raise ValueError('Optimization did not converge')
    
    int_dict = dict(zip(int_set, res.x))
    
    if return_all:
        tpl = target_fn(res.x, *args, **kwargs, return_raw=True)
        assert type(tpl) == tuple and len(tpl) == 2
        imp_prob, acc_prob = tpl # type: ignore
        perf_crits = {'imp_prob': imp_prob, 'acc_prob': acc_prob,
                      'cost': cost_fn(res.x), 'constraint_satisfied': constraint_fn(res.x) >= 0}  # type: ignore
        return int_dict, perf_crits
    else:
        return int_dict
    
def inner_optim_deap(int_set: list, predict: Callable[[pd.DataFrame], np.ndarray],
                                  scm: SCM, obs: dict | pd.Series, thresh_recourse: float, costs: dict[str, float],
                                  goal: Literal['imp', 'acc']='imp', approach: Literal['ind', 'sub']='ind',
                                  fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                                  SAMPLING_SIZE: int=10**3, return_all: bool=False,
                                  initial_guess: Optional[pd.Series]=None,
                                  pop_size: int = 50, ngen: int = 100,
                                  cxpb: float = 0.5, mutpb: float = 0.1,
                                  bounds: Optional[dict[str, tuple[float, float]]] = None,
                                  discrete_vars: Optional[Iterable[str]] = None,
                                  penalty_weight: float = 1e4) -> tuple[dict[str, float], dict[str, float]] | dict[str, float]: # type: ignore
    """
    Solves the inner recourse optimization problem using DEAP.
    
    This implementation sets up a genetic algorithm where the fitness is defined as the intervention cost
    plus a quadratic penalty if the threshold constraint on the target probability is not met.
    
    Parameters
    ----------
    int_set : list
        The intervention set (variables to be intervened upon).
    predict : Callable
        Function to predict the target variable given a DataFrame of features.
    scm : SCM
        The structural causal model.
    obs : pd.Series or dict
        Observed feature values.
    thresh_recourse : float
        The threshold that the target probability must exceed.
    goal : Literal['imp', 'acc'], optional
        Whether to consider improvement ('imp') or acceptance ('acc') probability. Default is 'imp'.
    approach : Literal['ind', 'sub'], optional
        Whether to use individual-based or subpopulation-based probability estimation. Default is 'ind'.
    fn_l : Optional[Callable], optional
        Optional function to apply to the target variable.
    SAMPLING_SIZE : int, optional
        Number of samples for evaluating the SCM. Default is 10**3.
    return_all : bool, optional
        Whether to return performance criteria along with the best intervention.
    initial_guess : Optional[pd.Series], optional
        An initial guess for intervention values (should be indexable by int_set).
    pop_size : int, optional
        Population size. Default is 50.
    ngen : int, optional
        Number of generations. Default is 100.
    cxpb : float, optional
        Crossover probability. Default is 0.5.
    mutpb : float, optional
        Mutation probability. Default is 0.1.
    penalty_weight : float, optional
        Weight for the constraint penalty. Default is 1e4.
    
    Returns
    -------
    If return_all is False:
        tuple of (best intervention dict, best fitness value)
    If return_all is True:
        tuple of (performance criteria dict, best intervention dict)
    """
    assert discrete_vars is None
    if bounds is None:
        bounds = scm.get_bounds()
    obs_safe = get_safe_obs(obs)
    # Prepare the SCM only once
    scm_safe = prepare_scm(scm, obs_safe, approach, SAMPLING_SIZE)
    dim = len(int_set)
    
    cost_fn_dict = create_cost_fn(costs, obs_safe)
    
    # Define the cost and constraint functions.
    def cost_fn(ind):
        int_dict = dict(zip(int_set, ind))
        return cost_fn_dict(int_dict)
    
    def constraint_value(ind):
        prob = target_fn(np.array(ind), int_set, predict, scm_safe, obs_safe, 
                           goal=goal, approach=approach, fn_l=fn_l, 
                           SAMPLING_SIZE=SAMPLING_SIZE, return_raw=False)
        return prob - thresh_recourse # type: ignore

    def eval_ind(ind):
        base_cost = cost_fn(ind)
        violation = max(0.0, -constraint_value(ind))
        fitness = base_cost + penalty_weight * violation**2
        return (fitness,)

    # Set bounds for each intervention variable.
    if initial_guess is not None:
        init = np.array(initial_guess[int_set], dtype=np.float64)
    else:
        init = obs_safe[int_set].to_numpy(dtype=np.float64)
        
    lower_bounds = np.array([bounds[var][0] for var in int_set], dtype=np.float64)
    upper_bounds = np.array([bounds[var][1] for var in int_set], dtype=np.float64)

    # DEAP setup: create a minimizing fitness and individual representation.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin) # type: ignore
    
    toolbox = base.Toolbox()
    # Attribute generator: random uniform within the bounds for each gene.
    toolbox.register("attr_float", lambda lb, ub: random.uniform(lb, ub))
    # Structure initializer: individual is a list of dim floats.
    toolbox.register("individual", tools.initCycle, creator.Individual, # type: ignore
                     ([lambda lb=lb, ub=ub: toolbox.attr_float(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)] )) # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
    
    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population, ensuring the first individual is the initial guess.
    pop = toolbox.population(n=pop_size) # type: ignore
    pop[0][:] = init.tolist()
    
    # Run the genetic algorithm.
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    
    # Get the best individual.
    best_ind = tools.selBest(pop, k=1)[0]
    best_int_dict = dict(zip(int_set, best_ind))
    best_cost = cost_fn(best_ind)
    best_constraint = constraint_value(best_ind)
    best_target_prob = target_fn(np.array(best_ind), int_set, predict, scm_safe, obs_safe,
                                  goal=goal, approach=approach, fn_l=fn_l,
                                  SAMPLING_SIZE=SAMPLING_SIZE, return_raw=False)
    best_fitness = eval_ind(best_ind)[0]
    
    if return_all:
        perf_crits = {
            'cost': best_cost,
            'constraint_satisfied': best_constraint >= 0,
            'target_prob': best_target_prob,
            'fitness': best_fitness
        }
        return best_int_dict, perf_crits
    else:
        return best_int_dict

def outer_optim_bruteforce(inner_optim_fn: Callable, predict: Callable[[pd.DataFrame], np.ndarray],
                           scm: SCM, obs: dict | pd.Series, thresh_recourse: float, costs: dict[str, float],
                           goal: Literal['imp', 'acc']='imp', approach: Literal['ind', 'sub']='ind',
                           fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                           SAMPLING_SIZE: int=10**3,
                           discrete_vars: Optional[Iterable[str]] = None,
                           initial_guess: Optional[pd.Series]=None) -> tuple[dict, dict]:
    """
    Solves the combinatorial problem in the bilevel recourse optimization problem, which
    involves optimizing over sets of variables to intervene upon (the outer problem) and
    the values of the interventions (the inner problem). The outer problem is solved by
    an exhaustive search over all subsets of the observed variables. The inner problem is
    solved by the rec_fn provided as an argument.
    
    Parameters
    ----------
    rec_fn : function
        The function that solves the inner problem. It must take the intervention set,
        the predictor, the structural causal model, the observed values and the threshold
        as arguments. It must return the scipy.optimize.OptimizeResult object.
    predict : function
        A function that takes a pandas DataFrame with the feature values and returns
        the predicted target values.
    scm : SCM
        The structural causal model that is intervened upon.
    obs : pandas DataFrame
        The observed feature values.
    thresh_recourse : float
        The threshold for the acceptance or improvement probability.
    goal : str, optional
        The goal of the recourse problem. The default is 'imp'.
    approach : str, optional
        The approach to calculate the probabilities. The default is 'ind'.
    fn_l : function, optional
        A function that is applied to the target variable before calculating the
        improvement probability. The default is None.
    SAMPLING_SIZE : int, optional
        The number of samples to draw for calculating the probabilities. The default is 10**4.
    return_all : bool, optional
        Whether to return the full result including the performance criteria. The default
        is False.
    """
    obs_safe = get_safe_obs(obs)    
    scm_safe = prepare_scm(scm, obs_safe, approach, SAMPLING_SIZE)
    
    # if improvement only consider causes, otherwise all observed variables
    cands = None
    target_node = [node for node in scm.nodes if node not in obs_safe.index]
    assert len(target_node) == 1
    target_node = target_node[0]
    if goal == 'imp':
        cands = set(scm_safe.get_ascendants([target_node])).intersection(set(obs_safe.index))
        cands = list(cands)
    elif goal == 'acc':
        cands = list(obs_safe.index)
    cands = all_subsets(cands)
    args = (predict, scm_safe, obs_safe, thresh_recourse, costs)
    kwargs = {'goal': goal, 'approach': approach,
              'fn_l': fn_l, 'SAMPLING_SIZE': SAMPLING_SIZE,
              'initial_guess': initial_guess,
              'discrete_vars': discrete_vars}
    
    best_value = np.inf
    best_int_dict = None
    best_perf_crits = None
    for cand in cands:
        cand = list(cand)
        int_dict, perf_crits = inner_optim_fn(cand, *args, **kwargs, return_all=True)
        if perf_crits['constraint_satisfied'] and perf_crits['cost'] < best_value:
            best_value = perf_crits['cost']
            best_int_dict = int_dict
            best_perf_crits = perf_crits
                                
    return best_int_dict, best_perf_crits # type: ignore

def combined_optim_deap(
    candidate_vars: list[str],            # Potential variables to intervene on
    predict: Callable[[pd.DataFrame], np.ndarray],
    scm: SCM,
    obs: pd.Series | dict,
    thresh_recourse: float,
    costs: dict[str, float],
    goal: Literal['imp', 'acc'] = 'imp',
    approach: Literal['ind', 'sub', 'ce'] = 'ind',
    fn_l: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    SAMPLING_SIZE: int = 10**3,
    pop_size: int = 50,
    ngen: int = 100,
    cxpb: float = 0.5,
    mutpb: float = 0.1,
    penalty_weight_constraint: float = 1e4,
    penalty_weight_nr_int: float = 1e-2,
    sigma_mut: float = 0.1,
    indpb: float = 0.2,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
    support: Optional[dict[str, np.ndarray]] = None,
    discrete_vars: Optional[Iterable[str]] = None,
    return_all: bool = False
):
    """
    A single DEAP-based optimization that decides *which variables to intervene upon*
    and *how to set them*, using two contiguous sub-vectors for alpha and beta:

      Individual = [ alpha[0], alpha[1], ..., alpha[M-1],
                     beta[0],  beta[1],  ..., beta[M-1] ]

    where alpha[j] in [0,1] (interpreted >= 0.5 as '1') and beta[j] is real-coded.

    Parameters
    ----------
    candidate_vars : list of str
        The set of possible variables we might intervene on (size = M).
    predict : Callable
        Function for predicting the target variable from a DataFrame of features.
    scm : SCM
        The structural causal model.
    obs : pd.Series or dict
        Observed values (the individual's features).
    thresh_recourse : float
        The threshold for acceptance or improvement probability.
    goal : {'imp', 'acc'}, optional
        Whether to consider improvement (imp) or acceptance (acc).
    approach : {'ind', 'sub'}, optional
        Whether to use individual-based or subpopulation-based probability estimation.
    fn_l : Optional[Callable], optional
        Optional transformation on the target variable for the improvement probability.
    SAMPLING_SIZE : int, optional
        Number of samples for evaluating the SCM.
    pop_size : int, optional
        Size of the GA population. Default is 50.
    ngen : int, optional
        Number of generations. Default is 100.
    cxpb : float, optional
        Probability of crossover. Default is 0.5.
    mutpb : float, optional
        Probability of mutation. Default is 0.1.
    penalty_weight : float, optional
        Weight for penalizing constraint violations. Default is 1e4.
    bound_offset : float, optional
        The ± range around obs for real-coded genes. Default is 1.0.
    sigma_mut : float, optional
        Gaussian mutation sigma. Default is 0.1.
    indpb : float, optional
        Independent probability for each gene to be mutated. Default is 0.2.
    return_all : bool, optional
        Whether to return a detailed performance dictionary with the best solution.

    Returns
    -------
    best_int_dict : dict[str, float]
        The best intervention dictionary: which variables to intervene on and the new values.
    If return_all is True, also returns:
        perf_crits : dict
            Dictionary containing cost, target probability, constraint satisfaction, etc.
    """
    if approach == 'ce':
        assert goal == 'acc', "CE approach only supports acceptance probability."
    obs_safe = get_safe_obs(obs)

    if bounds is None:
        # atm this returns the min and max values for each variable as observed in scm.obss
        scm_safe = scm.copy()
        scm_safe.sample(sample_shape=(SAMPLING_SIZE*10,))
        bounds = scm_safe.get_bounds()
    
    scm_safe = prepare_scm(scm, obs_safe, approach, SAMPLING_SIZE)
    
    candidate_vars = candidate_vars.copy()
    
    # get candidates that are causes of the target variable
    if goal == 'imp':
        target_var = [node for node in scm.nodes if node not in set(obs_safe.index)]
        causes = scm.get_ascendants(target_var)
        candidate_vars = list(set(candidate_vars).intersection(set(causes)))    

    # Number of candidate variables
    M = len(candidate_vars)
    if M == 0:
        raise ValueError("No candidate variables provided.")

    # lower and upper bounds for beta (the intervention value)
    lower_bounds = np.array([bounds[var][0] for var in candidate_vars], dtype=np.float64)
    upper_bounds = np.array([bounds[var][1] for var in candidate_vars], dtype=np.float64)
    
    # round the bounds to the nearest integer if the variable is discrete
    discrete_support = {}
    if discrete_vars is not None:
        for i, var in enumerate(candidate_vars):
            if var in discrete_vars:
                lower_bounds[i] = round(lower_bounds[i])
                upper_bounds[i] = round(upper_bounds[i])
                discrete_support[var] = scm.get_support(var)
    
    cost_fnc_dict = create_cost_fn(costs, obs_safe)
    

    # ---------- Fitness Function ----------
    def fitness_fn(ind):
        """
        Interpret the individual:
          alpha[j] = ind[j],     j in [0 .. M-1]
          beta[j]  = ind[M + j], j in [0 .. M-1]
        We'll treat alpha >= 0.5 as "1" (include variable j in the intervention).
        """
        # Build the intervention dictionary
        int_dict = {}
        for j, var in enumerate(candidate_vars):
            alpha_j = ind[j]
            beta_j  = ind[M + j]
            if alpha_j >= 0.5:
                int_dict[var] = beta_j

        # cost is sum of squared differences for only intervened variables
        cost_int = cost_fnc_dict(int_dict)
            
        # additional penalty for whether intervention was performed
        nr_interventions = len(int_dict)

        # Evaluate the target probability
        # if nr_interventions == 0:
        #     # No intervention => baseline distribution
        #     prob = target_fn([], [], predict, scm_safe, obs_safe, # type: ignore
        #                      goal=goal, approach=approach, fn_l=fn_l,
        #                      SAMPLING_SIZE=SAMPLING_SIZE, return_raw=False)
        # else:
        if approach == 'ind':
            imp_prob, acc_prob = individual_probs(
                predict, scm_safe, obs_safe, int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE
            )
        elif approach == 'sub':
            imp_prob, acc_prob = subpop_probs(
                predict, scm_safe, obs_safe, int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE
            )
        elif approach == 'ce':
            imp_prob, acc_prob = ce_probs(
                predict, scm_safe, obs_safe, int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE
            )
        prob = imp_prob if goal == 'imp' else acc_prob

        # Constraint: prob >= thresh => violation = max(0, thresh - prob)
        assert type(prob) is float
        violation = max(0.0, thresh_recourse - prob)
        return (cost_int + penalty_weight_constraint * (violation**2) + penalty_weight_nr_int * nr_interventions,)

    # ---------- DEAP Setup ----------
    # We'll use new names to avoid conflicts with repeated creations in the same session
    name_fit = "FitnessMin_combined_sep"
    name_ind = "Individual_combined_sep"

    if not hasattr(creator, name_fit):
        creator.create(name_fit, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, name_ind):
        creator.create(name_ind, list, fitness=getattr(creator, name_fit))

    # 1) alpha in [0,1]
    def init_alpha():
        return random.randint(0, 1)

    # 2) beta in [lb, ub]
    def init_beta(lb, ub, discrete=False, support=None):
        init_val = random.uniform(lb, ub)
        if discrete:
            if support is None:
                init_val = round(init_val)
            else:
                init_val = np.random.choice(support, 1)[0]
        return init_val

    def init_individual():
        """
        First M genes = alpha sub-vector
        Next  M genes = beta sub-vector
        """
        ind = []
        # alpha sub-vector
        for _ in range(M):
            ind.append(init_alpha())
        # beta sub-vector
        for i in range(M):
            lb, ub = lower_bounds[i], upper_bounds[i]
            var_name = candidate_vars[i]
            discrete = var_name in (discrete_vars if discrete_vars is not None else [])
            support = None
            if discrete:
                support = discrete_support[var_name]
            ind.append(init_beta(lb, ub, discrete=discrete, support=support))
        return creator.Individual_combined_sep(ind) # type: ignore

    # Register population creation
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual) # type: ignore
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore
    # Evaluation function
    toolbox.register("evaluate", fitness_fn) # type: ignore 
    
    def custom_cxBlend(ind1, ind2, alpha=0.5):
        """
        Custom blend crossover that respects the binary nature of alpha and the discrete nature of beta.
        """
        M = len(candidate_vars)  # Number of candidate variables
        for i in range(M):
            # Handle alpha (binary)
            if random.random() < 0.5:  # Swap alpha values with 50% probability
                ind1[i], ind2[i] = ind2[i], ind1[i]

            # Handle beta (real or discrete)
            idx_beta = M + i
            if candidate_vars[i] in (discrete_vars if discrete_vars is not None else []):
                # For discrete beta, randomly choose one parent's value
                if random.random() < 0.5:
                    ind1[idx_beta], ind2[idx_beta] = ind2[idx_beta], ind1[idx_beta]
            else:
                # For continuous beta, perform blend crossover
                beta1 = ind1[idx_beta]
                beta2 = ind2[idx_beta]
                ind1[idx_beta] = (1 - alpha) * beta1 + alpha * beta2
                ind2[idx_beta] = (1 - alpha) * beta2 + alpha * beta1

        return ind1, ind2
    
    # toolbox.register("mate", tools.cxBlend, alpha=0.2) # type: ignore
    toolbox.register("mate", custom_cxBlend, alpha=0.5) # type: ignore

    # Mutation
    def custom_mutate(ind):
        # alpha part (0 .. M-1), beta part (M .. 2M-1)
        for j in range(M):
            # Possibly mutate alpha_j
            if random.random() < indpb:
                # alpha_j is in [0,1]
                ind[j] = (ind[j] + 1) % 2 # flip 
        for j in range(M):
            idx_beta = M + j
            if random.random() < indpb:
                # clamp within [lb, ub]
                lb = lower_bounds[j]
                ub = upper_bounds[j]
                ind[idx_beta] += random.gauss(0, sigma_mut)
                var_name = candidate_vars[j]
                if var_name in (discrete_vars if discrete_vars is not None else []):
                    if var_name in discrete_support:
                        ind[idx_beta] = np.random.choice(discrete_support[var_name], 1)[0]
                        # ind[idx_beta] = scm.seqs[var_name].d.sample((1,)).numpy()[0]
                    else:
                        ind[idx_beta] = round(ind[idx_beta])
                ind[idx_beta] = min(ub, max(lb, ind[idx_beta]))
        return (ind,)

    toolbox.register("mutate", custom_mutate) # type: ignore
    toolbox.register("select", tools.selTournament, tournsize=3) # type: ignore

    # Create and evolve the population
    pop = toolbox.population(n=pop_size) # type: ignore
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

    # Best solution
    best_ind = tools.selBest(pop, k=1)[0]
    best_fitness = best_ind.fitness.values[0]

    # Build final intervention dictionary from best_ind
    best_int_dict = {}
    for j, var in enumerate(candidate_vars):
        alpha_j = best_ind[j]
        beta_j  = best_ind[M + j]
        if alpha_j >= 0.5:
            best_int_dict[var] = beta_j

    # # If the intervention imposes no change, remove the key
    # # It is kind of important to keep this in because intervening with the same value can mean holding the variable fixed
    # # although the variable could be changed by another intervetion 
    # for var in list(best_int_dict.keys()):
    #     if best_int_dict[var] == obs_safe[var]:
    #         del best_int_dict[var]

    # Evaluate cost and probability
    final_cost = cost_fnc_dict(best_int_dict)
    # if len(best_int_dict) == 0:
    #     final_probs = target_fn([], [], predict, scm_safe, obs_safe, # type: ignore
    #                             goal=goal, approach=approach, fn_l=fn_l,
    #                             SAMPLING_SIZE=SAMPLING_SIZE, return_raw=True)
    # else:
    #     int_values = np.array(list(best_int_dict.values()), dtype=np.float64)
    #     final_probs = target_fn(int_values, list(best_int_dict.keys()), predict, scm_safe, obs_safe,
    #                             goal=goal, approach=approach, fn_l=fn_l,
    #                             SAMPLING_SIZE=SAMPLING_SIZE, return_raw=True)
        # if approach == 'ind':
        #     imp_prob, acc_prob = individual_probs(
        #         predict, scm_safe, obs_safe, best_int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE
        #     )
        # else:
        #     imp_prob, acc_prob = subpop_probs(
        #         predict, scm_safe, obs_safe, best_int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE, return_raw=True
        #     )        
    if approach == 'ind':
        final_probs = individual_probs(
            predict, scm_safe, obs_safe, best_int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE
        )
    elif approach == 'sub':
        final_probs = subpop_probs(
            predict, scm_safe, obs_safe, best_int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE
        )
        if goal == 'imp':
            assert final_probs[0] is not None
    elif approach == 'ce':
        final_probs = ce_probs(
            predict, scm_safe, obs_safe, best_int_dict, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE
        )

    assert type(final_probs) == tuple and len(final_probs) == 2 
    goal_prob = final_probs[0] if goal == 'imp' else final_probs[1]
    assert type(goal_prob) is float
    constraint_violation = goal_prob - thresh_recourse
    
    if return_all:
        perf_crits = {
            'cost': final_cost,
            'imp_prob': final_probs[0],
            'acc_prob': final_probs[1],
            'constraint_satisfied': constraint_violation,
            'fitness': best_fitness
        }
        return best_int_dict, perf_crits
    else:
        return best_int_dict

def get_initial_guess(data: pd.DataFrame, predict: Callable[[pd.DataFrame], np.ndarray],
                      fs: list[str], 
                      fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                      fn_yh: Optional[Callable[[np.ndarray], np.ndarray]]=None):
    data = data.copy()
    if fn_yh is None:
        fn_yh = lambda x: x
    data['y_pred'] = fn_yh(predict(data[fs]))
    if fn_l is not None:
        data['l'] = fn_l(data['y_pred'].to_numpy())
    # find individual with positive classifiction
    pos = data[data['y_pred'] >= 1]
    # if fn_l is provided, find individual with positive label
    if fn_l is not None:
        pos = pos[pos['l'] == 1]
    # sort by y_pred and take the largest value
    pos = pos.sort_values('y_pred', ascending=False)
    return pos.iloc[0][fs]

def get_recourse(optim: tuple[Callable, Callable] | Callable, obss: pd.DataFrame,
                 predict: Callable[[pd.DataFrame], np.ndarray], scm: SCM, thresh_recourse: float, costs: dict[str, float],
                 goal: Literal['imp', 'acc'], approach: Literal['ind', 'sub', 'ce'],
                 fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None,
                 SAMPLING_SIZE:int=10**3,
                 discrete_vars: Optional[Iterable[str]]=None,
                 initial_guess:Optional[pd.Series]=None,
                 catch_exceptions:bool=False,
                 **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    assert goal in ['imp', 'acc']
    assert approach in ['ind', 'sub', 'ce']
    if approach == 'ce':
        assert goal == 'acc'
    
    success = True
    recs = []
    perfs = []
    for i in tqdm.tqdm(range(obss.shape[0])):
        try:
            if type(optim) is tuple:
                outer_optim, inner_optim = optim[0], optim[1] # type: ignore
                res = outer_optim(inner_optim, predict, scm, obss.iloc[i], thresh_recourse, costs,
                                goal=goal, approach=approach, fn_l=fn_l,
                                SAMPLING_SIZE=SAMPLING_SIZE, initial_guess=initial_guess, discrete_vars=discrete_vars, **kwargs)
            elif callable(optim):
                res = optim(list(obss.columns), predict, scm, obss.iloc[i], thresh_recourse, costs,
                            goal=goal, approach=approach, fn_l=fn_l, SAMPLING_SIZE=SAMPLING_SIZE,
                            return_all=True, discrete_vars=discrete_vars, **kwargs)
            else:
                raise ValueError('optim must be either a tuple of two functions or a single function')
            recs.append(res[0])
            perfs.append(res[1])
        except Exception as e:
            if catch_exceptions:
                print(f"Exception for index {i}: {e}")
                recs.append({})
                perfs.append({})
                success = False
                continue
            else:
                raise e
    recs = pd.DataFrame(recs, index=obss.index)
    perfs = pd.DataFrame(perfs, index=obss.index)
    return recs, perfs, success

def true_pr_outcomes(scm: SCM, epss: pd.DataFrame, ints: pd.DataFrame,
                     fn_l: Optional[Callable[[np.ndarray], np.ndarray]]=None, target_var: Optional[str]=None,
                     predict: Optional[Callable[[pd.DataFrame], np.ndarray]]=None) -> pd.DataFrame:
    """
    Compute the true post-recourse outcomes given a scm, a state for eps and the respective intervention
    """
    assert epss.shape[0] == ints.shape[0]
    
    new_obss = []
    # iterate over all rows
    for ii in range(epss.shape[0]):
        # get the observed values
        # get the epsilon values
        eps = epss.iloc[ii]
        eps_tens = torch.tensor(eps.to_numpy()).reshape(1, -1)
        # get the intervention values
        int_dict = ints.iloc[ii].to_dict()
        # remove dict entries where the value is no number (e.g. a nan or none)
        int_dict = {k: v for k, v in int_dict.items() if not np.isnan(v) and v is not None}
        
        # create a new scm with the observed values and epsilon values
        scm_ = scm.copy()
        scm_.set_eps(eps_tens)
        scm_ = scm_.do(int_dict)
        new_obs, new_eps = scm_.compute()
        # append to the list
        new_obss.append(new_obs)
    
    new_obss = pd.concat(new_obss)
    new_obss.index = epss.index.copy()
    
    if predict is not None:
        assert target_var is not None
        X = new_obss.drop(columns=[target_var])
        new_obss['y_pred'] = predict(X)   
    
    if fn_l is not None:
        assert target_var is not None
        new_obss['l'] = fn_l(new_obss[target_var].to_numpy())
    
    return new_obss

