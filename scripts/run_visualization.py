import pandas as pd
import numpy as np
import json
import os
import cloudpickle as pickle
from recourse import SCM
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
from sklearn.base import clone
from pandas.errors import EmptyDataError
from recourse.viz import prepare_heatmap_data, plot_heatmap, get_diff_heatmap, rename_sort_scms
from recourse.viz import plot_cond_diff, plot_accrates, METHOD_PALETTE, plot_accrates_agg, plot_cond_diff_agg, plot_accrates_diff_agg
from recourse.dgps import shortnames_scms
from recourse.applications import shortnames_applications
import random

VIZ_SUBFOLDER_NAME = 'plots'

def collect_subfolders(path: str | Path) -> list[str]:
    """
    Collect all subfolders in a given path.
    """
    path = Path(path)
    subfolders = [str(f) for f in path.iterdir() if f.is_dir()]
    return subfolders

def load_config(path: str | Path) -> dict:
    """
    Load the config file from the given path.
    """
    path = Path(path) / 'params.json'
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def load_data(loadpath: str | Path) -> dict:
    """
    Load the data
    """
    loadpath = str(loadpath) + '/'
    new_obss = pd.read_csv(loadpath + 'new_observations.csv')
    obss_rej = pd.read_csv(loadpath + 'obss_rej.csv')
    epss_rej = pd.read_csv(loadpath + 'epss_rej.csv')
    obss = pd.read_csv(loadpath + 'obss.csv')
    epss = pd.read_csv(loadpath + 'epss.csv')

    try:
        perfs = pd.read_csv(loadpath + 'perfs.csv')
        recs = pd.read_csv(loadpath + 'recs.csv')
    except EmptyDataError as e:
        perfs = pd.DataFrame([], index=new_obss.index)
        recs = pd.DataFrame([], index=obss_rej.index, columns=obss_rej.columns) 

    # load args.json
    with open(os.path.join(loadpath, 'args.json'), 'r') as f:
        args = json.load(f)
    # load cost_dict.json
    with open(os.path.join(loadpath, 'cost_dict.json'), 'r') as f:
        cost_dict = json.load(f)
    # load rec_kwargs.json
    with open(os.path.join(loadpath, 'rec_kwargs.json'), 'r') as f:
        rec_kwargs = json.load(f)
        
    try:
        scm = SCM.load(os.path.join(loadpath, 'scm.pkl'))
    except Exception as e:
        scm = None
    
    try:
        with open(os.path.join(loadpath, 'predict.pkl'), 'rb') as f:
            predict = pickle.load(f)
    except Exception as e:
        predict = None
        
    try:
        with open(os.path.join(loadpath, 'fn_l.pkl'), 'rb') as f:
            fn_l = pickle.load(f)
    except Exception as e:
        fn_l = None

    try:
        with open(os.path.join(loadpath, 'model_clone.pkl'), 'rb') as f:
            model_clone = pickle.load(f)
    except Exception as e:
        model_clone = None
    
    result_dict = {
        'recs': recs,
        'perfs': perfs,
        'new_obss': new_obss,
        'obss_rej': obss_rej,
        'epss_rej': epss_rej,
        'args': args,
        'cost_dict': cost_dict,
        'rec_kwargs': rec_kwargs,
        'scm': scm,
        'predict': predict,
        'model_clone': model_clone,
        'fn_l': fn_l,
        'obss': obss,
        'epss': epss,
    }
    
    return result_dict 
    
def heatmap_processing(folderpath):
    """
    Process the heatmap data and save it to a folder.
    """
    loadpath = Path(folderpath)
    savepath = loadpath / VIZ_SUBFOLDER_NAME
    savepath.mkdir(parents=True, exist_ok=True)
    result_dict = load_data(loadpath)

    heatmap_data_pr, count_data_pr = prepare_heatmap_data(result_dict['new_obss'], x1='x1', x2='x2', y='l')
    heatmap_savepath = savepath / 'heatmap_obss_new.pdf'
    plt.figure(figsize=(10, 8))
    plot_heatmap(heatmap_data_pr, x1='x1', x2='x2')
    plt.savefig(heatmap_savepath, bbox_inches='tight')

    heatmap_data, count_data = prepare_heatmap_data(result_dict['obss'], x1='x1', x2='x2', y='l')
    heatmap_savepath = savepath / 'heatmap_obss.pdf'
    plt.figure(figsize=(10, 8))
    plot_heatmap(heatmap_data, x1='x1', x2='x2')
    plt.savefig(heatmap_savepath, bbox_inches='tight')
        
    heatmap_diff, heatmap_weights, heatmap_stats = get_diff_heatmap(
        result_dict['new_obss'],
        result_dict['obss'],
        x1='x1',
        x2='x2',
        y='l',
    )

    heatmap_savepath = savepath / 'heatmap_diff.pdf'
    plt.figure(figsize=(10, 8))
    plot_heatmap(heatmap_diff, x1='x1', x2='x2')
    plt.savefig(heatmap_savepath, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    plot_heatmap(heatmap_weights, x1='x1', x2='x2')
    plt.savefig(savepath / 'heatmap_weights.pdf', bbox_inches='tight')
    plt.close()

    stats_savepath = savepath / 'heatmap_stats.json'
    with open(stats_savepath, 'w') as f:
        json.dump(heatmap_stats, f)
    
    # Save the heatmap data
    heatmap_data.to_csv(savepath / 'heatmap_data.csv', index=False)
    count_data.to_csv(savepath / 'count_data.csv', index=False)
    heatmap_data_pr.to_csv(savepath / 'heatmap_data_pr.csv', index=False)
    count_data_pr.to_csv(savepath / 'count_data_pr.csv', index=False)
    
    return heatmap_stats

def compile_pre_post_dfs(config_folder: str, run_folder: str) -> dict:
    """Loads all kinds of data from the folder and compiles
    a dictionary with pre and post recourse obss and epss.

    Args:
        config_folder (_type_): the folder with the config (hyperparameters for optimization etc)
        run_folder (_type_): the folder with the specific run (dgp)

    Returns:
        dict: a dictionary with the following keys
            - obss_1_pre: the first sample of pre recourse obss
            - obss_1_post: the first sample of post recourse obss
            - epss_1: the first sample of epss
            - obss_2_pre: the second sample of pre recourse obss
            - obss_2_post: the second sample of post recourse obss
            - epss_2: the second sample of epss
    """
    result_dict = load_data(run_folder)
    config = load_config(config_folder)

    seed = config['seed']
    if seed == 129 and 'binomial_linear_multiplicative_acc_sub' in run_folder:
        print("breakpoint here")
    obss = result_dict['obss']
    epss = result_dict['epss']
    predict_pre = result_dict['predict']
    fn_l = result_dict['fn_l']
    obss_rej = result_dict['obss_rej']
    obss_new = result_dict['new_obss']
    epss_rej = result_dict['epss_rej']
    
    # add feature to indicate whether recourse was performed
    obss_new['recourse'] = True
    obss['recourse'] = False
    obss_rej['recourse'] = False

    ## Now I create three samples: From the original distribution, and two samples from the post-recourse distribution
    N_rec = config['N_Rec']
    N_fit = math.floor(N_rec / 2)

    # split obss_new in half
    obss_new_1 = obss_new.sample(N_fit, random_state=seed)
    obss_new_2 = obss_new.drop(obss_new_1.index).sample(N_fit, random_state=seed)
    obss_rej_1 = obss_rej.loc[obss_new_1.index]
    obss_rej_2 = obss_rej.loc[obss_new_2.index]
    epss_rej_1 = epss_rej.loc[obss_new_1.index]
    epss_rej_2 = epss_rej.loc[obss_new_2.index]

    # match each of the sample to accepted individuals from the original distribution
    obss_acc = obss.loc[obss['y_pred'] == 1]
    obss_acc_1 = obss_acc.sample(N_fit, random_state=seed)
    obss_acc_2 = obss_acc.drop(obss_acc_1.index).sample(N_fit, random_state=seed)
    epss_acc_1 = epss.loc[obss_acc_1.index]
    epss_acc_2 = epss.loc[obss_acc_2.index]

    # combine the samples into 2x2 dataframes that each include both accepted and rejected individuals (one pre and one post recourse each)

    obss_1_pre = pd.concat([obss_acc_1, obss_rej_1])
    obss_1_post = pd.concat([obss_acc_1, obss_new_1])
    epss_1 = pd.concat([epss_acc_1, epss_rej_1])

    obss_2_pre = pd.concat([obss_acc_2, obss_rej_2])
    obss_2_post = pd.concat([obss_acc_2, obss_new_2])
    epss_2 = pd.concat([epss_acc_2, epss_rej_2])
    
    return_dict = {
        'obss_1_pre': obss_1_pre,
        'obss_1_post': obss_1_post,
        'epss_1': epss_1,
        'obss_2_pre': obss_2_pre,
        'obss_2_post': obss_2_post,
        'epss_2': epss_2,
    }
    
    return return_dict

def get_feature_names(obss, target_name):
    """
    Get the feature names from the obss dataframe.
    """
    feature_names = obss.columns.tolist()
    feature_names.remove(target_name)
    feature_names.remove('l')
    feature_names.remove('y_pred')
    feature_names.remove('recourse')
    if 'y_pred_refit' in feature_names:
        feature_names.remove('y_pred_refit')
    return feature_names           

def fit_new_predictor(obss, model_clone, target_name, thresh_pred, fn_l):
    """
    Clones the predictor and fits it again on a given dataset.
    """
    fs_names = get_feature_names(obss, target_name)
    model_new = clone(model_clone)
    model_new.fit(obss[fs_names], fn_l(obss[target_name]))
    def predict(X):
        y_pred = model_new.predict(X)
        y_pred = np.where(y_pred > thresh_pred, 1, 0)
        return y_pred
    return predict

def apply_predictors(obss: dict[str,pd.DataFrame], preds: dict, fs_names: list[str]):
    """ Apply the predictors given in a dictionary to all datasets given in a dictionary"""
    for pred_name in preds.keys():
        for obss_name in obss.keys():
            pred = preds[pred_name](obss[obss_name][fs_names])
            obss[obss_name][f'{pred_name}'] = pred
    return obss

def extract_accrates(obss_processed):
    """"Given a dataset with new prediction columns, exctract all kinds of acceptance rates for a specific dataset"""
    assert 'y_pred' in obss_processed.columns, "y_hat not in obss_processed"
    assert 'y_pred_refit' in obss_processed.columns, "y_hat_refit not in obss_processed"
    assert 'recourse' in obss_processed.columns, "recourse not in obss_processed"
    rec_ind = obss_processed['recourse'] == True
    accrate_rec_orig = float(obss_processed.loc[rec_ind, 'y_pred'].mean())
    accrate_rec_refit = float(obss_processed.loc[rec_ind, 'y_pred_refit'].mean())
    accrate_nrec_orig = float(obss_processed.loc[~rec_ind, 'y_pred'].mean())
    accrate_nrec_refit = float(obss_processed.loc[~rec_ind, 'y_pred_refit'].mean())
    accrate_overall_orig = float(obss_processed['y_pred'].mean())
    accrate_overall_refit = float(obss_processed['y_pred_refit'].mean())
    accrate_dict = {
        'rec_orig': accrate_rec_orig,
        'rec_refit': accrate_rec_refit,
        'nrec_orig': accrate_nrec_orig,
        'nrec_refit': accrate_nrec_refit,
        'overall_orig': accrate_overall_orig,
        'overall_refit': accrate_overall_refit,
    }
    return accrate_dict

def all_accrates(obss_dict):
    """ Get acceptance rates for all datasets in a dictionary"""
    accrates = {}
    for key in obss_dict.keys():
        accrates[key] = extract_accrates(obss_dict[key])
    return accrates

def get_int_summaries(result_dict: dict):
    scm = result_dict['scm']
    target = result_dict['args']['target_name']
    ## get causes from scm
    causes = scm.get_ascendants([target])
    noncauses = list(set(scm.get_nodes_ordered()) - set(causes) - set([target]))

    ## import the interventions and compile stats
    recs = result_dict['recs']
    obss_rec = result_dict['obss_rej']
    recs_diff = (recs - obss_rec).abs()
    recs_diff = recs_diff[causes + noncauses].copy()
    recs_diff_bin = (recs_diff > 0)
    recs_int_bin = (~recs_diff.isna())
    perc_diff = {}
    perc_int = {}
    perc_int['any'] = recs_int_bin.any(axis=1).mean()
    perc_int['causes'] = recs_int_bin[causes].any(axis=1).mean()
    perc_int['noncauses'] = recs_int_bin[noncauses].any(axis=1).mean()
    perc_diff['any'] = recs_diff_bin.any(axis=1).mean()
    perc_diff['causes'] = recs_diff_bin[causes].any(axis=1).mean()
    perc_diff['noncauses'] = recs_diff_bin[noncauses].any(axis=1).mean()

    ## save the results
    series_diff = pd.Series(perc_diff)
    series_int = pd.Series(perc_int)
    int_df = pd.DataFrame({'percent_absdiff>0': series_diff, 'percent_intset': series_int})

    return int_df

def result_table_row(scm_name: str, model: str, goal: str, approach: str, accrates_mean: dict, accrates_std: dict, int_df: pd.DataFrame, heatmap_stats: dict):
    res_dict = {
        'goal': goal,
        'approach': approach,
        'scm_name': scm_name,
        'model': model,
        'int_causes': int_df.loc['causes', 'percent_intset'],
        'int_any': int_df.loc['any', 'percent_intset'],
        'rec_orig': accrates_mean['rec_orig'],
        'rec_refit': accrates_mean['rec_refit'],
        'rec_orig_std': accrates_std['rec_orig'],
        'rec_refit_std': accrates_std['rec_refit'],
        'cond_diff_max': heatmap_stats['max_diff'],
        'cond_diff_min': heatmap_stats['min_diff'],
        'cond_diff_exp': heatmap_stats['expected_diff'],
    }
    # dataframe where the dict keys are the columns and there is one row
    df = pd.DataFrame.from_dict(res_dict, orient='index').T
    return df

def process_folder(configpath, runpath, nr_refits=10):
    try:
        heatmap_stats = heatmap_processing(runpath)    
    except Exception as e:
        print(f"❌ Failed to process heatmap for {runpath}. Error: {e}")
        heatmap_stats = {
            'max_diff': np.nan,
            'min_diff': np.nan,
            'expected_diff': np.nan,
        }
    result_dict = load_data(runpath)
    config = load_config(configpath)

    ## set the seed
    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)

    # create subfolder for processed data
    savepath = Path(runpath) / VIZ_SUBFOLDER_NAME
    savepath.mkdir(parents=True, exist_ok=True)


    compiled_dfs = compile_pre_post_dfs(configpath, runpath)
    results = load_data(runpath)

    post_recourse_only_shift = compiled_dfs['obss_1_post'].loc[compiled_dfs['obss_1_post']['recourse']]
    obss_train_new = pd.concat([compiled_dfs['obss_1_pre'], post_recourse_only_shift])
    
    accrates_all = []
    for ii in range(nr_refits):
        predict_new = fit_new_predictor(obss_train_new, results['model_clone'],
                                        results['args']['target_name'], config['thresh_pred'], results['fn_l'])

        preds = {
            'y_pred': results['predict'],
            'y_pred_refit': predict_new,
        }

        obsss_dict = {
            'obss_1_pre': compiled_dfs['obss_1_pre'],
            'obss_2_pre': compiled_dfs['obss_2_pre'],
            'obss_1_post': compiled_dfs['obss_1_post'],
            'obss_2_post': compiled_dfs['obss_2_post'],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                }

        fs_names = get_feature_names(compiled_dfs['obss_1_pre'], results['args']['target_name'])
        obss_dict = apply_predictors(obsss_dict, preds, fs_names)
        accrates = all_accrates(obss_dict)
   
        # save the processed data
        obss_dict['obss_1_pre'].to_csv(savepath / f'obss_1_pre_refit{ii}.csv', index=False)
        obss_dict['obss_2_pre'].to_csv(savepath / f'obss_2_pre_refit{ii}.csv', index=False)
        obss_dict['obss_1_post'].to_csv(savepath / f'obss_1_post_refit{ii}.csv', index=False)
        obss_dict['obss_2_post'].to_csv(savepath / f'obss_2_post_refit{ii}.csv', index=False)
        # save the accrates
        accrates_df = pd.DataFrame.from_dict(accrates, orient='index')
        accrates_df.to_csv(savepath / f'accrates_refit{ii}.csv') 
        accrates_all.append(accrates_df)
    
    data_array = np.stack([accdf.values for accdf in accrates_all], axis=2)
    accrates_mean = np.mean(data_array, axis=2)
    accrates_std = np.std(data_array, axis=2)
    
    accrates_mean_df = pd.DataFrame(accrates_mean, index=accrates_all[0].index, columns=accrates_all[0].columns)
    accrates_std_df = pd.DataFrame(accrates_std, index=accrates_all[0].index, columns=accrates_all[0].columns)
    accrates_mean_df.to_csv(savepath / 'accrates_mean.csv')
    accrates_std_df.to_csv(savepath / 'accrates_std.csv')

    int_df = get_int_summaries(result_dict)
    int_df.to_csv(savepath / 'int_summaries.csv')

    # row for the final result table
    df_row = result_table_row(
        results['args']['scm_name'],
        config['model'],
        results['args']['goal'],
        results['args']['approach'],
        accrates_mean_df.loc['obss_2_post'].to_dict(),
        accrates_std_df.loc['obss_2_post'].to_dict(),
        int_df,
        heatmap_stats
    )
    df_row.to_csv(savepath / 'result_table_row.csv', index=False) 
    plt.close('all')
    
def aggregate_rows_result_table(config_folder):
    config_folder = Path(config_folder)
    run_folders = collect_subfolders(config_folder)
    if len(run_folders) == 0:
        print(f"❌ No run folders found in {config_folder}.")
        return
    for run_folder in run_folders:
        run_folder = Path(run_folder)
        result_table_row_path = run_folder / VIZ_SUBFOLDER_NAME / 'result_table_row.csv'
        if not result_table_row_path.exists():
            continue
        df_row = pd.read_csv(result_table_row_path)
        if 'result_table' not in locals():
            result_table = df_row
        else:
            result_table = pd.concat([result_table, df_row], ignore_index=True)
    result_table = result_table.reset_index(drop=True)
    result_table.to_csv(config_folder / 'result_table.csv', index=False)
    
    ## export to latex
    
    # adjust column order
    column_order = ['scm_name', 'model', 'goal', 'approach', 'int_causes', 'int_any', 'rec_orig', 'rec_refit', 'cond_diff_min', 'cond_diff_max', 'cond_diff_exp']
    result_table = result_table[column_order]
    
    config_cols = ['goal', 'approach', 'scm_name', 'model']
    # sort by scm_name, goal, approach
    result_table = result_table.sort_values(by=config_cols)
    result_table = result_table.reset_index(drop=True)
   
   # replace any '_' in strings with ' '
    result_table = result_table.replace('_', ' ', regex=True)
    result_table.columns = result_table.columns.str.replace('_', ' ')
    config_cols = [col.replace('_', ' ') for col in config_cols]
    
    # export to latex
    result_table.to_latex(config_folder / 'result_table.tex', index=False, escape=False, float_format="%.2f")
 
    # don't repeat stuff
    result_table_sparse = result_table.copy()
    
    for col in config_cols:
        result_table_sparse[col] = result_table_sparse[col].where(result_table_sparse[col] != result_table_sparse[col].shift(), '')

    result_table_sparse.to_latex(config_folder / 'result_table_sparse.tex', index=False, escape=False, float_format="%.2f")
    
def visualize_result_table(config_folder):
    config_folder = Path(config_folder)
    result_table_path = config_folder / 'result_table.csv'
    if not result_table_path.exists():
        print(f"❌ No result table found in {config_folder}.")
        return
    result_table = pd.read_csv(result_table_path)
    # default aesthetics
    
    kwargs = {
        'fig_size':(4.5, 2.5),
        'font_size': 9,
        'pointsize': 5,
        'palette': METHOD_PALETTE,
    } 
    
    # plot the result table
    plot_cond_diff(result_table, shortnames_scms, config_folder / 'result_table_cond_diff.pdf', **kwargs)
    plot_accrates(result_table, shortnames_scms, config_folder / 'result_table_accrates.pdf', **kwargs)
    plt.close('all')
    
def visualize_result_tables(base_path: str | Path, filters, drop_settings=None):
    base_path = Path(base_path)
    folders = get_relevant_folders(base_path, filters)
    
    name = '_'.join(filters)
    
    result_tables = []
    for folder in folders:
        folder = Path(folder)
        result_table_path = folder / 'result_table.csv'
        if not result_table_path.exists():
            print(f"❌ No result table found in {folder}.")
            continue
        result_table = pd.read_csv(result_table_path)
        result_tables.append(result_table)
    # aggregate the tables
    
    big_table = pd.concat(result_tables, ignore_index=True)
    big_table['rec_diff'] = big_table['rec_refit'] - big_table['rec_orig']
    config_cols = ['goal', 'approach', 'scm_name', 'model']
    # group by the config columns, and get the mean and std of the rest
    agg_table = big_table.groupby(config_cols).agg(['mean', 'std'])
    agg_table.columns = agg_table.columns.map(lambda x: f"{x[0]}_{x[1]}")
    agg_table.reset_index(inplace=True)
    
    agg_table.to_csv(base_path / (name + '_agg_table.csv'), index=False)

    agg_table['rec_diff_lower'] = agg_table['rec_diff_mean'] - agg_table['rec_diff_std']
    agg_table['rec_diff_upper'] = agg_table['rec_diff_mean'] + agg_table['rec_diff_std']
    shortnames = shortnames_scms.copy()
    shortnames.update(shortnames_applications)
    
    scms_in_ex = agg_table['scm_name'].unique()
    for key in list(shortnames.keys()):
        if key not in scms_in_ex:
            shortnames.pop(key, None)

    if drop_settings is not None:
        for scm_name in drop_settings:
            agg_table.drop(agg_table[agg_table['scm_name'] == scm_name].index, inplace=True)
            shortnames.pop(scm_name, None)
    
    # default aesthetics
    kwargs = {
        'fig_size':(4.5, 2.7),
        'font_size': 10,
        'pointsize': 6.5,
        'palette': METHOD_PALETTE,
    } 
        
    plot_cond_diff_agg(agg_table, shortnames, base_path / (name + '_agg_table_cond_diff.pdf'), legend=True, **kwargs)
    plot_accrates_agg(agg_table, shortnames, base_path / (name + '_agg_table_accrates.pdf'), **kwargs)
    plot_accrates_diff_agg(agg_table, shortnames, base_path / (name + '_agg_table_accrates_diff.pdf'), **kwargs)
    plt.close('all')
    
    shortnames_sub = {k: v for k, v in shortnames.items() if k in agg_table['scm_name'].values}
    df = rename_sort_scms(agg_table, shortnames_sub)
    df_latex = df[['type', 'scm_name', 'cond_diff_min_mean', 'cond_diff_min_std',
                   'cond_diff_exp_mean', 'cond_diff_exp_std', 'cond_diff_max_mean', 'cond_diff_max_std',
                   'rec_diff_mean', 'rec_diff_std']]
    
    latex_code = produce_latex_table(df_latex)
    # find entries $\pm nan$ or $nan$ and replace with $-$ and $-$
    latex_code = latex_code.replace('$\\pm nan$', '$-$')
    latex_code = latex_code.replace('$nan$', '$-$')
    
    with open(base_path / (name + '_agg_table.tex'), 'w') as f:
        f.write(latex_code) 
    
def produce_latex_table(df_latex: pd.DataFrame):
    # 1. Sort by type and scm_name
    df = df_latex.sort_values(by=['type', 'scm_name'])

    # 2. Set MultiIndex
    df.set_index(['type', 'scm_name'], inplace=True)
    # sort clumns
    df = df.reindex(sorted(df.columns), axis=1)

    # 3. Identify outcomes
    outcomes = sorted(set(col.rsplit('_', 1)[0] for col in df.columns))

    # 4. Build MultiIndex for columns
    arrays = [[outcome for outcome in outcomes for _ in (0, 1)],
            ['mean', 'std'] * len(outcomes)]
    df.columns = pd.MultiIndex.from_arrays(arrays)

    # 5. Format the values as LaTeX strings
    def format_value(val, kind):
        if kind == 'mean':
            return f"${val:.2f}$"
        else:
            return f"$\\pm {val:.2f}$"

    formatted_df = df.copy()
    for outcome in outcomes:
        for kind in ['mean', 'std']:
            formatted_df[(outcome, kind)] = df[(outcome, kind)].apply(lambda x: format_value(x, kind))

    # rename columns
    rename_dict = {
        'scm_name': 'Setting',
        'type': 'Method',
        'cond_diff_min': 'Min. Dist. Diff.',
        'cond_diff_exp': 'Exp. Dist. Diff.',
        'cond_diff_max': 'Max. Dist. Diff.',
        'rec_diff': 'Acc. Rate Diff.',
    }
    
    formatted_df.rename(columns=rename_dict, inplace=True)
    #rename index levels
    formatted_df.index.names = ['Method', 'Setting']

    # 6. Export to LaTeX
    latex_code = formatted_df.to_latex(
        multicolumn=True,
        multicolumn_format='c',
        multirow=True,
        escape=False,
        column_format='ll' + 'cc' * len(outcomes)
    )

    return latex_code



def get_relevant_folders(base_folder: str | Path, filters: list[str]) -> list[str]:
    """
    Get the relevant folders for the analysis.
    """
    base_folder = Path(base_folder)
    config_folders = collect_subfolders(base_folder)
    config_folders = [f for f in config_folders if all(x in f for x in filters)]
    # assert len(config_folders) == 22
    return config_folders

    
# BASE_FOLDER = Path('results/')
# config_folder = collect_subfolders(BASE_FOLDER)
# run_folder = collect_subfolders(config_folder[-1])[-3]

# result_dict = load_data(run_folder)
# config = load_config(config_folder[-1])

FILTERS = ['cfg_seed-4', '0.9', 'N-SAMPLES-1000']

DISABLE_MAIN = False
SKIP_NORMAL = False 
FORCE_RECOMPUTE = False
if __name__ == '__main__' and not DISABLE_MAIN:
    BASE_FOLDER = Path('results/recourse/')
    config_folders = get_relevant_folders(BASE_FOLDER, FILTERS)
    if not SKIP_NORMAL:
        for config_folder in config_folders:
            print(f"📂 Processing {config_folder}")
            config_folder = Path(config_folder)
            final_result_path = config_folder / 'result_table_accrates.pdf'
            if not final_result_path.exists() or FORCE_RECOMPUTE:
                run_folders = collect_subfolders(config_folder)
                for run_folder in run_folders:
                    # print(f"📂📁 Processing {run_folder}")
                    try:
                        process_folder(config_folder, run_folder)
                        print(f"📂📁✅ {run_folder} completed successfully.")
                    except Exception as e:
                        print(f"📂📁❌ {run_folder} failed. Error: {e}")
                        continue
                try:
                    aggregate_rows_result_table(config_folder)
                    visualize_result_table(config_folder)
                    print(f"📂✅ Aggregated rows and visualized result table for {config_folder}.")
                except Exception as e:
                    print(f"📂❌ Failed to aggregate rows for {config_folder}. Error: {e}")
                    continue
    # aggregate all result tables
    visualize_result_tables(BASE_FOLDER, FILTERS, drop_settings= ['binomial_linear_noninvertiblepolynomial'])
    print("All simulations processed.")
