import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import seaborn.objects as so


HEATMAP_KWARGS = {
    'cmap': 'coolwarm',
    'cbar': True,
    'cbar_kws': {'label': 'Proportion of y=1'},
    'linewidths': 0.5,
    'linecolor': 'gray',
    'square': True,
    'annot': True,
    'center': 0,
    'vmin': -0.5,
    'vmax': 0.5,
}

METHOD_PALETTE = [
    '#29B390',
    '#5CBACC',
    '#BD62CC',
    '#B33970',
    '#B3AB14'
]


def prepare_heatmap_data(df: pd.DataFrame, x1: str='x1', x2: str='x2', y: str='l'):
    # Compute proportion of y=1 for each combination
    prop_df = (
        df.groupby([x1, x2])[y]
        .mean()
        .reset_index(name='p_y1')
    )
    
    # get count of each combination
    count_df = (
        df.groupby([x1, x2])[y]
        .count()
        .reset_index(name='count')
    )

    # Pivot to 2D matrix for heatmap
    heatmap_data = prop_df.pivot(index=x2, columns=x1, values='p_y1')
    
    # pivot to 2D matrix for count
    count_data = count_df.pivot(index=x2, columns=x1, values='count')

    return heatmap_data, count_data

def get_diff_heatmap(df_pre: pd.DataFrame, df_post:pd.DataFrame,
                     x1:str='x1', x2:str='x2', y:str='l'):
    """Creates diff between two heatmaps, uniformly weighs all cells supported by both."""
    h1, c1 = prepare_heatmap_data(df_pre, x1=x1, x2=x2, y=y)
    h2, c2 = prepare_heatmap_data(df_post, x1=x1, x2=x2, y=y)
    diff = h1 - h2
    # weigh by the empirical P(X^p=x)*(P(X=x)>0) (normalized)
    diff_weights = c2 * (c1 > 0).astype(int) 
    diff_weights = diff_weights / diff_weights.sum().sum()
    diff_uniform_weights = ((c1 * c2) > 0).astype(int)
    diff_uniform_weights = diff_uniform_weights / diff_uniform_weights.sum().sum()
    heatmap_stats = {
        'expected_diff': (diff * diff_weights).sum().sum(),
        'expected_abs_diff': (diff.abs() * diff_weights).sum().sum(),
        'uniform_expected_diff': (diff * diff_uniform_weights).sum().sum(),
        'uniform_expected_abs_diff': (diff.abs() * diff_uniform_weights).sum().sum(),
        'max_diff': diff.max().max(),
        'min_diff': diff.min().min(), 
    }
    return diff, diff_weights, heatmap_stats

def plot_heatmap(heatmap_data: pd.DataFrame, x1:str='x1', x2:str='x2', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    kwargs_local = HEATMAP_KWARGS.copy()

    for key, value in kwargs.items():
        kwargs_local[key] = value

    ax = sns.heatmap(
        heatmap_data,
        ax=ax,
        **kwargs_local,
    )    
    ax.set_xlabel(x1)
    ax.set_ylabel(x2)
    
def rename_sort_scms(df_orig, namedict):
    # take the scm_name column and replace the values with the namedict[value] value
    df = df_orig.copy()
    df['scm_name'] = df['scm_name'].replace(namedict)
    # sort the dataframe by the order of the keys in the namedict
    df['scm_name'] = pd.Categorical(df['scm_name'], list(namedict.values()), ordered=True)
    df = df.sort_values('scm_name')
    
    # rename goal to method and approach to version
    df['method'] = df['goal']
    df['version'] = df['approach']
    
    # rename the entries for method using the dictionary
    df['method'] = df['method'].replace({'acc': 'CR', 'imp': 'ICR'})
    # rename the entries for version using the dictionary
    df['version'] = df['version'].replace({'ind': 'ind.', 'sub': 'sub.', 'ce': 'CE'})
    df['type'] = df['version'] + ' ' + df['method']
    df['type'] = df['type'].replace({'CE CR': 'CE'})
    
    # custom ordering for type
    df['type'] = pd.Categorical(df['type'], ['ind. ICR', 'sub. ICR', 'ind. CR', 'sub. CR', 'CE'], ordered=True)
    
    df['method'] = df['method'].astype('category')
    df['version'] = df['version'].astype('category')
    
    return df

def plot_cond_diff(
        df_orig: pd.DataFrame,
        shortnames_scms: dict,
        savepath: str | Path,
        *,
        within_gap: float = 0.35,          # width of dodged offsets
        fig_size: tuple[float, float] = (7.2, 3.0),
        font_size: int = 9,
        pointsize: int = 9,
        palette: str | list | tuple = "muted",
        y_limits: tuple[float, float] = (-1.1, 1.1),
        dpi: int = 300,
):
    df = rename_sort_scms(df_orig, shortnames_scms)
    with plt.rc_context({
        "font.size":        font_size,
        "axes.labelsize":   font_size,
        "xtick.labelsize":  font_size,
        "ytick.labelsize":  font_size,
    }):
        fig, ax = plt.subplots(figsize=fig_size)

        (
            so.Plot(
                df, x="scm_name", y="cond_diff_exp",
                ymin="cond_diff_min", ymax="cond_diff_max",
                color="type"
            )
            .add(so.Range(alpha=.6, linewidth=1.3), so.Dodge(gap=within_gap), legend=False)
            .add(so.Dot(pointsize=pointsize),         so.Dodge(gap=within_gap), legend=False)
            .label(x="SCM",
                y="$P(Y^p=1|X^p=x) - P(Y=1|X=x)$",
                color="")
            # ---- scales -------------------------------------------------------------
            .scale(color=palette)    # type: ignore                 # your custom colours
            # ------------------------------------------------------------------------
            .theme({"legend.fontsize": font_size,
                    "legend.title_fontsize": font_size})
            .limit(y=y_limits)
            .on(ax)
            .plot()
        )
                
        sns.despine(left=True)

        fig.tight_layout()
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        
        
def plot_accrates(
        df_orig: pd.DataFrame,
        shortnames_scms: dict,
        savepath: str | Path,
        *,
        within_gap: float = 0.35,          # width of dodged offsets
        fig_size: tuple[float, float] = (7.2, 3.0),
        font_size: int = 10,
        pointsize: int = 10,
        palette: str | list | tuple = "muted",
        y_limits: tuple[float, float] = (-0.1, 1.1),
        dpi: int = 300,
):
    df = rename_sort_scms(df_orig, shortnames_scms)
    with plt.rc_context({
        "font.size":        font_size,
        "axes.labelsize":font_size,
        "xtick.labelsize":  font_size,
        "ytick.labelsize":  font_size,
    }):
        fig, ax = plt.subplots(figsize=fig_size)

        (
            so.Plot(
                df, x="scm_name",
                color="type"
            )
            .add(so.Range(alpha=.6, linewidth=1.3), so.Dodge(gap=within_gap),
                 ymin='rec_orig', ymax='rec_refit', legend=False)
            .add(so.Dot(pointsize=pointsize, marker='d'), so.Dodge(gap=within_gap),
                 y='rec_refit', legend=False)
            .add(so.Dot(pointsize=pointsize*0.5, marker='o'), so.Dodge(gap=within_gap),
                 y='rec_orig', legend=False)
            .label(x="SCM",y="Acceptance rate pre vs. post refit",
                color="")
            # ---- scales -------------------------------------------------------------
            .scale(color=palette)   # type: ignore                  # your custom colours
            # ------------------------------------------------------------------------
            .theme({"legend.fontsize": font_size,
                    "legend.title_fontsize": font_size})
            .limit(y=y_limits)
            .on(ax)
            .plot()
        )
                
        sns.despine(left=True)

        fig.tight_layout()
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def plot_accrates_agg(
        df_orig: pd.DataFrame,
        shortnames_scms: dict,
        savepath: str | Path,
        *,
        within_gap: float = 0.35,          # horizontal offset of the two points
        fig_size: tuple[float, float] = (7.2, 3.0),
        font_size: int = 10,
        pointsize: int = 10,
        palette: str | list | tuple = "muted",
        y_limits: tuple[float, float] = (-0.1, 1.1),
        dpi: int = 300,
):
    df = rename_sort_scms(df_orig, shortnames_scms)
    with plt.rc_context({
        "font.size":        font_size,
        "axes.labelsize":font_size,
        "xtick.labelsize":  font_size,
        "ytick.labelsize":  font_size,
    }):
        fig, ax = plt.subplots(figsize=fig_size)

        (
            so.Plot(
                df, x="scm_name",
                color="type"
            )
            .add(so.Range(alpha=.6, linewidth=1.3), so.Dodge(gap=within_gap),
                 ymin='rec_orig_mean', ymax='rec_refit_mean', legend=False)
            .add(so.Dot(pointsize=pointsize, marker='d'), so.Dodge(gap=within_gap),
                 y='rec_refit_mean', legend=False)
            .add(so.Dot(pointsize=pointsize*0.5, marker='o'), so.Dodge(gap=within_gap),
                 y='rec_orig_mean', legend=False)
            .label(x="SCM",y="Acc. rate pre vs. post",
                color="")
            # ---- scales -------------------------------------------------------------
            .scale(color=palette)   # type: ignore                  # your custom colours
            # ------------------------------------------------------------------------
            .theme({"legend.fontsize": font_size,
                    "legend.title_fontsize": font_size})
            .limit(y=y_limits)
            .on(ax)
            .plot()
        )
                
        sns.despine(left=True)

        fig.tight_layout()
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def plot_accrates_diff_agg(
        df_orig: pd.DataFrame,
        shortnames_scms: dict,
        savepath: str | Path,
        *,
        within_gap: float = 0.35,          # horizontal offset of the two points
        fig_size: tuple[float, float] = (7.2, 3.0),
        font_size: int = 10,
        pointsize: int = 10,
        palette: str | list | tuple = "muted",
        y_limits: tuple[float, float] = (-1.1, 1.1),
        dpi: int = 300,
        legend: bool = False,
):
    df = rename_sort_scms(df_orig, shortnames_scms)
    with plt.rc_context({
        "font.size":        font_size,
        "axes.labelsize":font_size,
        "xtick.labelsize":  font_size,
        "ytick.labelsize":  font_size,
    }):
        fig, ax = plt.subplots(figsize=fig_size)

        (
            so.Plot(
                df, x="scm_name", y="rec_diff_mean",
                ymin="rec_diff_lower", ymax="rec_diff_upper",
                color="type"
            )
            .add(so.Range(alpha=.6, linewidth=1.3), so.Dodge(gap=within_gap), legend=legend)
            .add(so.Dot(pointsize=pointsize),         so.Dodge(gap=within_gap), legend=legend)
            .label(x="SCM",
                y="Acc. rate (refit - original)",
                color="")
            # ---- scales -------------------------------------------------------------
            .scale(color=palette)    # type: ignore                 # your custom colours
            # ------------------------------------------------------------------------
            .theme({"legend.fontsize": font_size,
                    "legend.title_fontsize": font_size})
            .limit(y=y_limits)
            .on(ax)
            .plot()
        )
                
        sns.despine(left=True)

        fig.tight_layout()
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_cond_diff_agg(
        df_orig: pd.DataFrame,
        shortnames_scms: dict,
        savepath: str | Path,
        *,
        within_gap: float = 0.35,          # width of dodged offsets
        fig_size: tuple[float, float] = (7.2, 3.0),
        font_size: int = 9,
        pointsize: int = 9,
        palette: str | list | tuple = "muted",
        y_limits: tuple[float, float] = (-1.1, 1.1),
        dpi: int = 300,
        legend: bool = True,
):
    # drop all rowns where cond_diff_mean in NaN
    df_orig = df_orig.dropna(subset=['cond_diff_exp_mean'])
    # drop all shortnames that are not in df_orig['scm_name']
    shortnames_scms = {k: v for k, v in shortnames_scms.items() if k in df_orig['scm_name'].values}
    df = rename_sort_scms(df_orig, shortnames_scms)
    with plt.rc_context({
        "font.size":        font_size,
        "axes.labelsize":   font_size,
        "xtick.labelsize":  font_size,
        "ytick.labelsize":  font_size,
    }):
        fig, ax = plt.subplots(figsize=fig_size)

        (
            so.Plot(
                df, x="scm_name", y="cond_diff_exp_mean",
                ymin="cond_diff_min_mean", ymax="cond_diff_max_mean",
                color="type"
            )
            .add(so.Range(alpha=.6, linewidth=1.3), so.Dodge(gap=within_gap), legend=legend)
            .add(so.Dot(pointsize=pointsize),         so.Dodge(gap=within_gap), legend=legend)
            .label(x="SCM",
                y="$P(L^p=1|X^p=x)-P(L=1|X=x)$",
                color="")
            # ---- scales -------------------------------------------------------------
            .scale(color=palette)    # type: ignore                 # your custom colours
            # ------------------------------------------------------------------------
            .theme({"legend.fontsize": font_size,
                    "legend.title_fontsize": font_size})
            .limit(y=y_limits)
            .on(ax)
            .plot()
        )
                
        sns.despine(left=True)

        fig.tight_layout()
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

