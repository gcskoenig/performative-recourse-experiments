import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from recourse.dgps import get_rec_setup, invertible_scms, noninvertible_scms

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from recourse.viz import prepare_heatmap_data, plot_heatmap

TARGET_NAME = 'y'
MODEL = DecisionTreeClassifier()
N_FIT = 1000
N_REC = 1000
THRESH_PRED = 0.5
FIGSIZE = (10, 8)

SAVEPATH = Path("results") / "dgpviz"
SAVEPATH.mkdir(parents=True, exist_ok=True)

all_scms = {**invertible_scms, **noninvertible_scms}

for (name, scm) in all_scms.items():
    print(f"Processing {name}...")
    model = clone(MODEL) 
    out = get_rec_setup(scm, TARGET_NAME, N_REC, N_FIT, THRESH_PRED, model)
    scm, target_name, thresh_label, fn_l, cost_dict, predict, X_rej, (obss_rej, epss_rej) = out

    obss, epss = scm.sample(sample_shape=(10**6,))

    # get heatmap data
    heatmap_data, count_data = prepare_heatmap_data(obss, x1='x1', x2='x2', y=TARGET_NAME)
   
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE) 
    plot_heatmap(heatmap_data, x1='x1', x2='x2', ax=axs[0], vmin=None, vmax=None, center=None)
    axs[0].set_title("E[Y|X]") 
    plot_heatmap(count_data, x1='x1', x2='x2', ax=axs[1], vmin=0, vmax=None, center=None)
    axs[1].set_title("Count")
    plt.savefig(SAVEPATH / f"{name}_heatmap.pdf")
    plt.close(fig)

    # compute L and preds
    X = obss[['x1', 'x2']]
    obss['y_pred'] = predict(X)
    obss['l'] = fn_l(obss[TARGET_NAME])

    # get heatmap w.r.t. L
    heatmap_data, count_data = prepare_heatmap_data(obss, x1='x1', x2='x2', y='l')
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
    plot_heatmap(heatmap_data, x1='x1', x2='x2', ax=axs[0], vmin=0, vmax=1, center=0.5)
    axs[0].set_title("E[L|X]")
    plot_heatmap(count_data, x1='x1', x2='x2', ax=axs[1], vmin=0, vmax=None, center=None)
    axs[1].set_title("Count")
    plt.savefig(SAVEPATH / f"{name}_heatmap_L.pdf")
    plt.close(fig)
    # get heatmap w.r.t. preds
    heatmap_data, count_data = prepare_heatmap_data(obss, x1='x1', x2='x2', y='y_pred')
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
    plot_heatmap(heatmap_data, x1='x1', x2='x2', ax=axs[0], vmin=0, vmax=1, center=0.5)
    axs[0].set_title("E[Y_pred|X]")
    plot_heatmap(count_data, x1='x1', x2='x2', ax=axs[1], vmin=0, vmax=None, center=None)
    axs[1].set_title("Count")
    plt.savefig(SAVEPATH / f"{name}_heatmap_y_pred.pdf")
    plt.close(fig)
    