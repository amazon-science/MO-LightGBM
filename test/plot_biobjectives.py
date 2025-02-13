import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from matplotlib.gridspec import GridSpec
import json
import os
from latex_utils import latexify
from utils import log2dataframe, varying_alpha_line
import pickle as pkl

baseline_folder = "baseline_results"
biobjective_folder = "biobjective_results"
with open(os.path.join(baseline_folder, "log_results.json"), 'r') as f:
    baseline_log_results = json.load(f)
with open(os.path.join(biobjective_folder, "log_results.pkl"), 'rb') as f:
    pref_biobjective_log_results = pkl.load(f)
with open(os.path.join(biobjective_folder, "ec_log_results.pkl"), 'rb') as f:
    ecub_biobjective_log_results = pkl.load(f)

pref_combinators = ["linear_scalarization",
                    "chebychev_scalarization",
                    # "stochastic_label_aggregation",
                    "epo_search",
                    # "wc_mgda",
                    ]
ecub_combinators = ["e_constraint",
                    # "ec_mgda",
                    ]
mg_combinators = pref_combinators + ecub_combinators

comb_legends = {"epo_search": "EPO-Search",
                "e_constraint": "$\epsilon-$Constraint",
                "linear_scalarization": "LS",
                "wc_mgda": "WC-MGDA",
                "chebychev_scalarization": "CS",
                "stochastic_label_aggregation": "SLA"}
markers = {"linear_scalarization": "s",
           "epo_search": "*",
           # "wc_mgda": "^",
           "e_constraint": r'$\clubsuit$',
           "chebychev_scalarization": '^',
           "stochastic_label_aggregation": 'x'
           }
msz = {"epo_search": 40,
       "linear_scalarization": 18,
       "wc_mgda": 18,
       "e_constraint": 30,
       "chebychev_scalarization": 25,
       "stochastic_label_aggregation": 18}

# all_labels = ["130", "132", "133", "134", "135", "136", "target"]
# bilabels = [(mo_label, target) for i, mo_label in enumerate(all_labels[:-1]) for target in all_labels[i+1:]]
all_labels = ["132", "133", "134", "136", "target"]
# bilabels = [('130', '135'), ("134", "135"), ('133', 'target')] # , ('134', '136'), ('132', 'target'), ('134', 'target')]
bilabels = [('132', '133'), ('132', 'target'), ('134', '136'), ('134', 'target')]
label_lengends = {'132': "Quality Score",
                  '133': "Quality Score2",
                  '134': "Query-URL ClickCount",
                  '136': "URL Dwell Time",
                  'target': "Relevance Judgment"
                  }


num_prefs = 5
var_types = {"loss": "Training-loss", "ndcg": "Validation-ndcg@5"}
variables = list(var_types.values())
last_iter = 600
snapshots_at = np.arange(int(last_iter*0.1), last_iter)
# snapshots_at = np.arange([last_iter])
remove_iters = [1,2]
snapshots_at = np.delete(snapshots_at, remove_iters)
# snapshots_at = [10, 20, 40, 80, 160, 240, 480, 600]


def get_bilabel_data(log_results, combinators, parameter='preferences'):
    data = dict()
    for label_pair in bilabels:
        data[label_pair] = {combinator: {"loss": [], "ndcg": []} for combinator in combinators}
        data[label_pair][parameter] = []
        for i, param_data in enumerate(log_results[label_pair]):
            data[label_pair][parameter].append(param_data[parameter])
            for combinator in combinators:
                logfilename = param_data[combinator]
                logfile = os.path.join(biobjective_folder, logfilename)
                results = log2dataframe(logfile, variables)
                for vname, vtype in var_types.items():
                    try:
                        results_at_snapshots = results[vtype].iloc[snapshots_at].to_numpy().T
                    except IndexError as e:
                        ealy_stop_iter = results[vtype].shape[0]
                        if ealy_stop_iter < 1:
                            raise IndexError(e)
                        print(f'**Warning**: {logfilename} early stopped at {ealy_stop_iter}')
                        temp_snapshots_at = [ss for ss in snapshots_at if ss < ealy_stop_iter-1] + [ealy_stop_iter-1]
                        temp_result = results[vtype].iloc[temp_snapshots_at].to_numpy().T
                        last_results = [temp_result[:, -1] for _ in range(len(snapshots_at) - len(temp_snapshots_at))]
                        results_at_snapshots = np.column_stack([temp_result] + last_results)

                    data[label_pair][combinator][vname].append(results_at_snapshots)

        # make everything ndarray
        data[label_pair][parameter] = np.asarray(data[label_pair][parameter])
        for combinator in combinators:
            for vname in var_types:
                data[label_pair][combinator][vname] = np.asarray(data[label_pair][combinator][vname])

    return data

pref_data = get_bilabel_data(pref_biobjective_log_results, pref_combinators, parameter='preferences')
ub_data = get_bilabel_data(ecub_biobjective_log_results, ecub_combinators, parameter='ub')

# Get baseline data
baseline_results = dict()
for label in all_labels:
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, variables)
    baseline_results[label] = dict()
    for vname, vtype in var_types.items():
        results[vtype].columns = all_labels
        baseline_results[label][vname] = results[vtype].iloc[last_iter][label]  # .iloc[snapshots_at][0].to_numpy()
# data for baseline
baseline_data = dict()
for label_pair in bilabels:
    sec_label, prim_label = label_pair
    baseline_data[label_pair] = dict()
    for vname in var_types:
        baseline_data[label_pair][vname] = (baseline_results[sec_label][vname], baseline_results[prim_label][vname])

colors = sns.color_palette("tab10", num_prefs)
pdf = backend_pdf.PdfPages(f"figures/bi_objectives-{last_iter}.pdf")    # in one pdf
latexify(fig_width=4, fig_height=1.8)
for pi, label_pair in enumerate(bilabels):
    mo_label, target = label_pair
    fig = plt.figure()
    
    # ------------------ LOSS PLOT -------------------
    ax = fig.add_subplot(121)
    loss_ulim = np.max(np.stack([np.max(pref_data[label_pair][comb]['loss'], axis=(0, 2)) for comb in pref_combinators]), axis=0)
    ax.set_xlim(0, loss_ulim[0])
    ax.set_ylim(0, loss_ulim[1])
    ax.set_title(f'Training Cost')
    ax.set_xlabel(f"{label_lengends[mo_label]}", labelpad=1)
    ax.set_ylabel(f"{label_lengends[target]}")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot baseline
    sec_loss, prim_loss =  baseline_data[label_pair]["loss"]
    ax.plot([0, loss_ulim.sum()], [prim_loss, prim_loss], lw=2, alpha=0.4, c='k')   # horizontal line
    ax.plot([sec_loss, sec_loss], [0, loss_ulim.sum()], lw=2, alpha=0.4, c='k')     # vertical line

    # plot preference rays and trajectories
    for pref_id, r in enumerate(pref_data[label_pair]["preferences"]):
        lgnd = r"$\mathbf{r}^{-1}$ Ray" if pref_id == 0 and pi == 0 else ""
        r_inv = 1. / r
        r_inv /= np.linalg.norm(r_inv)
        r_inv /= r_inv.max()
        ax.plot([0, loss_ulim.sum() * r_inv[0]], [0, loss_ulim.sum() * r_inv[1]], color=colors[pref_id],
                lw=0.5, alpha=0.5, ls='--', dashes=(10, 2), label=lgnd)
        # Plot trajectories
        #     for combinator in combinators:
        #         ls = data[label_pair][combinator]["loss"]
        #         x, y = ls[pref_id, 0, :], ls[pref_id, 1, :]
        #         varying_alpha_line(x, y, ax, c=colors[pref_id], lw=0.5)

    # plot ub bound lines
    for ub_id, ub in enumerate(ub_data[label_pair]['ub']):
        lgnd = r"$\epsilon$ bound" if ub_id == 0 and pi == 2 else ""
        ax.plot([ub, ub], [0, loss_ulim.sum()], color=colors[ub_id],
                lw=0.5, alpha=0.5, ls='-', label=lgnd)
    # Plot end points
    for combinators, data in zip([pref_combinators, ecub_combinators], [pref_data, ub_data]):
        for combinator in combinators:
            ls = data[label_pair][combinator]["loss"]
            ax.scatter(ls[:, 0, -1], ls[:, 1, -1], marker=markers[combinator], edgecolor='k', lw=0.3,
                       c=colors, s=msz[combinator], zorder=10)
    if pi in [0, 2]:
        ax.legend(fontsize=7, loc='lower right', facecolor=(1, 1, 1, 1))

    # # -------------- NDGC plot ---------------
    llim = np.min(np.stack([np.min(pref_data[label_pair][comb]['ndcg'], axis=(0, 2)) for comb in pref_combinators]), axis=0)
    ulim = np.max(np.stack([np.max(pref_data[label_pair][comb]['ndcg'], axis=(0, 2)) for comb in pref_combinators]), axis=0)
    ax = fig.add_subplot(122)
    ax.set_title(f"Validation NDCG@5", y=-0.2)
    ax.set_xlabel(f"{label_lengends[mo_label]}")
    ax.set_ylabel(f"{label_lengends[target]}")
    ax.set_xlim(llim[0] * 0.99, ulim[0] * 1.005)
    ax.set_ylim(llim[1] * 0.99, ulim[1] * 1.005)
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_label_position('top')

    sec_ndcg, prim_ndcg = baseline_data[label_pair]["ndcg"]
    lgnd = "baseline" if pi == 1 else ""
    ax.plot([0, 1], [prim_ndcg, prim_ndcg], lw=2, alpha=0.4, c='k', label=lgnd)   # horizontal line
    ax.plot([sec_ndcg, sec_ndcg], [0, 1], lw=2, alpha=0.4, c='k')                 # vertical line

    for combinators, data in zip([pref_combinators, ecub_combinators], [pref_data, ub_data]):
        for combinator in combinators:
            ndcgs = data[label_pair][combinator]["ndcg"]
            if (combinator in pref_combinators and pi == 0) or (combinator in ecub_combinators and pi == 2):
                lgnd = comb_legends[combinator]
            else:
                lgnd = ''
            ax.scatter(ndcgs[:, 0, -1], ndcgs[:, 1, -1], marker=markers[combinator], edgecolor='k', lw=0.3,
                       c=colors, s=msz[combinator], label=lgnd, zorder=10)
    if pi in [0, 1, 2]:
        ax.legend(fontsize=7, loc='lower left', facecolor=(1, 1, 1, 1))
    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.87, top=0.78 if pi > 1 else 0.82, bottom=0.175)
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
