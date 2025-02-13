import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import json
import os
from latex_utils import latexify
from utils import log2dataframe, varying_alpha_line
import pickle as pkl

baseline_folder = "baseline_results"
biobjective_folder = "biobjective_results"
baseline_log_results_file = os.path.join(baseline_folder, "log_results.json")
biobjective_log_results_file = os.path.join(biobjective_folder, "log_results.pkl")
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
with open(biobjective_log_results_file, 'rb') as f:
    biobjective_log_results = pkl.load(f)
    # print(f'all pairs:\n{list(biobjective_log_results.keys())}')

combinators = ["linear_scalarization",
               "chebychev_scalarization",
               # "stochastic_label_aggregation",
               "epo_search",
               # "wc_mgda",
               # "ec_mgda",
               # "e_constraint",
               ]
comb_legends = {"epo_search": "EPO-Search",
               "linear_scalarization": "LS",
               "wc_mgda": "WC-MGDA",
               "chebychev_scalarization": "CS",
               "stochastic_label_aggregation": "SLA"}
markers = {"linear_scalarization": "s",
           "epo_search": "*",
           # "wc_mgda": "^",
           "chebychev_scalarization": '^',
           "stochastic_label_aggregation": 'x'
           }
msz = {"epo_search": 30,
       "linear_scalarization": 13,
       "wc_mgda": 18,
       "chebychev_scalarization": 18,
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

# Get baseline data
baseline_data = dict()
for label in all_labels:
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, variables)
    baseline_data[label] = dict()
    for vname, vtype in var_types.items():
        results[vtype].columns = all_labels
        baseline_data[label][vname] = results[vtype].iloc[last_iter][label]  # .iloc[snapshots_at][0].to_numpy()

# print(f"{label}:{baseline_data[label]['loss']}" for label in all_labels)

data = dict()
for label_pair in bilabels:
    data[label_pair] = {combinator: {"loss": [], "ndcg": []} for combinator in combinators}
    data[label_pair]['preferences'] = [None] * num_prefs
    data[label_pair]['preflens'] = [0] * num_prefs
    for i, pref_data in enumerate(biobjective_log_results[label_pair]):
        data[label_pair]['preferences'][i] = pref_data['preferences']
        # print([f"losses; {label} : {baseline_data[label]['loss']}" for label in label_pair])
        # print(f"pref: {1.0 / np.asarray(pref_data['preferences'])}")
        for combinator in combinators:
            logfilename = pref_data[combinator]
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
                if vname == 'loss':
                    loss_norms = np.linalg.norm(results_at_snapshots, axis=0)
                    data[label_pair]['preflens'][i] = max(np.max(loss_norms), data[label_pair]['preflens'][i])


    data[label_pair]['preferences'] = np.asarray(data[label_pair]['preferences'])
    for combinator in combinators:
        for vname in var_types:
            data[label_pair][combinator][vname] = np.asarray(data[label_pair][combinator][vname])

    # data for baseline
    data[label_pair]["baseline_loss"] = []
    data[label_pair]["baseline_ndcg"] = []
    len_baseline = max(data[label_pair]['preflens'])
    mo_label, target = label_pair
    for axid, lbl in enumerate(label_pair[::-1]):
        loss, ndcg = baseline_data[lbl]["loss"], baseline_data[lbl]["ndcg"]
        loss_y = np.array([loss]*2) if axid == 0 else np.array([0, 1.1*len_baseline])
        loss_x = np.array([0, 1.1*len_baseline]) if axid == 0 else np.array([loss]*2)
        ndcg_y = np.array([ndcg]*2) if axid == 0 else np.array([0, 1])
        ndcg_x = np.array([0, 1]) if axid == 0 else np.array([ndcg]*2)
        data[label_pair]["baseline_loss"].append((loss_x, loss_y))
        data[label_pair]["baseline_ndcg"].append((ndcg_x, ndcg_y))

colors = sns.color_palette("tab10", num_prefs)
# in one pdf
pdf = backend_pdf.PdfPages(f"figures/bi_objectives-{last_iter}.pdf")
# latexify(fig_width=2.25, fig_height=1.8)  # (fig_width=2.25, fig_height=1.5)
latexify(fig_width=4, fig_height=2)
for pi, label_pair in enumerate(bilabels):
    mo_label, target = label_pair
    fig = plt.figure()
    
#     sps1, sps2 = GridSpec(1,2, figure=fig)

    # ------------------ LOSS PLOT -------------------
    ax = fig.add_subplot(121)
#     ax = brokenaxes(xlims=((.1, .3), (.7, .8)), subplot_spec=sps1)
    loss_ulim = np.max(np.stack([np.max(data[label_pair][comb]['loss'], axis=(0, 2)) for comb in combinators]), axis=0)
    ax.set_xlim(-0.1, loss_ulim[0] * 1.1)
    ax.set_ylim(-0.1, loss_ulim[1] * 1.1)
    ax.set_title(f'Training Cost')
    ax.set_xlabel(f"{label_lengends[mo_label]}")
    ax.set_ylabel(f"{label_lengends[target]}")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot baseline
    for axid, (x, y) in enumerate(data[label_pair]["baseline_loss"]):
        ax.plot(x, y, lw=2, alpha=0.4, c='k')

    # plot preference rays and trajectories
    for pref_id, r in enumerate(data[label_pair]["preferences"]):
        lgnd = r"$r^{-1}$ Ray" if pref_id == 0 else ""
        r_inv = 1. / r
        r_inv /= np.linalg.norm(r_inv)
        r_inv /= r_inv.max()
        ax.plot([0, loss_ulim.sum() * r_inv[0]], [0, loss_ulim.sum() * r_inv[1]], color=colors[pref_id],
                lw=0.5, alpha=0.5, ls='--', dashes=(10, 2), label=lgnd)
        # Plot trajectories
#         for combinator in combinators:
#             ls = data[label_pair][combinator]["loss"]
#             x, y = ls[pref_id, 0, :], ls[pref_id, 1, :]
#             varying_alpha_line(x, y, ax, c=colors[pref_id], lw=0.5)

    # Plot end points
    for combinator in combinators:
        ls = data[label_pair][combinator]["loss"]
        ax.scatter(ls[:, 0, -1], ls[:, 1, -1], marker=markers[combinator], edgecolor='k', lw=0.3,
                   c=colors, s=msz[combinator], zorder=10)
    if pi == 0:
        ax.legend(fontsize=7, loc='upper left')

    # # -------------- NDGC plot ---------------
    llim = np.min(np.stack([np.min(data[label_pair][comb]['ndcg'], axis=(0,2)) for comb in combinators]), axis=0)
    ulim = np.max(np.stack([np.max(data[label_pair][comb]['ndcg'], axis=(0,2)) for comb in combinators]), axis=0)
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
    for bl_id, (x, y) in enumerate(data[label_pair]["baseline_ndcg"]):
        lgnd = "baseline" if bl_id == 0 and pi == 0 else ""
        ax.plot(x, y, lw=2, alpha=0.4, c='k', label=lgnd)

    for combinator in combinators:
        ndcgs = data[label_pair][combinator]["ndcg"]
        if (pi == 0 and combinator in ['linear_scalarization']) or (pi == 1 and combinator in ['epo_search', 'chebychev_scalarization']):
            lgnd = comb_legends[combinator]
        else:
            lgnd = ''
        ax.scatter(ndcgs[:, 0, -1], ndcgs[:, 1, -1], marker=markers[combinator], edgecolor='k', lw=0.3,
                   c=colors, s=msz[combinator], label=lgnd, zorder=10)
    if pi in [0,1]:
        ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
pdf.close()
