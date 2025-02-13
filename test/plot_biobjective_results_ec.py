import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
import json
import os
from latex_utils import latexify
from utils import log2dataframe
import pickle as pkl

baseline_folder = "baseline_results"
biobjective_folder = "biobjective_results"
log_results_file = "log_results.json"
ec_log_results_file = "ec_log_results.pkl"
baseline_log_results_file = os.path.join(baseline_folder, log_results_file)
biobjective_ec_log_results_file = os.path.join(biobjective_folder, ec_log_results_file)
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
with open(biobjective_ec_log_results_file, 'rb') as f:
    biobjective_ec_log_results = pkl.load(f)
    # print(f'all pairs:\n{list(biobjective_log_results.keys())}')

combinators = [
#             "ec_mgda",
               "e_constraint"]   #
markers = {"epo_search": "*", "linear_scalarization": "s", "e_constraint": "^", "ec_mgda": "s"}
msz = {"epo_search": 60, "linear_scalarization": 20, "e_constraint": 40, "ec_mgda": 40}

# all_labels = ["130", "132", "133", "134", "135", "136", "target"]
# bilabels = [(mo_label, target) for i, mo_label in enumerate(all_labels[:-1]) for target in all_labels[i+1:]]
all_labels = ["132", "133", "134", "136", "target"]  # , "134", "136", 'target']
# bilabels = [('130', '135'), ("134", "135"), ('133', 'target')] # , ('134', '136'), ('132', 'target'), ('134', 'target')]
bilabels = [('132', '133'), ('132', 'target'), ('134', '136'), ('134', 'target')]
num_ubs = 5
var_types = {"loss": "Training-loss", "ndcg": "Training-ndcg@5"}
variables = list(var_types.values())
# snapshots_at = [100, 200, 300]
# snapshots_at = [200, 300, 400, 500, 600]
snapshots_at = [10, 50, 100, 200, 400, 600]  # , 900, 1200, 1500]
# snapshots_at = list(range(0, 11))
# snapshots_at = list(range(0,31,4))

# Get baseline data
baseline_data = dict()
for label in all_labels:
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, variables)
    baseline_data[label] = dict()
    for vname, vtype in var_types.items():
        results[vtype].columns = all_labels
        baseline_data[label][vname] = results[vtype].iloc[snapshots_at][label].to_numpy()   # .iloc[snapshot_iteration][0]

data = dict()
for label_pair in bilabels:
    data[label_pair] = {combinator: {"loss": [], "ndcg": []} for combinator in combinators}
    data[label_pair]['ubs'] = [None] * num_ubs
    data[label_pair]['preflens'] = [0] * num_ubs
    for i, ub_data in enumerate(biobjective_ec_log_results[label_pair]):
        data[label_pair]['ubs'][i] = ub_data['ub']
        for combinator in combinators:
            logfilename = ub_data[combinator]
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


    data[label_pair]['ubs'] = np.asarray(data[label_pair]['ubs'])
    for combinator in combinators:
        for vname in var_types:
            data[label_pair][combinator][vname] = np.asarray(data[label_pair][combinator][vname])

    # data for baseline
    data[label_pair]["baseline_loss"] = []
    data[label_pair]["baseline_ndcg"] = []
    len_baseline = max(data[label_pair]['preflens'])
    mo_label, target = label_pair
    for axid in range(2):
        if axid == 0:   # horizontal baseline
            loss_y, ndcg_y = baseline_data[target]["loss"], baseline_data[target]["ndcg"]
            loss_xs = np.stack([np.zeros(len(snapshots_at)), np.ones(len(snapshots_at)) * 1.1 * len_baseline])
            loss_ys = np.stack([loss_y, loss_y])
            ndcg_xs = np.stack([np.zeros(len(snapshots_at)), np.ones(len(snapshots_at))])
            ndcg_ys = np.stack([ndcg_y, ndcg_y])
        else:           # vertical baselie
            loss_x, ndcg_x = baseline_data[mo_label]["loss"], baseline_data[mo_label]["ndcg"]
            loss_xs = np.stack([loss_x, loss_x])
            loss_ys = np.stack([np.zeros(len(snapshots_at)), np.ones(len(snapshots_at)) * 1.1 * len_baseline])
            ndcg_xs = np.stack([ndcg_x, ndcg_x])
            ndcg_ys = np.stack([np.zeros(len(snapshots_at)), np.ones(len(snapshots_at))])
        data[label_pair]["baseline_loss"].append((loss_xs, loss_ys))
        data[label_pair]["baseline_ndcg"].append((ndcg_xs, ndcg_ys))

# ------------------ LOSS PLOT -------------------
# latexify(fig_width=2.25, fig_height=1.8)  # (fig_width=2.25, fig_height=1.5)
latexify(fig_width=4.5, fig_height=2)
for label_pair in bilabels:
    mo_label, target = label_pair
    # loss_pdf = backend_pdf.PdfPages(f"figures/{mo_label}-{target}_loss_{snapshots_at[0]}-{snapshots_at[-1]}.pdf")

    # in one pdf
    pdf = backend_pdf.PdfPages(f"figures/ec_{mo_label}-{target}_{snapshots_at[0]}-{snapshots_at[-1]}.pdf")
    llim = np.min(np.stack([np.min(data[label_pair][comb]['ndcg'], axis=(0,2)) for comb in combinators]), axis=0)
    ulim = np.max(np.stack([np.max(data[label_pair][comb]['ndcg'], axis=(0,2)) for comb in combinators]), axis=0)
    loss_llim = np.min(np.stack([np.min(data[label_pair][comb]['loss'], axis=(0,2)) for comb in combinators]), axis=0)
    loss_ulim = np.max(np.stack([np.max(data[label_pair][comb]['loss'], axis=(0,2)) for comb in combinators]), axis=0)
    for i, snapshot_at in enumerate(snapshots_at):
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        ax = fig.add_subplot(121)
        axlim = max(data[label_pair]['preflens'])
        ax.set_xlim(-0.1, max(loss_ulim[0], data[label_pair]["ubs"].max()) * 1.1)
        ax.set_ylim(-0.1, loss_ulim[1] * 1.1)
        ax.set_title(f'at iteration {snapshot_at}')
        ax.set_xlabel(f"{mo_label} lamdarank loss")
        ax.set_ylabel(f"{target} lambdarank loss")
        for axid, (xs, ys) in enumerate(data[label_pair]["baseline_loss"]):
            ax.plot(xs[:, i], ys[:, i], lw=2, alpha=0.4, c='k')

        colors = []
        for ub_id, ub in enumerate(data[label_pair]["ubs"]):
            lgnd = "ub" if ub_id == 0 else ""
            lines = ax.plot([ub, ub], [-0.1, axlim],
                            lw=1, alpha=0.5, ls='--', dashes=(10, 2), label=lgnd)
            colors.append(lines[0].get_color())

        for combinator in combinators:
            ls = data[label_pair][combinator]["loss"]
            ax.scatter(ls[:, 0, i], ls[:, 1, i], marker=markers[combinator],
                       c=colors, s=msz[combinator])
        ax.legend()
        # fig.tight_layout()
        # loss_pdf.savefig(fig)
        # plt.close(fig)
        # loss_pdf.close()

        # # -------------- NDGC plot ---------------
        # for label_pair in bilabels:
        #     mo_label, target = label_pair
        #     ndcg_pdf = backend_pdf.PdfPages(f"figures/{mo_label}-{target}_ndcg_{snapshots_at[0]}-{snapshots_at[-1]}.pdf")
        #     llim = np.min(np.stack([np.min(data[label_pair][comb]['ndcg'], axis=(0,2)) for comb in combinators]), axis=0)
        #     ulim = np.max(np.stack([np.max(data[label_pair][comb]['ndcg'], axis=(0,2)) for comb in combinators]), axis=0)
        #     for i, snapshot_at in enumerate(snapshots_at):
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111)
        ax = fig.add_subplot(122)
        # ax.set_title(f'at iteration {snapshot_at}', y=-0.2)
        ax.set_xlabel(f"{mo_label} " + var_types['ndcg'])
        ax.set_ylabel(f"{target} " + var_types['ndcg'])
        ax.set_xlim(llim[0] * 0.99, ulim[0] * 1.001)
        ax.set_ylim(llim[1] * 0.99, ulim[1] * 1.001)
        # Hide the right and top spines
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_label_position('right')
        ax.xaxis.set_label_position('top')

        for bl_id, (xs, ys) in enumerate(data[label_pair]["baseline_ndcg"]):
            lgnd = "baseline" if bl_id == 0 else ""
            ax.plot(xs[:, i], ys[:, i], lw=2, alpha=0.4, c='k', label=lgnd)

        for combinator in combinators:
            ndcgs = data[label_pair][combinator]["ndcg"]
            lgnd = combinator  # if fig_id == 1 else ""
            ax.scatter(ndcgs[:, 0, i], ndcgs[:, 1, i], marker=markers[combinator],
                       c=colors, s=msz[combinator], label=lgnd)

        # if fig_id in [1, 2]:
        ax.legend(fontsize=5)
        fig.tight_layout()
        # ndcg_pdf.savefig(fig)
        pdf.savefig(fig)
        plt.close(fig)
    # ndcg_pdf.close()
    pdf.close()
