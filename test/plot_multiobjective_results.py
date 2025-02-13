import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from utils import log2dataframe
from latex_utils import latexify
import os


baseline_folder = "baseline_results"
multiobjective_folder = "multiobjective_results"
log_results_file = "log_results.json"
baseline_log_results_file = os.path.join(baseline_folder, log_results_file)
multiobjective_log_results_file = os.path.join(multiobjective_folder, log_results_file)
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
with open(multiobjective_log_results_file, 'r') as f:
    multiobjective_log_results = json.load(f)

all_labels = ["130", "134", "135", "136", "target"]   # the ordering is important
mo_labels = ["130", "134", "135", "136"]
main_label = "target"
var_types = {"loss": "Training-loss", "ndcg": "Training-ndcg@30"}
variables = list(var_types.values())
combinators = ["epo_search", "linear_scalarization"]
markers = {"epo_search": "*", "linear_scalarization": "s"}
msz = {"epo_search": 5, "linear_scalarization": 2}
colors = {"epo_search": "g", "linear_scalarization": "r"}
snapshot_iteration = 800

latexify(fig_width=3.3, fig_height=1.8)
# Baseline stem plot
baseline_data = dict()
for label in all_labels:
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, variables)
    baseline_data[label] = dict()
    for vname, vtype in var_types.items():
        baseline_data[label][vname] = results[vtype].iloc[snapshot_iteration][0]

baseline_l = [baseline_data[label]["loss"] / len(all_labels) for label in all_labels]

markerline, stemlines, baseline = plt.stem(baseline_l, label='Baseline',
                                           use_line_collection=True,
                                           basefmt=' ')
plt.setp(stemlines, 'linewidth', 8, 'alpha', 0.5,
         'color', 'gray', 'zorder', -5)
plt.setp(markerline, 'ms', 8, 'color', 'gray', 'zorder', -5, 'marker', '_')

# Other combinators
for combinator in combinators:
    rls = []
    for dat in multiobjective_log_results[combinator]:
        r = np.asarray(dat["preferences"])
        logfilename = dat["logfile"]
        logfile = os.path.join(multiobjective_folder, logfilename)
        results = log2dataframe(logfile, variables)
        combinator_l = results[var_types["loss"]].iloc[snapshot_iteration].to_numpy()
        # print(r)
        # print(r.sum())
        r = r / r.sum()
        # r *= m
        rls.append(r * combinator_l)
    rls = np.stack(rls)
    df = pd.DataFrame(rls)
    low, high = 0.05, .95
    quant = df.quantile([low, high])
    rl_mean, rl_std = [], []
    for j in range(len(all_labels)):
        rljs = [rlj_ for rlj_ in rls[:, j]
                if rlj_ > quant.loc[low, j] and rlj_ < quant.loc[high, j]]
        rl_mean.append(np.mean(rljs))
        rl_std.append(np.std(rljs))
    # rl_mean = rls.mean(axis=0)
    # rl_std = rls.std(axis=0)
    # print(combinator)
    # print(f"stds={rl_std}")
    plt.plot(rl_mean, c=colors[combinator], lw=0.5,
             marker=markers[combinator], ms=msz[combinator], label=combinator)
    plt.errorbar(range(len(rl_mean)), rl_mean, elinewidth=0.9,
                 yerr=rl_std, fmt=' ', color=colors[combinator])

plt.xlabel('Labels')
plt.ylabel(r'$r \odot f$')
plt.title(f'iteration {snapshot_iteration}')
# plt.ylim([-0.1, 5])
plt.xticks(ticks=range(len(all_labels)), labels=all_labels)
# plt.text(-.5, 7, r'$\times 10^3$', fontsize=6)
ax = plt.gca()
ax.xaxis.set_label_coords(1.07, -0.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()

plt.savefig(f'figures/multi_obj_{snapshot_iteration}.pdf')
plt.show()
