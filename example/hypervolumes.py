import os
import json
import numpy as np
from utils import log2dataframe
import pickle as pkl
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt
import seaborn as sns
from latex_utils import latexify


baseline_folder = "baseline_results"
multiobjective_folder = "multiobjective_results"
baseline_log_results_file = os.path.join(baseline_folder, "log_results.json")
multiobjective_log_results_file = os.path.join(multiobjective_folder, "log_results.pkl")
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
with open(multiobjective_log_results_file, 'rb') as f:
    multiobjective_log_results = pkl.load(f)

var_types = {"loss": "Training-loss", "ndcg": "Training-ndcg@5"}
variables = list(var_types.values())

all_labels = ["132", "133", "134", "136", "target"]
mg_combinators = ["linear_scalarization", "chebyshev_scalarization", "epo_search"]

# Get baseline costs
baseline_costs = dict()
for i, label in enumerate(all_labels):
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, ["Training-loss"])["Training-loss"]
    results.columns = all_labels
    baseline_costs[label] = results

data = dict()
pref_data = dict()
for combinator in mg_combinators:
    data[combinator] = {'loss': [], 'ndcg': []}
    pref_data[combinator] = []
    for dat in multiobjective_log_results[combinator]:
        pref_data[combinator].append(np.asarray(dat["preferences"]))
        logfilename = dat["logfile"]
        logfile = os.path.join(multiobjective_folder, logfilename)
        results = log2dataframe(logfile, variables)
        for vtype, vname in var_types.items():
            results[vname].columns = all_labels
            data[combinator][vtype].append(results[vname])

hv_values = {comb: [] for comb in mg_combinators}
cs_values = {comb: [] for comb in mg_combinators}
cs_dfs = []
comb_legends = {"epo_search": "EPO-Search",
               "linear_scalarization": "LS",
               "wc_mgda": "WC-MGDA",
               "chebyshev_scalarization": "CS",
               "stochastic_label_aggregation": "SLA"}
for i in range(601):
    for comb in mg_combinators:
        points = [data[comb]['loss'][j].iloc[i].to_numpy() for j in range(20)]
        rel_points = [pt * pref_data[comb][i] for i, pt in enumerate(points)]
        hv = pg.hypervolume(points)
        ref = np.max(np.stack(points), axis=0) + 1e-10
#         print(np.stack(points))
#         print("ndr\n", ndr)

        hv_values[comb].append(hv.compute(ref))
        df = pd.DataFrame(np.max(np.stack(rel_points), axis=1))
        df.columns = ['$\max_k\ r_kl_k$']
        df['iteration'] = i
        df['Combinator'] = comb_legends[comb]
        cs_dfs.append(df)
        cs_values[comb].append(np.max(np.stack(rel_points), axis=1))

latexify(fig_width=4, fig_height=1.8)
fig,axs = plt.subplots(1,2)
for comb in mg_combinators:
    axs[0].plot(range(100,601), hv_values[comb][100:], label=comb_legends[comb], lw=0.8)
axs[0].set_ylabel('hypervolume')
axs[0].set_xlabel('iteration')
axs[0].legend(fontsize=7)

cs_data = pd.concat(cs_dfs, ignore_index=True)

sns.lineplot(x="iteration", y="$\max_k\ r_kl_k$", hue="Combinator",
             data=cs_data[cs_data['iteration']>100], legend=False, ax=axs[1], lw=0.8)

for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('figures/multi-obj_hv-mrov.pdf')