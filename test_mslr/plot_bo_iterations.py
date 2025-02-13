import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import os
from utils import log2dataframe
import pickle as pkl

baseline_folder = "baseline_results"
biobjective_folder = "biobjective_results"
multiobjective_folder = "multiobjective_results"
log_results_file = "log_results.json"
baseline_log_results_file = os.path.join(baseline_folder, log_results_file)
biobjective_log_results_file = os.path.join(biobjective_folder, log_results_file)
multiobjective_log_results_file = os.path.join(multiobjective_folder, log_results_file)
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
with open(biobjective_log_results_file, 'r') as f:
    biobjective_log_results = json.load(f)
# with open(multiobjective_log_results_file, 'r') as f:
#     multiobjective_log_results = json.load(f)
# all_labels = ["130", "134", "135", "136", "target"]  # the ordering is important
# mo_labels = ["130", "134", "135", "136"]
# main_label = "target"
all_labels = ["135", "134"]   # the ordering is important
mo_labels = ["135"]
main_label = "134"
all_variables = ["Training-loss", "Validation-loss",
                 # "Training-ndcg@5", "Validation-ndcg@5",
                 "Training-ndcg@30", "Validation-ndcg@30"
                 ]
to_extract = {"loss": {"train": "Training-loss", "valid": "Validation-loss"},
              # "ndcg@5": {'train': "Training-ndcg@5", 'valid': "Validation-ndcg@5"},
              "ndcg@30": {'train': "Training-ndcg@30", 'valid': "Validation-ndcg@30"}
              }
combinators = ["linear_scalarization", "epo_search"]

# data_filename = "iterations_data.pkl"
# if os.path.exists(data_filename):
#     print("Using data from ", data_filename)
#     data = pkl.load(open(data_filename, 'rb'))
# else:
#     dfs = []    # list of dataframes, where each has 4 columns, objective, ndcg, label, data
#
# extract for baseline
sodata = {label: {'loss': None, 'ndcg@30': None} for label in all_labels}
for label in all_labels:
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, all_variables)
    for var in to_extract:
        for data in to_extract[var]:
            varname = to_extract[var][data]
            if varname not in results:
                continue
            results[varname].columns = [label]
            results[varname]['iterations'] = range(len(results[varname]))
            results[varname]['data'] = data
            sodata[label][var] = results[varname]

bidata = {label: {'ndcg@30': [], 'loss': []} for label in mo_labels}
# Extract for bi-objective
for label in mo_labels:
    for pref_data in biobjective_log_results[label]:
        for combinator in combinators:
            r = np.asarray(pref_data["preferences"], dtype=float)
            r /= r.sum()
            logfilename = pref_data[combinator]
            logfile = os.path.join(biobjective_folder, logfilename)
            results = log2dataframe(logfile, all_variables)
            for var in to_extract:
                for data in to_extract[var]:
                    varname = to_extract[var][data]
                    if varname not in results:
                        continue
                    results[varname].columns = [label, main_label]
                    results[varname]['iterations'] = range(len(results[varname]))
                    results[varname]['combinator'] = combinator
                    results[varname]['data'] = data
                    bidata[label][var].append(results[varname])

for label in mo_labels:
    for var in bidata[label]:
        bidata[label][var] = pd.concat(bidata[label][var], ignore_index=True)

colors = sns.color_palette("husl", len(all_labels))
objtype = 'lambdarank'
start = 1
max_iters = 800
step = 10
for label in mo_labels:
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))
    for i, var in enumerate(bidata[label]):
        df = bidata[label][var]
        df = df[df.iterations.isin(range(start, max_iters, step))]
        print("shpae of plotted data:", df.shape)
        for j, lbl in enumerate([label, main_label]):
            sns.lineplot(x='iterations', y=lbl, hue='combinator', style='data',
                         err_kws={'visible': True, 'alpha': 0.05},
                         data=df, ax=axs[i][j], legend=False)
            # sns.lineplot(x='iterations', y=lbl, style='data', color='k',
            #              data=sodata[lbl][var], ax=axs[i][j], legend=False)
            # if i == 0:
            #     axs[i].legend(loc=(-1.3, 0))
            # axs[i][j].set_title(var)
    fig.suptitle(f'{label} vs {main_label}')
    fig.tight_layout()
    fig.savefig(f'figures/{label}_vs_{main_label}_{start}_{max_iters}.pdf')

# plt.show()