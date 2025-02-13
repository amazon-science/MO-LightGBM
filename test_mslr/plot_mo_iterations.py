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
with open(multiobjective_log_results_file, 'r') as f:
    multiobjective_log_results = json.load(f)
all_labels = ["130", "134", "135", "136", "target"]  # the ordering is important
mo_labels = ["130", "134", "135", "136"]
main_label = "target"
all_variables = ["Training-loss", "Validation-loss",
                 "Training-ndcg@5", "Validation-ndcg@5",
                 # "Training-ndcg@30", "Validation-ndcg@30"
                 ]
to_extract = {"loss": {"train": "Training-loss", "valid": "Validation-loss"},
              "ndcg@5": {'train': "Training-ndcg@5", 'valid': "Validation-ndcg@5"},
              # "ndcg@30": {'train': "Training-ndcg@30", 'valid': "Validation-ndcg@30"}
              }
combinators = ["linear_scalarization", "epo_search"]

# data_filename = "iterations_data.pkl"
# if os.path.exists(data_filename):
#     print("Using data from ", data_filename)
#     data = pkl.load(open(data_filename, 'rb'))
# else:
#     dfs = []    # list of dataframes, where each has 4 columns, objective, ndcg, label, data
#
#     # extract for baseline
#     for label in all_labels:
#         logfile = os.path.join(baseline_folder, baseline_log_results[label])
#         results = log2dataframe(logfile, all_variables)
#         dfs += extract4label(results, label, 0)

# Extract for multi-objective
modata = {'loss': [],
          'ndcg@5': [],
          # 'ndcg@30': []
          }

r_ndcg = {'preference': {comb: [] for comb in combinators},
          'ndcg@5': {comb: [] for comb in combinators},
          # 'ndcg@30': {comb: [] for comb in combinators}
          }
ndcg_snapshot_at = 100

for combinator in combinators:
    for dat in multiobjective_log_results[combinator]:
        r = np.asarray(dat["preferences"], dtype=float)
        r /= r.sum()
        logfilename = dat["logfile"]
        logfile = os.path.join(multiobjective_folder, logfilename)
        results = log2dataframe(logfile, all_variables)
        for var in to_extract:
            for data in to_extract[var]:
                varname = to_extract[var][data]
                if varname not in results:
                    continue
                if 'ndcg' in varname:
                    r_ndcg['preference'][combinator].append(r)
                    r_ndcg[var][combinator].append(results[varname].iloc[ndcg_snapshot_at].to_numpy())
                results[varname].columns = all_labels
                # if 'loss' in varname:
                #     results[varname] = results[varname].mul(r, axis=1)
                # else:
                #     print("multiplying 1 - r")
                #     results[varname] = results[varname].mul(1 - r, axis=1)
                results[varname]['iterations'] = range(len(results[varname]))
                results[varname]['combinator'] = combinator
                results[varname]['data'] = data
                modata[var].append(results[varname])

for var in modata:
    modata[var] = pd.concat(modata[var], ignore_index=True)
    print(f'{var}:', modata[var].shape)

# for comb in r_ndcg['ndcg@30']:
#     r_ndcg['ndcg@30'][comb] = np.stack(r_ndcg['ndcg@30'][comb])
#     r_ndcg['preference'][comb] = np.stack(r_ndcg['preference'][comb])
#
# colors = sns.color_palette("husl", len(all_labels))
# r_ndcg_corr_data = {label: [] for label in all_labels}
# for i, label in enumerate(all_labels):
#     for comb in r_ndcg['ndcg@30']:
#         ndcg = r_ndcg['ndcg@30'][comb][:, i]
#         ri = r_ndcg['preference'][comb][:, i]
#         df = pd.DataFrame({'preference': ri, 'ndcg@30': ndcg})
#         df['combinator'] = comb
#         r_ndcg_corr_data[label].append(df)
#     r_ndcg_corr_data[label] = pd.concat(r_ndcg_corr_data[label], ignore_index=True)
#     sns.lmplot(x='preference', y="ndcg@30", hue='combinator',
#                data=r_ndcg_corr_data[label])
#     plt.title(label)

objtype = 'lambdarank'
start = 1
max_iters = 800
step = 30
for var in modata:
    fig, axs = plt.subplots(1, len(all_labels), figsize=(len(all_labels) * 3, 3))
    for i, label in enumerate(all_labels):
        df = modata[var]
        df = df[df.iterations.isin(range(start, max_iters, step))]
        print("shpae of plotted data:", df.shape)
        g = sns.lineplot(x='iterations', y=all_labels[i], hue='combinator', style='data',
                         err_kws={'visible': True, 'alpha': 0.05},
                         data=df, ax=axs[i], legend=True if i == 0 and var == 'loss' else False)
        if i == 0:
            axs[i].legend(loc=(-1.3, 0))
        # axs[i].set_title('{}_{}'.format(var, i+1))
    fig.suptitle('{}; iteration {}'.format(var, max_iters))
    fig.tight_layout()
    fig.savefig(f'figures/{var}_{start}_{max_iters}.pdf')

plt.show()