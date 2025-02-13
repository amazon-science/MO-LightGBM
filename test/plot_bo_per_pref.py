import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
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
with open(biobjective_log_results_file, 'rb') as f:
    biobjective_log_results = pkl.load(f)
# all_labels = ["130", "132", "133", "134", "135", "136", "target"]
# bilabels = [(mo_label, target) for i, mo_label in enumerate(all_labels[:-1]) for target in all_labels[i+1:]]
all_labels = ["132", "133", "134", "136"]
bilabels = [('132', '133'), ('134', '136')]
num_prefs = 5
all_variables = ["Training-loss", "Training-ndcg@30"]
to_extract = {"loss": "Training-loss", "ndcg@30": "Training-ndcg@30"}
combinators = ["epo_search", "linear_scalarization"]

# data_filename = "iterations_data.pkl"
# if os.path.exists(data_filename):
#     print("Using data from ", data_filename)
#     data = pkl.load(open(data_filename, 'rb'))
# else:
#     dfs = []    # list of dataframes, where each has 4 columns, objective, ndcg, label, data
#
# extract for baseline
# sodata = {label: {'loss': None, 'ndcg@30': None} for label in all_labels}
# for label in all_labels:
#     logfile = os.path.join(baseline_folder, baseline_log_results[label])
#     results = log2dataframe(logfile, all_variables)
#     for var in to_extract:
#         for data in to_extract[var]:
#             varname = to_extract[var][data]
#             if varname not in results:
#                 continue
#             results[varname].columns = [label]
#             results[varname]['iterations'] = range(len(results[varname]))
#             results[varname]['data'] = data
#             sodata[label][var] = results[varname]

bidata = {label_pair: [] for label_pair in bilabels}
# Extract for bi-objective
for label_pair in bilabels:
    mo_label, target = label_pair
    for pref_data in biobjective_log_results[label_pair]:
        res = {'ndcg@30': [], 'loss': [], 'r': None}
        for combinator in combinators:
            r = np.asarray(pref_data["preferences"], dtype=float)
            r /= r.sum()
            res['r'] = r
            logfilename = pref_data[combinator]
            logfile = os.path.join(biobjective_folder, logfilename)
            results = log2dataframe(logfile, all_variables)
            for var in to_extract:
                varname = to_extract[var]
                if varname not in results:
                    continue
                results[varname].columns = [mo_label, target]
                results[varname] = results[varname].mul(r, axis=1)
                results[varname]['iterations'] = range(len(results[varname]))
                results[varname]['combinator'] = combinator
                dfs = []
                for lbl in [mo_label, target]:
                    df = results[varname][['iterations', lbl, 'combinator']]
                    df.columns = ['iterations', 'relative_loss', 'combinator']
                    df['label'] = lbl
                    dfs.append(df)
                res[var].append(pd.concat(dfs, ignore_index=True))
        for var in to_extract:
            res[var] = pd.concat(res[var], ignore_index=True)
        bidata[label_pair].append(res)

colors = sns.color_palette("husl", len(all_labels))
start = 0
max_iters = [10, 20, 40, 80, 200, 300, 400, 500, 600]
step = 1
for label_pair in bilabels:
    mo_label, target = label_pair
    pdf = backend_pdf.PdfPages(f'figures/{mo_label}_vs_{target}.pdf')
    for max_iter in max_iters:
        fig, axs = plt.subplots(len(to_extract), len(bidata[label_pair]), figsize=(2.5*len(bidata[label_pair]), num_prefs))
        for i, var in enumerate(to_extract):
            for j, res in enumerate(bidata[label_pair]):
                df = res[var]
                df = df[df.iterations.isin(range(max(max_iter-100, 0), max_iter, step))]
                sns.lineplot(x='iterations', y='relative_loss', hue='label', style='combinator',
                             # err_kws={'visible': True, 'alpha': 0.05},
                             data=df, ax=axs[i][j], legend=True if i == 0 and j == len(bidata[label_pair]) - 1 else False)
                if i == 0 and j == len(bidata[label_pair]) - 1:
                    axs[i][j].legend(loc=(1., 0))
                axs[i][j].set_title(np.array2string(res['r'], precision=3, separator=',', suppress_small=True))
        fig.suptitle(f'{mo_label} vs {target}; {max_iter} iterations')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()