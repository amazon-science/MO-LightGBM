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
all_labels = ["130", "134", "135", "136", "target"]   # the ordering is important
mo_labels = ["130", "134", "135", "136"]
main_label = "target"
all_variables = ["Training-ndcg@5", "Training-ndcg@30", "Training-loss",
                 "Validation-ndcg@5", "Validation-ndcg@30", "Validation-loss"]
to_extract = {"train": {"Training-ndcg@5": 'ndcg@5', "Training-ndcg@30": 'ndcg@30', "Training-loss": 'loss'},
              "valid": {"Validation-ndcg@5": 'ndcg@5', "Validation-ndcg@30": 'ndcg@30', "Validation-loss": 'loss'}}


def extract4label(results, label, label_idx):
    label_dfs = []
    for data in to_extract:
        df = pd.DataFrame()
        for var in to_extract[data]:
            df[to_extract[data][var]] = results[var].to_numpy()[:, label_idx]
        df["data"] = data
        df["label"] = label
        label_dfs.append(df)
    return label_dfs

data_filename = "obj_ndcg.pkl"
if os.path.exists(data_filename):
    print("Using data from ", data_filename)
    data = pkl.load(open(data_filename, 'rb'))
else:
    dfs = []    # list of dataframes, where each has 4 columns, objective, ndcg, label, data

    # extract for baseline
    for label in all_labels:
        logfile = os.path.join(baseline_folder, baseline_log_results[label])
        results = log2dataframe(logfile, all_variables)
        dfs += extract4label(results, label, 0)

    # Extract for bi-objective
    combinators = ["epo_search", "linear_scalarization"]
    for label in mo_labels:
        for pref_data in biobjective_log_results[label]:
            for combinator in combinators:
                logfilename = pref_data[combinator]
                logfile = os.path.join(biobjective_folder, logfilename)
                results = log2dataframe(logfile, all_variables)
                dfs += extract4label(results, label, 0)
                dfs += extract4label(results, main_label, 1)

    # Extract for multi-objective
    for combinator in combinators:
        for dat in multiobjective_log_results[combinator]:
            logfilename = dat["logfile"]
            logfile = os.path.join(multiobjective_folder, logfilename)
            results = log2dataframe(logfile, all_variables)
            for label_idx, label in enumerate(all_labels):
                dfs += extract4label(results, label, label_idx)

    data = pd.concat(dfs, ignore_index=True)
    pkl.dump(data, open(data_filename, "wb"))

print(data.columns, data.shape)
colors = sns.color_palette("husl", len(all_labels))
objtype = 'lambdarank'
for ndcg in ['ndcg@5', 'ndcg@30']:
    # plt.figure()
    totalcorr = data[[ndcg, 'loss']].corr().iloc[0, 1]
    sns.lmplot(x=ndcg, y="loss", data=data, scatter_kws={"s": 1, "color": 'k'})
    plt.title(f'all data: {len(data)} points;' + r' $\rho$ ' + '= {:.4f}'.format(totalcorr))
    plt.tight_layout()
    plt.savefig(f'figures/{ndcg}_{objtype}-loss.pdf')

    label2corr = data[['loss', 'ndcg@30', 'label']].groupby('label').corr().unstack().iloc[:, 1]
    for i, label in enumerate(all_labels):
        # plt.figure()
        label_data = data[data['label'] == label]
        sns.lmplot(x=ndcg, y="loss", data=label_data, scatter_kws={"s": 1, "color": colors[i]})
        plt.title(f'{label} label: {len(label_data)} points;' + r' $\rho$ ' + '= {:.4f}'.format(label2corr[label]))
        plt.tight_layout()
        plt.savefig(f'figures/{label}_{ndcg}_{objtype}-loss.pdf')

plt.show()
