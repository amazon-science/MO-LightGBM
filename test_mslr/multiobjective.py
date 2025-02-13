import os
from copy import deepcopy
from utils import config2dict, dict2config, log2dataframe, sample_inverse_preferences
import json
import pickle as pkl
import numpy as np

# Relevant paths
cwd = os.getcwd()
outfolder = os.path.join(cwd, "multiobjective_results")
runscript = os.path.join(outfolder, "run_experiment.sh")
lightgbm = os.path.join(cwd, os.pardir, "LightGBM", "lightgbm")
log_results = os.path.join(outfolder, "log_results.pkl")
sampleconfig = os.path.join(cwd, "sample_config.conf")
trainfile = os.path.join(cwd, "datasets", "inverted_qs_train.tsv")
validfile = os.path.join(cwd, "datasets", "inverted_qs_valid.tsv")

# Read sample config and make changes for single objective ranking optimization
params = config2dict(sampleconfig)
params["train_data_file"] = trainfile
params["valid_data_file"] = validfile

all_labels = ["132", "133", "134", "136", "target"]
features = {"132", "133", "134", "136"}

params["label_column"] = "name:" + all_labels[-1]
params["mo_labels"] = "name:" + ",".join(all_labels[:-1])
params["ignore_column"] = "name:" + ",".join(list(features))

mg_combinators = ["linear_scalarization", "chebychev_scalarization", 
                  "epo_search", 
                  "stochastic_label_aggregation"
                    ]
logs = {combinator: [] for combinator in mg_combinators}

# --------- Get Baseline costs for all labels ---------------
baseline_folder = "baseline_results"
baseline_log_results_file = os.path.join(baseline_folder, "log_results.json")
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)
# Get baseline costs
last_iter_baseline = 600
baseline_costs = dict()
for i, label in enumerate(all_labels):
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, ["Training-loss"])["Training-loss"]
    results.columns = all_labels
    baseline_costs[label] = results.iloc[-1].to_numpy()
baseline_costs = np.stack([baseline_costs[label] for label in all_labels])

num_obj = len(all_labels)
num_prefs = 20
inv_preferences = np.stack(sample_inverse_preferences(baseline_costs, num_prefs))
preferences = 1.0 / inv_preferences

scriptlines = []
for preference in preferences:
    params["mo_preferences"] = ",".join([str(r) for r in preference])
    for combinator in mg_combinators:
        params["mg_combination"] = combinator
        configfilename = combinator + "_" + "-".join([str(r) for r in preference]) + ".conf"
        configfile = os.path.join(outfolder, configfilename)
        dict2config(params, configfile)

        logfile = combinator + "_" + "-".join([str(r) for r in preference]) + ".log"
        logs[combinator].append({"preferences": preference,
                                 "logfile": logfile})

        scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logfile
        scriptlines.append(scriptline)

with open(runscript, 'w') as f:
    f.write('\n'.join(scriptlines))

with open(log_results, 'wb') as f:
    pkl.dump(logs, f)
