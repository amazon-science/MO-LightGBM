import os
from copy import deepcopy
from utils import config2dict, dict2config, log2dataframe, sample_inverse_preferences
import json
import pickle as pkl
import numpy as np
import itertools

# Relevant paths
cwd = os.getcwd()
outfolder = os.path.join(cwd, "multiobjective_results")
runscript = os.path.join(outfolder, "ec_run_experiment.sh")
lightgbm = os.path.join(cwd, os.pardir, "LightGBM", "lightgbm")
log_results = os.path.join(outfolder, "ec_log_results.pkl")
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
params["mo_preferences"] = ",".join(['1']*len(all_labels))

mg_combinators = ["e_constraint",
                  # "ec_mgda"
                  ]
logs = {combinator: [] for combinator in mg_combinators}

# --------- Get Baseline costs for all labels ---------------
baseline_folder = "baseline_results"
baseline_log_results_file = os.path.join(baseline_folder, "log_results.json")
with open(baseline_log_results_file, 'r') as f:
    baseline_log_results = json.load(f)

all_config = list(itertools.product(*([['l', 'm', 't']]*4)))  # l: loose bound, m: mid bound, t: tight bound
ts = np.array([len([1 for c in config if c == 't']) for config in all_config])
ms = np.array([len([1 for c in config if c == 'm']) for config in all_config])
# should have at max one tightest bound
# should not have a tightest bound and a mid-bound
allowed = (ts <= 1) * np.logical_not((ms > 0) * (ts > 0))   # sum(allowed) = 20

last_iter_baseline = 600
baseline_costs = []
for i, label in enumerate(all_labels):
    logfile = os.path.join(baseline_folder, baseline_log_results[label])
    results = log2dataframe(logfile, ["Training-loss"])["Training-loss"]
    results.columns = all_labels
    baseline_costs.append(results.iloc[last_iter_baseline].to_numpy())
baseline_costs = np.stack(baseline_costs)
upperbounds = [np.linspace(label_costs.max(), label_costs.min(), 3) for label_costs in baseline_costs.T]
upperbounds = np.asarray(list(itertools.product(*upperbounds[:-1])))[allowed]
num_obj = len(all_labels)
num_prefs = 20
inv_preferences = np.stack(sample_inverse_preferences(baseline_costs, num_prefs))
preferences = 1.0 / inv_preferences

scriptlines = []
for ub in upperbounds:
    params["mo_ub_sec_obj"] = ",".join([str(e) for e in ub])
    for combinator in mg_combinators:
        params["mg_combination"] = combinator
        configfilename = combinator + "_" + "-".join([str(e) for e in ub]) + ".conf"
        configfile = os.path.join(outfolder, configfilename)
        dict2config(params, configfile)

        logfile = combinator + "_" + "-".join([str(e) for e in ub]) + ".log"
        logs[combinator].append({"preferences": ub,
                                 "logfile": logfile})

        scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logfile
        scriptlines.append(scriptline)

with open(runscript, 'w') as f:
    f.write('\n'.join(scriptlines))

with open(log_results, 'wb') as f:
    pkl.dump(logs, f)
