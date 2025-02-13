import os
from copy import deepcopy
from utils import config2dict, dict2config
import json

# Relevant paths
cwd = os.getcwd()
outfolder = os.path.join(cwd, "baseline_results")
runscript = os.path.join(outfolder, "run_experiment.sh")
lightgbm = os.path.join(cwd, os.pardir, "LightGBM", "lightgbm")
log_results = os.path.join(outfolder, "log_results.json")
sampleconfig = os.path.join(cwd, "sample_config.conf")
trainfile = os.path.join(cwd, "datasets", "inverted_qs_train.tsv")
validfile = os.path.join(cwd, "datasets", "inverted_qs_valid.tsv")

# Read sample config and make changes for single objective ranking optimization
params = config2dict(sampleconfig)
params["train_data_file"] = trainfile
params["valid_data_file"] = validfile
# params["num_iterations"] = "1500"

# all_labels = ["130", "132", "133", "134", "135", "136", "target"]
all_labels = ["132", "133", "134", "136", "target"]   # ordering is important
logs = {label: label + ".log" for label in all_labels}
# logs['label_order'] = all_labels

params["label_column"] = "name:" + all_labels[-1]
params["mo_labels"] = "name:" + ",".join(all_labels[:-1])
params["mg_combination"] = 'stochastic_label_aggregation'
params["ignore_column"] = "name:" + ",".join(all_labels[:-1])
scriptlines = []
for i, label in enumerate(all_labels):
    prefs = ["0"]*len(all_labels)
    prefs[i] = "1"
    params["mo_preferences"] = ','.join(prefs)
    configfilename = label + ".conf"
    configfile = os.path.join(outfolder, configfilename)
    dict2config(params, configfile)
    scriptline = lightgbm + " config=" + configfilename + " 2>&1 | tee " + logs[label]
    scriptlines.append(scriptline)

with open(runscript, 'w') as f:
    f.write('\n'.join(scriptlines))

with open(log_results, 'w') as f:
    json.dump(logs, f)
