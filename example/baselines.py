import os
from utils import config2dict, dict2config
import yaml
import sys

configfile = sys.argv[1]
with open(configfile, 'r') as f:
    config = yaml.safe_load(f)
dataset = config["dataset"]

# Relevant paths
config_root = os.path.dirname(os.path.abspath(configfile))
objective_type = config["lightgbm_parameters"]["objective_type"]
objective_type_root = os.path.join(config_root
                                   , f"results_{dataset['name']}"
                                   , f"{objective_type}"
                                   )

outfolder = os.path.join(objective_type_root, f"baseline_results")
if not os.path.isdir(outfolder):
    os.makedirs(outfolder)

lightgbm = os.path.join(config_root, config['lightgbm_path'])

all_labels = dataset["all_labels"]
ignore_columns = dataset["ignore_columns"]

params = config2dict(config['sample_lightgbm_config'])
params["train_data_file"] = os.path.join(config_root, dataset['train_file'])
params["valid_data_file"] = os.path.join(config_root, dataset['valid_file'])
#params["test_data_file"] = os.path.join(config_root, dataset['test_file'])

infile = open(params["train_data_file"], 'r')
cols = infile.readline().strip().split("\t")

params["group_column"] = "name:" + dataset['query_column']
params["label_column"] = "name:" + all_labels[-1]
params["mo_labels"] = "name:" + ",".join(all_labels[:-1])
params["ignore_column"] = "name:" + ",".join(list(set(ignore_columns).intersection(set(cols)) - {all_labels[-1]}))

params["mg_combination"] = 'stochastic_label_aggregation'
for key, value in config['lightgbm_parameters'].items():
    params[key] = value


logs = {label: label + ".log" for label in all_labels}
# logs['label_order'] = all_labels
scriptlines = []
for i, label in enumerate(all_labels):
    prefs = ["0"]*len(all_labels)
    prefs[i] = "1"
    params["mo_preferences"] = ','.join(prefs)
    lightgbm_config = label + ".conf"
    dict2config(params, os.path.join(outfolder, lightgbm_config))
    scriptline = lightgbm + " config=" + lightgbm_config + " 2>&1 | tee " + logs[label]
    scriptlines.append(scriptline)

runscript = os.path.join(outfolder, "run_experiment.sh")
with open(runscript, 'w') as f:
    f.write('\n'.join(scriptlines))

log_results = os.path.join(outfolder, "log_results.yml")
with open(log_results, 'w') as f:
    yaml.dump(logs, f)
