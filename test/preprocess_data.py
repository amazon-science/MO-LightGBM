import pandas as pd
import os

"""
- QualityScore : 132       => 255 - value
- QualityScore2 : 133      => 255 - value
- PageRank : 130
- Query-url click count : 134
- url click count : 135
- url dwell time : 136     
"""

#input_datafolder = "../LTRdatasets/MSLR-WEB10K"
input_datafolder = "../LTRdatasets/MSLR-WEB30K"
datafiles = ["train.tsv", "valid.tsv", "test.tsv"]
#output_datafolder = "datasets"
output_datafolder = "datasets-30K"

for datafile in datafiles:
    in_filepath = os.path.join(input_datafolder, datafile)
    df = pd.read_csv(in_filepath, sep='\t')
    df['132'] = 255 - df['132']
    df['133'] = 255 - df['133']
    out_filepath = os.path.join(output_datafolder, 'inverted_qs_' + datafile)
    df.to_csv(out_filepath, sep='\t', index=False)

