import os
from glob import glob
import shutil
import pandas as pd
import ipdb
import json

main_table_files = glob('../outputs/*/plms/*/main_table*')
all_main_table_results=[]

for main_table_path in main_table_files:

    table = pd.read_csv(main_table_path)

    file_dir = '/'.join(main_table_path.split('/')[:-1])
    subset_file = glob(file_dir + '/subset_config*')[0]
    subset_config = json.load(open(subset_file, 'r'))

    for key in subset_config.keys():
        table[key] = subset_config[key]

    table['path'] = main_table_path
    all_main_table_results.append(table)

pd.concat(all_main_table_results).to_csv('all_main_tables.tsv')