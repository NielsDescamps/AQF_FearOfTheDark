import pandas as pd
import os

def load_data(save_dir, exp_date, type):
    if type == 0:
        type_prod = 'call'
    elif type == 1:
        type_prod = 'put'
    ticker = 'SEDG'
    filename = "../" + save_dir + '/' + ticker + '_' + type_prod + '_chain_' + exp_date + ".csv"
    absolute_path = os.path.abspath(filename)
    print(absolute_path)
    data = pd.read_csv(absolute_path)
    out = data
    return out