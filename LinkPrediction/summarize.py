import numpy as np
import pandas as pd


ds_list = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI']
TAU = 6

for ds in ds_list:
    for seed in [0, 1, 2]:
        file_path = f'results_hptune/{ds}/TGCL_tau{TAU}/{seed}/result.txt'
        with open(file_path, 'r') as f:
            if seed == 0:
                content = pd.read_csv(f, sep=' ', header=None)
            else:
                content = pd.concat([content, pd.read_csv(f, sep=' ', header=None)], ignore_index=True)

    pdsla_mean = []
    pdsla_std = []
    pdsla_ckpt = []

    pdsla_mean.append(np.round(content[2].values.mean() * 100, decimals=2))
    pdsla_std.append(np.round(content[2].values.std() * 100, decimals=2))

    df = pd.DataFrame()
    df['Dataset'] = [ds]
    df['Method'] = [f'pDSLA-tau{TAU}']
    df['Mean'] = pdsla_mean
    df['Std'] = pdsla_std

    print(df)

0