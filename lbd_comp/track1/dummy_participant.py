import pandas as pd
from zipfile import ZipFile
import numpy as np

if __name__ == "__main__":
    # read submission template from training.zip
    with ZipFile('data/track1/starting_kit_track1.zip', 'r') as zf:
        df = pd.read_csv(zf.open('submission_template.csv'))

    # they need to fill in the U input
    for k, row in df.iterrows():
        # initial values
        X0 = row[[f'X{k}' for k in range(1, 15)]].values
        target = row['target']

        # some function they've learnt to determine the desired input
        # for example

        df.loc[k, [f'U{k}' for k in range(1, 9)]] = np.abs(
            (X0.mean() - target) * X0)[:8]

    # for evaluation, is round to 6 digits anyways, so we round here
    df.to_csv('data/track1/dummy_submission.csv',
              float_format='%.6f',
              encoding='utf8',
              index=False)
