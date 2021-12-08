import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    # specify traingdata path below (update this according to where
    # you saved the training data)
    training_data = Path('data/training_trajectories')
    model_data = Path(__file__).parent

    # read in system names
    with open(training_data / "systems", "r") as f:
        systems = f.read().splitlines()

    # train for each system separately
    for system in systems:
        # Only apply PID controller to bumblebee systems (otherwise use zero)
        if 'bumblebee' in system:
            csvs = training_data.glob(f"{system}_*.csv")
            positions = []
            controls = []
            states = []
            PP = []
            II = []
            DD = []
            # collect training data
            for csv in csvs:
                df = pd.read_csv(csv)
                # read out position of tip
                pos = df[["X", "Y"]].values
                # change in position
                diffpos = pos[1:, :] - pos[:-1, :]
                # D term
                DD.append((diffpos[1:, :] - diffpos[:-1, :])*200)
                # I term
                II.append(np.cumsum(diffpos, axis=0)[1:, :]/200)
                # P term
                PP.append(diffpos[1:, :])
                # save positions
                positions.append(pos[1:-1, :])
                # corresponding controls
                controls.append(df[[k
                                    for k in df.columns
                                    if k.startswith('U')]].values[1:-1, :])
                # corresponding states
                states.append(df[
                    [k
                     for k in df.columns
                     if not k.startswith('U')
                     and k not in ['ID', 'System', 't']]].values[1:-1, :])
            # aggregate features
            PP = np.vstack(PP)
            II = np.vstack(II)
            DD = np.vstack(DD)
            positions = np.vstack(positions)
            controls = np.vstack(controls)
            states = np.vstack(states)
            features = np.c_[PP, II, DD]

            # fit linear model
            lr_model = LinearRegression(fit_intercept=False).fit(features,
                                                                 controls)

            # save data
            joblib.dump(lr_model,
                        model_data / f"{system}.joblib",
                        compress=5)
