import numpy as np
import pandas as pd
from pathlib import Path
from track1.systems import systems, gset_evaluation_seed
import time
from tqdm import tqdm

num_instances = 50

finegrid = None


def get_finegrid(S):
    """ computes the fine timegrid, once since all S have the same timegrid """
    global finegrid
    if finegrid is None:
        finegrid = np.hstack([0] + [
            np.linspace(S.timegrid[k],
                        S.timegrid[k + 1],
                        S._resolution + 1)[1:]
            for k in range(S._timesteps - 1)])
    return finegrid


# J function
def J(S):
    # on fine system time scale
    grid = get_finegrid(S)
    # find index closest to second 40
    start = np.argmin(np.abs(grid - 40))
    deviation = 0
    for tsi in range(start, S._X.shape[1] - 1):
        deviation += (grid[tsi + 1] - grid[tsi]) * (S._X[0, tsi] - S.target)**2
    deviation /= (grid[-1] - grid[start])
    # on coarse grid, where participant submits input controls
    U = np.nan_to_num(S.U)
    # ||u||^2 / 8
    controlcost = np.nansum(np.einsum('ij,ij->j', U, U)) / 3 / 8
    return (np.sqrt(deviation), np.sqrt(controlcost) / 20)


def evaluate_track1(input_dir, output_dir=None, force_output=False, verbose=False):

    in_dir = Path(input_dir)
    assert in_dir.exists(), \
        f"Expected input_dir \"{in_dir}\" to exist, but it does not."
    infile = sorted(in_dir.glob('**/*.csv'))[0]

    if output_dir is None:
        output_results = False
    else:
        output_results = True
        out_dir = Path(output_dir)
        out_dir.mkdir(
            parents=True,
            exist_ok=force_output,
        )
    
    submission = pd.read_csv(infile).groupby('ID').first()

    if verbose:
        print(f"CHEM infile: {infile}")

    Js1 = np.inf * np.ones((len(systems), num_instances))
    Js2 = np.inf * np.ones((len(systems), num_instances))

    score_repetitions = range(num_instances)
    score_systems = [1, 3, 5, 7, 9, 11]

    # iterate over all systems
    # no matter which phase, we wish to ensure all IDs are there
    start = time.time()
    for sysid, system in enumerate(tqdm(systems)):
        kwargs = {
            'parameters': {'noise_sigma': .1},
        }
        kwargs['parameters'].update(system['parameters'])

        for repetition in range(num_instances):
            seed = gset_evaluation_seed(sysid, repetition)
            # test initial conditions
            corind = system['control_args']['corind']
            X0 = np.random.uniform(0, 1, 15)**2
            if corind[1] == 0 or corind[1] == 1:
                if corind[1] == 0:
                    # no shift independent control
                    u = np.random.normal(0, 2, 8)
                elif corind[1] == 1:
                    # increased variance indepdent control
                    val = np.random.uniform(3, 6, 1)[0]
                    sign = np.random.choice((-1, 1), 1,
                                            p=[0.5, 0.5])[0]
                    xx = sign * val
                    u = np.array([0, 0, 0, 0,
                                  xx, xx, -xx, -xx])
                S = system['system'](**kwargs, seed=seed, X0=X0)
                S.impulsecontrol(u)
                YT = S._X[0, -1]
                kwargs['target'] = np.round(YT, 6)
            else:
                # independently random target
                kwargs['target'] = np.round(2.5*np.random.rand(), 6)
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0=X0)

            if f"{hash(S)}!" in submission.index:
                u = submission.loc[
                    [f"{hash(S)}!"]][
                        [f'U{k + 1}' for k in range(S.d_U)]].values.ravel()
            else:
                u = None
                raise Exception(f'No u submission for {hash(S)}!')

            if sysid in score_systems and repetition in score_repetitions:
                S.impulsecontrol(u)
                # Compute score
                score_tmp = J(S)
                Js1[sysid, repetition] = score_tmp[0]
                Js2[sysid, repetition] = score_tmp[1]

    loss = Js1[score_systems, :][:, score_repetitions].mean() + \
        Js2[score_systems, :][:, score_repetitions].mean()

    duration = time.time() - start

    chem_eval_results = {
        "loss": loss,
        "duration": duration,
    }

    if verbose:
        print("Results:")
        print(f"Loss: {loss:.4f}")
        print(f"Duration (seconds): {duration:.2f}")

    if output_results:
        output_file_path = out_dir.joinpath("scores.txt")
        with open(output_file_path, "w") as text_file:
            text_file.write(f"Loss: {loss:.4f}\n")
            text_file.write(f"Duration: {duration:.2f}\n")
    
    return chem_eval_results
