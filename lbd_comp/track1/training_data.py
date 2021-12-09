import numpy as np
import zipfile
from pathlib import Path
from tqdm import tqdm

from track1.systems import systems


def generate_training_data(output_dir, output_as_zip=False, force_output=False):

    out_dir_as_path = Path(output_dir)
    out_dir_as_path.mkdir(
        parents=True,
        exist_ok=force_output,
    )

    # iterate over all systems
    for sysid, system in enumerate(tqdm(systems)):
        kwargs = {
            'parameters': {'noise_sigma': .1},
            'target': np.nan,
        }
        kwargs['parameters'].update(system['parameters'])

        for repetition in range(20):
            np.random.seed(sysid * 199933 + repetition * 993319)
            seed = np.random.randint(2**32 - 1)
            # Generate initial value and impulse controls
            # train initial conditions
            corind = system['control_args']['corind']
            if corind[0] == 0:
                # independent controls and initials
                X0 = np.random.uniform(0, 1, 15)**2
                u = np.random.normal(0, 2, 8)
            elif corind[0] == 1:
                # correlated controls and initials
                AA = system['control_args']['AA']
                # Compute variances
                totvars = np.repeat((1, 2), [15, 8])
                aavars = np.diag(AA.dot(AA.T))
                eps_sd = np.sqrt(totvars - aavars)
                # Generate initial values
                H = np.random.normal(0, 1, 10).reshape(-1, 1)
                eps = (np.random.normal(0, 1, 23) * eps_sd).reshape(-1, 1)
                X0 = (AA.dot(H)[:15] + eps[:15]).reshape(-1)**2
                u = (AA.dot(H)[15:] + eps[15:]).reshape(-1)
            # instantiate system
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0=X0)
            # impulse control
            S.impulsecontrol(u=u)

            # output instance
            dataframe = S.getDF()
            csv_filename = f'{S.name}_instance_{repetition:02d}.csv'
            write_opts = {
                'float_format': '%.6f',
                'encoding': 'utf8',
                'index': False,
            }
            if output_as_zip:
                zip_path = out_dir_as_path / "training_data.zip"
                with zipfile.ZipFile(
                    zip_path,
                    mode='a',
                    compression=zipfile.ZIP_DEFLATED,
                ) as zfile:
                    zfile.writestr(
                        csv_filename,
                        dataframe.to_csv(
                            None,
                            **write_opts
                        )
                    )
            else:
                csv_path = out_dir_as_path / csv_filename
                dataframe.to_csv(
                    csv_path,
                    **write_opts
                )
