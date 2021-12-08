import numpy as np
from track1.systems import systems
import zipfile


if __name__ == "__main__":
    # iterate over all systems
    for sysid, system in enumerate(systems):
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

            with zipfile.ZipFile('data/track1/CHEM_trainingdata.zip',
                                 mode='a',
                                 compression=zipfile.ZIP_DEFLATED
                                 ) as zfile:
                zfile.writestr(
                    f'{S.name}_instance_{repetition:02d}.csv',
                    S.getDF().to_csv(None,
                                     float_format='%.6f',
                                     encoding='utf8',
                                     index=False))
