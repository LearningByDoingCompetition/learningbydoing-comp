import numpy as np
from scipy.optimize import minimize
from track1.systems import systems, gset_evaluation_seed
from evaluate_track1 import get_finegrid, J
import zipfile
import time
num_instances = 50
oracle_submission = False
nelder_mead = False
finegrid = None

start = time.time()

if __name__ == "__main__":
    df = None
    oracledf = None
    oracledfExp = None

    # iterate over all systems
    for sysid, system in enumerate(systems):
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

            if df is None:
                df = S.getDF()
            else:
                df = df.append(S.getDF())

            odf = S.getDF()
            odf.loc[0, [f'U{k}' for k in range(1, 9)]] = np.zeros(8)
            odfExp = S.getDF()
            odfExp.loc[0, [f'U{k}' for k in range(1, 9)]] = np.zeros(8)
            if oracle_submission:
                def tar(u):
                    S = system['system'](**kwargs,
                                         name=system['name'],
                                         seed=seed,
                                         X0=X0)
                    S.impulsecontrol(u=u)
                    return J(S)

                def tars(u):
                    S = system['system'](**kwargs,
                                         name=system['name'],
                                         seed=seed,
                                         X0=X0)
                    S.impulsecontrol(u=u)
                    return sum(J(S))

                # compute oracle solution
                if nelder_mead:
                    results = minimize(tars,
                                       np.random.normal(size=8),
                                       method='nelder-mead')
                else:
                    results = minimize(tars,
                                       np.random.normal(size=8),
                                       method='powell',
                                       bounds=[(-5, 5)]*8)

                # compute oracle solution using only expensive controls
                # (U5, U6, U7, U8)
                def tarsExp(u):
                    S = system['system'](**kwargs,
                                         name=system['name'],
                                         seed=seed,
                                         X0=X0)
                    S.impulsecontrol(u=np.concatenate([np.zeros(4), u]))
                    return sum(J(S))

                resultsExp = minimize(tarsExp,
                                      np.random.normal(size=4),
                                      method='powell',
                                      bounds=[(-5, 5)]*4)

                odf.loc[0, [f'U{k}' for k in range(1, 9)]] = results.x
                odf = odf.assign(dev=tar(results.x)[0])
                odf = odf.assign(pen=tar(results.x)[1])

                odfExp.loc[0, [f'U{k}' for k in range(1, 9)]] = np.concatenate(
                    [np.zeros(4), resultsExp.x])
                odfExp = odfExp.assign(devExp=tar(
                    np.concatenate([np.zeros(4), resultsExp.x]))[0])
                odfExp = odfExp.assign(penExp=tar(
                    np.concatenate([np.zeros(4), resultsExp.x]))[1])

                print(sysid)
                print(repetition)
                print(time.time() - start)
            if oracledf is None:
                oracledf = odf
            else:
                oracledf = oracledf.append(odf)

            if oracledfExp is None:
                oracledfExp = odfExp
            else:
                oracledfExp = oracledfExp.append(odfExp)

    if oracle_submission:
        oracledf.to_csv('data/track1/oracle_submission.csv',
                        float_format='%.6f',
                        encoding='utf8',
                        index=False)
        oracledfExp.to_csv('data/track1/oracle_expensive_submission.csv',
                           float_format='%.6f',
                           encoding='utf8',
                           index=False)

    with zipfile.ZipFile('data/track1/CHEM_starter_kit.zip',
                         mode='a',
                         compression=zipfile.ZIP_DEFLATED
                         ) as zfile:
        zfile.writestr('submission_template.csv',
                       df.to_csv(None,
                                 index=False,
                                 encoding='utf8',
                                 float_format='%.6f'))
