import numpy as np
import pandas as pd
from pathlib import Path

from scipy.interpolate import interp1d

from lbd_comp.evaluate_track2 import lossfc
from track2.systems import systems
from track2.robot_control_baselines import RobotLQRController


def run_lqr_controller(system, scoring_targetvals, seed, target_traj):
    # LQR controller
    kwargs = {
        'parameters': {},
    }
    kwargs['parameters'].update(system['parameters'])
    X0 = {
        'mode': 'from_ws_point_no_vel',
        'ws_point': scoring_targetvals[:, 0],
    }
    S = system['system'](**kwargs,
                            seed=seed,
                            name=system['name'],
                            X0=X0)

    robot_ctrl = RobotLQRController(seed=seed,
                                    robot_sys=S)

    for t in range(S._timesteps - 1):
        current_obsv = S[:, t]
        reference_pose = np.r_[
            target_traj[:, t + 1],
            (target_traj[:, t + 2] - target_traj[:, t]) / 0.02]
        inp = robot_ctrl.get_input(current_obsv, reference_pose)
        S.step(u=inp)
    lqrloss = lossfc(scoring_targetvals, S)

    return lqrloss, S


def run_zero_controller(system, scoring_targetvals, seed):
    # zero controller
    kwargs = {
        'parameters': {},
    }
    kwargs['parameters'].update(system['parameters'])
    X0 = {
        'mode': 'from_ws_point_no_vel',
        'ws_point': scoring_targetvals[:, 0],
    }
    S = system['system'](**kwargs,
                            name=system['name'],
                            seed=seed,
                            X0=X0)

    for t in range(S._timesteps - 1):
        current_obsv = S[:, t]
        S.step(u=np.zeros(S.d_U))
    zeroloss = lossfc(scoring_targetvals, S)

    return zeroloss, S


def save_baseline_to_csv(system, repetition, out_dir_as_path, lqrloss, zeroloss):
    dataframe = pd.DataFrame([
        {'controller': 'lqr',
         'deviation': lqrloss[0],
         'controlcost': lqrloss[1],
        },
        {'controller': 'zero',
         'deviation': zeroloss[0],
         'controlcost': zeroloss[1],
        },
    ])
    dataframe.to_csv(
        out_dir_as_path / f"{system['name']}_{repetition:03d}.csv.baselines",
        float_format='%.8f',
        encoding='utf8',
        index=False,
    )


def save_animation(system, S, out_debug_dir_as_path, repetition, target_traj, lqrloss):
    animation_filename = f"{system['name']}_{repetition:03d}.gif"
    animation_path = Path(out_debug_dir_as_path) / animation_filename
    S.save_traj_animation(
        animation_path,
        target_traj[:2, :201].T,
        annot_text=f"Loss: {lqrloss[0]:.1f} {lqrloss[1]:.1f}",
    )


def generate_target_trajectories(output_dir, force_output=False, debug_output=False):

    out_dir_as_path = Path(output_dir)
    out_dir_as_path.mkdir(
        parents=True,
        exist_ok=force_output,
    )

    if debug_output:
        out_debug_dir_as_path = out_dir_as_path / "debug"
        out_debug_dir_as_path.mkdir(
            parents=True,
            exist_ok=force_output,
        )
        SAVE_ANIMATION = True
    else:
        SAVE_ANIMATION = False
    
    SAVE_SOME_ANIMATION = False

    # Part I ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for sysid, system in enumerate(systems):
        # for repetition in range(num_target_trajs):
        for repetition in [0, 2, 4, 6, 8]:

            np.random.seed(sysid * 199933 + repetition * 993319)
            seed = np.random.randint(2**32 - 1)

            goodtrajectory = False
            skips = 1
            while not goodtrajectory:
                kwargs = {
                    'parameters': {},
                }
                kwargs['parameters'].update(system['parameters'])
                X0 = {'mode': 'random_ws_no_vel'}
                S = system['system'](**kwargs,
                                     seed=seed,
                                     name=system['name'],
                                     X0=X0)

                # get base points
                base = [S.get_cart_pos()[:2],
                        S.sample_cart_point_from_ws()]
                for _ in range(skips):
                    S.seed = np.random.randint(2**32 - 1)
                base += [S.sample_cart_point_from_ws()]
                for _ in range(skips):
                    S.seed = np.random.randint(2**32 - 1)
                base += [S.sample_cart_point_from_ws()]

                # ensure nice trajectory
                for k in [1, 2, 3]:
                    while (
                        np.linalg.norm(base[k]-base[k-1]) > 2.4 or
                            np.linalg.norm(base[k]-base[k-1]) < .8):
                        base[k] = S.sample_cart_point_from_ws()
                        for _ in range(skips):
                            S.seed = np.random.randint(2**32 - 1)

                # build trajectory
                target_traj_a = interp1d(
                    [0, 67, 130, 202],
                    [
                        base[0][0],
                        base[1][0],
                        base[2][0],
                        base[3][0],
                    ],
                    kind='cubic')(np.arange(202))
                target_traj_b = interp1d(
                    [0, 67, 130, 202],
                    [
                        base[0][1],
                        base[1][1],
                        base[2][1],
                        base[3][1],
                    ],
                    kind='cubic')(np.arange(202))

                target_traj = np.c_[target_traj_a, target_traj_b].T

                # assemble df and save
                sysname = pd.DataFrame(data=(S.name, )*S._timesteps,
                                       columns=('System', ))
                t = pd.DataFrame(data=S.timegrid,
                                 columns=('t', ))

                target = pd.DataFrame(
                    data=target_traj[:, :201].T,
                    columns=['target X', 'target Y'])

                sysname.join(t).join(target).to_csv(
                    out_dir_as_path /
                    f"{system['name']}_{repetition:03d}.csv",
                    float_format='%.6f',
                    encoding='utf8',
                    index=False)

                # get baseline performances
                # load to get same precision for scoring
                targetdf = pd.read_csv(
                    out_dir_as_path /
                    f"{system['name']}_{repetition:03d}.csv")
                scoring_targetvals = targetdf[[
                    c for c in targetdf.columns
                    if c.startswith('target ')]].to_numpy().T

                # LQR controller
                lqrloss, S = run_lqr_controller(system, scoring_targetvals, seed, target_traj)

                # save animation
                if SAVE_ANIMATION:
                    save_animation(system, S, out_debug_dir_as_path, repetition, target_traj, lqrloss)

                # zero controller
                zeroloss, _ = run_zero_controller(system, scoring_targetvals, seed)

                save_baseline_to_csv(system, repetition, out_dir_as_path, lqrloss, zeroloss)

                # check whether trajectories are ok
                df = pd.read_csv(
                    out_dir_as_path /
                    f"{system['name']}_{repetition:03d}.csv.baselines",
                    index_col='controller')
                deviationw = 10 / df.loc['zero']['deviation']
                controlw = (1 - deviationw * df.loc['lqr']['deviation']
                            ) / df.loc['lqr']['controlcost']
                if controlw <= 0 or deviationw <= 0:
                    skips += 1
                else:
                    goodtrajectory = True
    
    # Part II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for sysid, system in enumerate(systems):
        # shifting to get the next 10 trajectories
        for repetition in range(10, 20, 2):

            np.random.seed(sysid * 189933 + repetition * 893319)
            seed = np.random.randint(2**32 - 1)

            kwargs = {
                'parameters': {},
            }
            kwargs['parameters'].update(system['parameters'])
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0='random_ws_no_vel')
            # first robot, sample candidate points
            candidate = [
                S.get_cart_pos(),
                np.r_[S.sample_cart_point_from_ws(), np.zeros(2)]]
            while np.linalg.norm(candidate[1] - candidate[0]) < 3:
                candidate[1] = np.r_[
                    S.sample_cart_point_from_ws(), np.zeros(2)]
                S.seed = np.random.randint(2**32 - 1)

            # now we have a valid candidate
            # walk the trajectory once with the
            # lqr controller and record it, to set as target
            # now these seeds are in line with the ingestion program
            np.random.seed(sysid * 199933 + repetition * 993319)
            seed = np.random.randint(2**32 - 1)

            # Create system
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0={
                                     'mode': 'from_ws_point_no_vel',
                                     'ws_point': candidate[0][:2],
                                 })
            repetition_name = f"{S.name}_{repetition:02d}"

            # Run controller
            robot_ctrl = RobotLQRController(
                seed=seed,
                robot_sys=S,
            )

            desired_ee_posvel = candidate[1]
            desired_ee_traj = np.tile(desired_ee_posvel, (S._timesteps - 1, 1))

            for t in range(S._timesteps - 1):
                # Observation with noise - consider whether using noiseless?
                current_obsv = S[:, t]
                reference_pose = desired_ee_traj[t]
                inp = robot_ctrl.get_input(current_obsv, reference_pose)
                S.step(u=inp)

            # now trajectory is recorded, save it as target_csv

            # assemble df and save
            sysname = pd.DataFrame(data=(S.name, )*S._timesteps,
                                   columns=('System', ))
            t = pd.DataFrame(data=S.timegrid,
                             columns=('t', ))

            target = pd.DataFrame(
                data=S.getDF()[['X', 'Y']].values,
                columns=['target X', 'target Y'])

            sysname.join(t).join(target).to_csv(
                out_dir_as_path /
                f"{system['name']}_{repetition:03d}.csv",
                float_format='%.6f',
                encoding='utf8',
                index=False)

            # get baseline performances

            # load to get same precision for scoring
            targetdf = pd.read_csv(
                out_dir_as_path /
                f"{system['name']}_{repetition:03d}.csv")
            scoring_targetvals = targetdf[[
                c for c in targetdf.columns
                if c.startswith('target ')]].to_numpy().T

            # append last point once for lqr oracle performance w/ speed
            target_traj = np.hstack([scoring_targetvals,
                                     scoring_targetvals[:, -1:]])

            # LQR controller
            lqrloss, S = run_lqr_controller(system, scoring_targetvals, seed, target_traj)

            # save animation
            if SAVE_ANIMATION:
                save_animation(system, S, out_debug_dir_as_path, repetition, target_traj, lqrloss)

            # zero controller
            zeroloss, _ = run_zero_controller(system, scoring_targetvals, seed)

            save_baseline_to_csv(system, repetition, out_dir_as_path, lqrloss, zeroloss)

    # Part III ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # iterate over all systems once
    # to generate list of ws points (in the intersection of all workspaces)
    # need 2 + 2 (2 full/half circles) + 3 (for rectangles)
    ws_pos = []
    k = 1943
    while len(ws_pos) < 2 + 2 + 3:
        k += 1
        validpoint = True
        for sysid, system in enumerate(systems):
            np.random.seed(sysid * 16933 + k * 3319)
            seed = np.random.randint(2**32 - 1)
            kwargs = {
                'parameters': {},
            }
            kwargs['parameters'].update(system['parameters'])
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0='random_ws_no_vel')
            # first robot, sample candidate points
            if sysid == 0:
                candidate = S.get_cart_pos()[:2]
            # other robots, test whether point is also in their workspace
            else:
                if not (S.is_cart_point_in_ws(candidate)
                        and candidate[0] > 1.3
                        and candidate[1] > 1.3
                        and np.linalg.norm(candidate) < 2.4):
                    validpoint = False
                    break
        if validpoint:
            ws_pos.append(candidate)

    # now we found the points, let's prepare the trajectories
    TRAJS = []
    # two full circles
    r = np.linalg.norm(ws_pos[0])
    x = r * np.cos(np.linspace(0, 4*np.pi, 400))
    y = r * np.sin(np.linspace(0, 4*np.pi, 400))
    ind = 41
    TRAJS.append(np.c_[x, y][ind:ind+201, :])
    #
    r = np.linalg.norm(ws_pos[1])
    x = r * np.cos(np.linspace(0, 4*np.pi, 400))
    y = r * np.sin(np.linspace(0, 4*np.pi, 400))
    ind = 96
    TRAJS.append(np.c_[x, y][ind:ind+201, :][::-1, :])
    # two half circles
    r = np.linalg.norm(ws_pos[2])
    x = r * np.cos(np.linspace(0, 2*np.pi, 400))
    y = r * np.sin(np.linspace(0, 2*np.pi, 400))
    ind = 112
    TRAJS.append(np.c_[x, y][ind:ind+201, :])
    #
    r = np.linalg.norm(ws_pos[3])
    x = r * np.cos(np.linspace(0, 2*np.pi, 400))
    y = r * np.sin(np.linspace(0, 2*np.pi, 400))
    ind = 67
    TRAJS.append(np.c_[x, y][ind:ind+201, :][::-1, :])
    # rectangles
    points = [
        ws_pos[4],
        ws_pos[4] * np.array([1, -1]),
        ws_pos[4] * np.array([-1, -1]),
        ws_pos[4] * np.array([-1, 1]),
        ws_pos[4],
    ]
    x = interp1d(
        [0, 40, 120, 160, 201],
        [p[0] for p in points],
        kind='slinear')(np.arange(201))
    y = interp1d(
        [0, 40, 120, 160, 201],
        [p[1] for p in points],
        kind='slinear')(np.arange(201))
    TRAJS.append(np.c_[x, y])
    #
    points = [
        ws_pos[5],
        ws_pos[5] * np.array([1, -1]),
        ws_pos[5] * np.array([-1, -1]),
        ws_pos[5] * np.array([-1, 1]),
        ws_pos[5],
    ]
    x = interp1d(
        [0, 40, 120, 160, 201],
        [p[0] for p in points],
        kind='slinear')(np.arange(201))
    y = interp1d(
        [0, 40, 120, 160, 201],
        [p[1] for p in points],
        kind='slinear')(np.arange(201))
    TRAJS.append(np.c_[x, y])
    #
    points = [
        ws_pos[6],
        ws_pos[6] * np.array([1, -1]),
        ws_pos[6] * np.array([-1, -1]),
        ws_pos[6] * np.array([-1, 1]),
        ws_pos[6],
    ]
    x = interp1d(
        [0, 40, 120, 160, 201],
        [p[0] for p in points],
        kind='slinear')(np.arange(201))
    y = interp1d(
        [0, 40, 120, 160, 201],
        [p[1] for p in points],
        kind='slinear')(np.arange(201))
    TRAJS.append(np.c_[x, y])

    # === LBD stuff ===
    n = 201
    # height 1, width .5 before scaling
    ##
    # Construct L
    ##
    L = np.zeros((n, 2))
    xpar = 0.45
    dist1 = 1
    dist2 = xpar
    len1 = int(n/(dist1 + dist2)*dist1)
    len2 = n - len1
    L[:len1, 1] = np.linspace(1, 0, num=len1)
    L[len1:, 0] = np.linspace(0, xpar, num=len2)
    # scale / move
    muxL = -1.528
    muyL = -1.492
    scaleL = 3.129
    L *= scaleL
    L[:, 0] = L[:, 0] + muxL
    L[:, 1] = L[:, 1] + muyL
    TRAJS.append(L)

    ##
    # Construct B
    ##
    B = np.zeros((n, 2))
    r1 = 0.2
    r2 = 0.3
    dist1 = 1
    dist2 = np.pi*r1
    dist3 = np.pi*r2
    dist_tot = dist1 + dist2 + dist3
    len1 = int(dist1/dist_tot*n)
    len2 = int(dist2/dist_tot*n)
    len3 = n - len1 - len2
    # Line
    B[:len1, 1] = np.linspace(0, 1, num=len1)
    # Upper circle
    B[len1:(len1 + len2), 0] = r1*np.cos(np.linspace(
        np.pi/2, -np.pi/2, num=len2))
    B[len1:(len1 + len2), 1] = r1*np.sin(np.linspace(
        np.pi/2, -np.pi/2, num=len2)) + 2*r2 + r1
    # Lower circle
    B[(len1 + len2):, 0] = r2*np.cos(np.linspace(
        np.pi/2, -np.pi/2, num=len3))
    B[(len1 + len2):, 1] = r2*np.sin(np.linspace(
        np.pi/2, -np.pi/2, num=len3)) + r2
    # scale / move
    muxB = 1.309
    muyB = -1.492
    scaleB = 3.129
    B *= scaleB
    B[:, 0] = B[:, 0] + muxB
    B[:, 1] = B[:, 1] + muyB
    TRAJS.append(B)

    ##
    # Construct D
    ##
    D = np.zeros((n, 2))
    rad = 4/10*np.pi
    r = 1/(2*np.sin(rad))
    dist1 = 1
    dist2 = 2*rad*r
    dist_tot = dist1 + dist2
    len1 = int(dist1/dist_tot*n)
    len2 = n - len1
    # Line
    D[:len1, 1] = np.linspace(0, 1, num=len1)
    # Circle
    D[len1:, 0] = r*np.cos(np.linspace(
        rad, -rad, num=len2)) - np.cos(rad)*r
    D[len1:, 1] = r*np.sin(np.linspace(
        rad, -rad, num=len2)) + 0.5
    # scale / move
    muxD = -2.109
    muyD = -1.492
    scaleD = 3.129
    D *= scaleD
    D[:, 0] = D[:, 0] + muxD
    D[:, 1] = D[:, 1] + muyD
    TRAJS.append(D)

    # === generate the trajectories as targets with LQR controller baseline

    # iterate over all systems
    for sysid, system in enumerate(systems):
        # shifting to get the next 10 trajectories
        for repetition in range(20, 20 + 10):
            # now these seeds are in line with the ingestion program
            np.random.seed(sysid * 199933 + repetition * 993319)
            seed = np.random.randint(2**32 - 1)

            target = TRAJS[repetition - 20]

            kwargs = {
                'parameters': {},
            }
            kwargs['parameters'].update(system['parameters'])

            # Create system
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0={
                                     'mode': 'from_ws_point_no_vel',
                                     'ws_point': target[0, :],
                                 })
            repetition_name = f"{S.name}_{repetition:02d}"

            # Run controller
            robot_ctrl = RobotLQRController(
                seed=seed,
                robot_sys=S,
            )

            # assemble df and save
            sysname = pd.DataFrame(data=(S.name, )*S._timesteps,
                                   columns=('System', ))
            t = pd.DataFrame(data=S.timegrid,
                             columns=('t', ))

            target = pd.DataFrame(
                data=target,
                columns=['target X', 'target Y'])

            sysname.join(t).join(target).to_csv(
                out_dir_as_path /
                f"{system['name']}_{repetition:03d}.csv",
                float_format='%.6f',
                encoding='utf8',
                index=False)

            # get baseline performances

            # load to get same precision for scoring
            targetdf = pd.read_csv(
                out_dir_as_path /
                f"{system['name']}_{repetition:03d}.csv")
            scoring_targetvals = targetdf[[
                c for c in targetdf.columns
                if c.startswith('target ')]].to_numpy().T

            # append last point once for lqr oracle performance w/ speed
            target_traj = np.hstack([scoring_targetvals,
                                     scoring_targetvals[:, -1:]])

            # LQR controller
            lqrloss, S = run_lqr_controller(system, scoring_targetvals, seed, target_traj)

            # save animation
            if SAVE_ANIMATION:
                save_animation(system, S, out_debug_dir_as_path, repetition, target_traj, lqrloss)

            # zero controller
            zeroloss, _ = run_zero_controller(system, scoring_targetvals, seed)

            save_baseline_to_csv(system, repetition, out_dir_as_path, lqrloss, zeroloss)
