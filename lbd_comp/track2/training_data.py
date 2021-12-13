import numpy as np
from pathlib import Path

from tqdm import tqdm

from track2.robot_control_baselines import (
    RobotRandomController,
    RobotLQRController,
)
from track2.systems import systems


def generate_training_data(output_dir, num_training_trajs, force_output=False, debug_output=False):

    # training configurations
    controller_to_use = 'lqr'
    generate_some_training_debug = False

    assert num_training_trajs > 0, (
        f"Expected \"num_training_trajs\" to be greater than zero, but it is {num_training_trajs}."
    )

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

    # iterate over all systems once
    # to generate list of pairs of ws points (intersection of all workspaces)
    # to use as training targets later
    ws_pos_pairs = []
    k = 0
    while len(ws_pos_pairs) < num_training_trajs:
        k += 1
        validpoint = True
        for sysid, system in enumerate(systems):
            np.random.seed(sysid * 199933 + k * 993319)
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
                candidate = [
                    S.get_cart_pos(),
                    np.r_[S.sample_cart_point_from_ws(), np.zeros(2)]]
                while np.linalg.norm(candidate[1] - candidate[0]) < 3:
                    candidate[1] = np.r_[
                        S.sample_cart_point_from_ws(), np.zeros(2)]
                    S.seed = np.random.randint(2**32 - 1)
            # other robots, test whether point is also in their workspace
            else:
                if (not S.is_cart_point_in_ws(candidate[0][:2])
                        or not S.is_cart_point_in_ws(candidate[1][:2])):
                    validpoint = False
                    break
        if validpoint:
            ws_pos_pairs.append(candidate + [k])

    # iterate over all systems
    for sysid, system in enumerate(tqdm(systems)):
        print(f"{system['name']} – {system['system']}")

        kwargs = {
            'parameters': {},
        }
        kwargs['parameters'].update(system['parameters'])

        # do some training trajectories per each system
        # with different random seeds
        # (also affects observational noise, if there is any,
        #  handled by the systems class)
        for repetition, (startpos, targetpos, k) in enumerate(
                ws_pos_pairs[:num_training_trajs]):
            np.random.seed(sysid * 199933 + k * 993319)
            seed = np.random.randint(2**32 - 1)

            # Create system
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0={
                                     'mode': 'from_ws_point_no_vel',
                                     'ws_point': startpos[:2],
                                 })
            repetition_name = f"{S.name}_{repetition:02d}"

            # Run controller
            if controller_to_use == 'random':
                robot_ctrl = RobotRandomController(seed, S.d_U)
            elif controller_to_use == 'lqr':
                robot_ctrl = RobotLQRController(
                    seed=seed,
                    robot_sys=S,
                )
                # REMOVED – we have clipping now in dgp/systems/robot_arm.py
                # turn gains down a bit to prevent solver overflow
                # fix this by adding a clip to step
                # robot_ctrl.Q = np.diag([10., 10., 0.1, 0.1])
            else:
                raise NotImplementedError

            # Pick random point in workspace with zero vel
            desired_ee_posvel = targetpos
            desired_ee_traj = np.tile(desired_ee_posvel, (S._timesteps - 1, 1))

            for t in range(S._timesteps - 1):
                # Observation with noise - consider whether using noiseless?
                current_obsv = S[:, t]
                reference_pose = desired_ee_traj[t]
                inp = robot_ctrl.get_input(current_obsv, reference_pose)
                S.step(u=inp)

            S.getDF().to_csv(
                out_dir_as_path / f"{repetition_name}.csv",
                float_format='%.6f',
                encoding='utf8',
                index=False)

            if debug_output or (
                    generate_some_training_debug and repetition
                    in [2, 12, 22, 26, 28]):
                # save animation to disk
                animation_path = (
                    out_debug_dir_as_path / repetition_name
                ).with_suffix('.gif')
                S.save_traj_animation(
                    animation_path,
                    desired_ee_traj[:, :2],
                )

    # for convenience, we provide a list of model names
    with open(out_dir_as_path / "systems", "w") as f:
        f.writelines([system['name'] + "\n"
                      for system in systems])
