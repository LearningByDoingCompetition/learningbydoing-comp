# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
# The code directory submission_program_dir (e.g. sample_code_submission/)
# should contain your
# code submission model.py (and possibly other functions it depends upon).

# =========================== BEGIN OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True  # outputs messages to stdout and stderr for debug purposes
subverbose = False

# Time budget
#############
# Maximum time for trajectory tracking in seconds.
# The code should keep track of time spent and NOT exceed the time limit.
# You can decrease the maximum time (in sec) with this variable:
# 0.01 * 200 is "system time", that is 200 inputs, and the overall trajectory
# spans a 2 seconds time window
# times have been doubled compared to alpha phase
# 2, 16, 16, 120
system_time = 0.01 * 200
max_time_to_get_first_control = 16
max_time_to_get_one_control = 16 / 200
max_time_per_trajectory = (200 * max_time_to_get_one_control
                           + max_time_to_get_first_control)


def alarm_get_one_control(signum, frame):
    raise Exception(
        "Next control input not retrieved in time; "
        f"took > {max_time_to_get_one_control}s. "
        "This may be, for example, due to exhausting the memory limit "
        "or the computations in get_input taking too long.")


def alarm_get_first_control(signum, frame):
    raise Exception(
        "First control input not retrieved in time; "
        f"took > {max_time_to_get_first_control}s. "
        "This may be, for example, due to exhausting the memory limit, "
        "or the computations in __init__ or get_input taking too long.")


# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# General purpose functions
# suppress import not at top warnings, so we can stick to file structure
import time  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datetime import datetime  # noqa: E402
import hashlib  # noqa: E402
import signal  # noqa: E402

# =========================== BEGIN PROGRAM ================================

import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: 402
import zmq  # noqa: 402

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib  # noqa: 402
import base64  # noqa: 402
from track2.systems import systems  # noqa: 402

from tqdm import tqdm

def get_next_input(sys_id, init, d_U, ee_obsv, ref_state, debug):
    # needs to be in sync with ingestion_wrapcontroller,
    # which receives these messages
    state = {
        'system': sys_id,
        'init': init,
        'd_control': d_U,
    }
    # overwrite true state and target in participant debug mode
    if debug:
        state['state'] = np.random.randn(*ee_obsv.shape).round(6).tolist()
        state['target'] = np.random.randn(*ref_state.shape).round(6).tolist()
    else:
        state['state'] = ee_obsv.round(6).tolist()
        state['target'] = ref_state.round(6).tolist()
    # append current position for convenience
    state['position'] = state['state'][:2]
    socket.send_json(state)
    poll()
    while socket.poll(timeout=1000) == 0:
        poll()
    if debug:
        return np.random.randn(
            *np.asarray(socket.recv_json(), dtype='float64').shape).round(6)
    else:
        return np.asarray(socket.recv_json(), dtype='float64').round(6)


def poll(error='Error in participant controller code.'):
    code = wrapcontroller.poll()
    # CHANGE: this fixes a weird docker bug
    # https://github.com/docker/classicswarm/issues/2835
    # if code is not None and code > 0 and code != 125:
    # maybe that isn't needed after all?
    if code is not None and code > 0:
        if error is None:
            return False
        else:
            raise Exception(error)
    return True


def lossfc(target, S):
    # per convention, the first two dimensions are end effectors X, Y coords
    deviation = np.sum((target[:, 1:] - S[:2, 1:])**2) / S.shape[1]
    controlcost = np.sum(np.einsum('ij,ij->j', S.U, S.U)) / S.U.shape[1]
    return np.array([deviation, controlcost])


def scaled_loss(target, S, target_traj_path):
    loss = lossfc(target, S)
    deviation = np.sqrt(loss[0])
    control = np.sqrt(loss[1])

    df = pd.read_csv(
        target_traj_path.parent / (
            target_traj_path.name + '.baselines'),
        index_col='controller')
    control_lqr = np.sqrt(df.loc['lqr']['controlcost'])
    deviation_lqr = np.sqrt(df.loc['lqr']['deviation'])
    deviation_zero = np.sqrt(df.loc['zero']['deviation'])
    e = np.log(100 * 2) / (np.log(deviation_zero) - np.log(deviation_lqr))

    scaled_deviation = np.min([(deviation / deviation_lqr)**e / 2, 100])
    scaled_control = control / control_lqr / 2
    return (np.min([scaled_deviation + scaled_control, 100]),
            deviation,
            control,
            scaled_deviation,
            scaled_control)


def evaluate_track2(controller_dir, trajectories_dir, output_dir=None, force_output=False, debug_output=False, show_viz=False, verbose=False):

    submission_dir = Path(controller_dir)
    assert submission_dir.exists(), \
        f"Expected controller_dir \"{submission_dir}\" to exist, but it does not."

    input_dir = Path(trajectories_dir)
    assert input_dir.exists(), \
        f"Expected trajectories_dir \"{input_dir}\" to exist, but it does not."

    LOCAL_DEBUG = debug_output

    if output_dir is None:
        output_results = False
    else:
        output_results = True
        out_dir = Path(output_dir)
        out_dir.mkdir(
            parents=True,
            exist_ok=force_output,
        )
        if LOCAL_DEBUG:
            output_debug_dir = out_dir / "debug"
            output_debug_dir.mkdir(
                parents=True,
                exist_ok=force_output,
            )

    competition_phase = 'final'
    DOCKER = False
    LOCAL = True
    # LOCAL_DEBUG = False
    SOME_LOCAL_DEBUG = False
    PARTICIPANT_DEBUG = False
    subverbose = False

    # FEW_TRAJECTORIES only evaluates on a few trajectories (for debugging)
    FEW_TRAJECTORIES = False

    # systems
    # +0: 'Prismatic-id',
    # 1: 'Prismatic-square-imbalanced',
    # +2: 'Prismatic-rectangular-small',
    # 3: 'Prismatic-rectangular',
    # 4: 'Prismatic-id-R',
    # +5: 'Prismatic-square-imbalanced-R',
    # 6: 'Prismatic-rectangular-small-R',
    # +7: 'Prismatic-rectangular-R',
    # +8: 'Rotational2-id',
    # 9: 'Rotational2-square-imbalanced',
    # +10: 'Rotational2-rectangular-small',
    # 11: 'Rotational2-rectangular',
    # 12: 'Rotational2-id-R',
    # +13: 'Rotational2-square-imbalanced-R',
    # 14: 'Rotational2-rectangular-small-R',
    # +15: 'Rotational2-rectangular-R',
    # +16: 'Rotational3-id',
    # 17: 'Rotational3-square-imbalanced',
    # +18: 'Rotational3-rectangular-small',
    # 19: 'Rotational3-rectangular',
    # 20: 'Rotational3-id-R',
    # +21: 'Rotational3-square-imbalanced-R',
    # 22: 'Rotational3-rectangular-small-R',
    # +23: 'Rotational3-rectangular-R',

    EVALUATION_SYSTEMS = [
         1,  3,  4,  6,
         9, 11, 12, 14,
        17, 19, 20, 22,
    ]
    REPORT_SYSTEMS = EVALUATION_SYSTEMS.copy()

    # get docker container name
    with open(Path(submission_dir) / 'metadata') as f:
        image = f.read()
    # just to ensure start.sh is there
    with open(Path(submission_dir) / 'start.sh') as f:
        pass

    image = '"{}"'.format(image.strip().split(' ')[0])
    image = re.sub('[^0-9a-zA-Z/.:-]+', '', image)
    image = shlex.quote(image)

    # Check whether everything went well (no time exceeded)
    execution_success = True

    if verbose:
        print(f"Using input_dir: {input_dir}")
        if output_results:
            print(f"Using output_dir: {out_dir}")
        else:
            print("Not saving results")
        print(f"Using submission_dir: {submission_dir}")

    dstderr, dstdout = None, None
    # hide controller stdout and stderr if non-loca, i.e. on codalab
    if not LOCAL or not subverbose:
        dstdout, dstderr = subprocess.DEVNULL, subprocess.DEVNULL

    # hide controller stdout and stderr if non-loca, i.e. on codalab
    if PARTICIPANT_DEBUG:
        dstderr, dstdout = None, None

    wrapcontrollercommand = [
        '/bin/bash',
        Path(submission_dir) / 'start.sh',
    ]

    # start the wrapcontroller
    if verbose:
        print("Starting controller...")
    global wrapcontroller
    wrapcontroller = subprocess.Popen(
        wrapcontrollercommand,
        stdout=dstdout,
        stderr=dstderr)

    for slt in range(1, 6):
        time.sleep(slt**2)
        if poll(None):
            break
    poll('Controller could not be started.')

    # inform participants that we swallow logs related to their controller
    if not LOCAL and not PARTICIPANT_DEBUG:
        print("Note: stdout/-err of controller is excluded from this log.")
    elif not LOCAL:
        print("PARTICIPANT CONTROLLER DEBUG MODE (fake system observations)")

    total_runtime = 0
    RESULTS = []

    # try block to ensure the wrapcontroller subprocess is terminated
    # when there are uncaught exceptions
    try:
        # Connect to wrapped controller
        context = zmq.Context()
        global socket
        socket = context.socket(zmq.REQ)
        socket.connect("ipc://" + str(Path(submission_dir) / "socket"))

        # =====
        # BELOW HERE IS WHERE THE ACTUAL CONTROLL STUFF IS HAPPENING
        # -----

        # Load target trajectories
        target_traj_paths = sorted((Path(input_dir)).glob('*.csv'))
        if FEW_TRAJECTORIES:
            target_traj_paths = [p for p in target_traj_paths
                                 if str(p)[-8:-4] in [
                                     '_001', '_012', '_016']]
        # filter target_trajs, to only keep the ones for EVALUATION_SYSTEMS
        whitelistnames = [systems[k]['name']
                          for k in EVALUATION_SYSTEMS]
        target_traj_paths = [p for p in target_traj_paths
                             if p.stem[:-4] in whitelistnames]

        max_time = max_time_per_trajectory * len(target_traj_paths)
        if verbose:
            print(f"Max controller time is {max_time/60:.1f} min "
                  f"for {len(target_traj_paths)} trajectories")

        # MAIN LOOP
        for target_traj_path in tqdm(target_traj_paths):
            robotname, repetition = str(target_traj_path.name)[:-4].split('_')
            repetition = int(repetition)

            if subverbose:
                print(
                    "Running controller for target trajectory: "
                    + str(target_traj_path.stem))

            targetdf = pd.read_csv(target_traj_path)
            target = targetdf[[c
                               for c in targetdf.columns
                               if c.startswith('target ')]].to_numpy().T

            # Instantiate robot
            # find the appropriate system
            sysid = [k for k, s in enumerate(systems)
                     if s['name'] == robotname][0]
            system = systems[sysid]
            # this needs to be in line with target_trajectories.py script
            # repetition is extracted from filename as above
            np.random.seed(sysid * 199933 + repetition * 993319)
            seed = np.random.randint(2**32 - 1)
            kwargs = {
                'parameters': {},
            }
            kwargs['parameters'].update(system['parameters'])
            X0 = {
                'mode': 'from_ws_point_no_vel',
                'ws_point': target[:, 0],
            }
            S = system['system'](**kwargs,
                                 name=system['name'],
                                 seed=seed,
                                 X0=X0)
            if subverbose:
                print(f"Instantiated {robotname} ({hash(S)}!)")
            # the hash os S should always be the same to ensure
            # reproducible results – sanity check: upload same controller
            # twice, then the overall score should be exactly the same

            for timestep in range(S._timesteps - 1):
                # time spent in get_next_input
                start = time.time()
                # set an alarm to limit time to get one control input
                if timestep == 0:
                    # initialising may take longer
                    signal.signal(signal.SIGALRM,
                                  alarm_get_first_control)
                    signal.setitimer(
                        signal.ITIMER_REAL,
                        max_time_to_get_first_control)
                else:
                    signal.signal(signal.SIGALRM,
                                  alarm_get_one_control)
                    signal.setitimer(
                        signal.ITIMER_REAL,
                        1.1 * max_time_to_get_one_control * (200 - timestep))
                inp = get_next_input(
                    robotname,
                    timestep == 0,
                    S.d_U,
                    S[:, timestep],
                    target[:, timestep + 1],  # target at next timestep
                    PARTICIPANT_DEBUG)
                signal.setitimer(signal.ITIMER_REAL, 0)
                trajtime = time.time() - start
                if (trajtime
                        > max_time_to_get_first_control
                        + 200 * max_time_to_get_one_control):
                    raise Exception(
                        "Max control compute time exceeded for this "
                        f"trajectory; took > {max_time_per_trajectory}s. "
                        "This may be, for example, "
                        "due to exhausting the memory limit, "
                        "or the computations in __init__ or "
                        "get_input taking too long.")
                total_runtime += trajtime
                # break if execution time is exceeded
                if total_runtime > max_time:
                    execution_success = False
                    print("Time limit exceeded")
                    break
                # we fail somewhat gracefully, that is participants
                # get to know that something was wrong with their input
                # and that the empty input will be used
                # Do we want to consider this a success still, or not?
                if not isinstance(inp, np.ndarray):
                    execution_success = False
                    print("Input command has wrong data type" +
                          " – no control is applied")
                    inp = None
                elif inp.shape != (S.d_U,):
                    execution_success = False
                    print("Input command has wrong shape " +
                          str(inp.shape) +
                          " instead of " +
                          str((S.d_U,)) +
                          " – no control is applied")
                    inp = None

                # Propagate robot model forward
                S.step(u=inp)

            if subverbose:
                if execution_success:
                    print("[+] Done")
                else:
                    print("[+] Done, but controller execution failed")
                print("[+] Overall controller time spent %5.2f sec "
                      % total_runtime
                      + "::  Remaining time budget %5.2f sec"
                      % (max_time - total_runtime))

            # reduce max_time as we want to limit overall runtime

            # kill if time exceeded
            if total_runtime > max_time:
                raise Exception(
                    f"Maximum controller time of {max_time} sec exceeded.")

            # -----
            # ABOVE HERE IS WHERE THE ACTUAL CONTROLL STUFF IS HAPPENING
            # =====

            # =====
            # BELOW HERE IS WHERE THE EVALUATION IS HAPPENING
            # -----

            target_traj_loss = scaled_loss(target, S, target_traj_path)

            if sysid in REPORT_SYSTEMS:
                RESULTS.append({
                    'robotname': robotname,
                    'repetition': repetition,
                    'loss': target_traj_loss,
                })

            # Display visualization to user
            if show_viz:
                S.show_traj_animation(
                    target.T,
                    annot_text=f"Score: {target_traj_loss[0]:.2f}",
                )

            # Write standard debug plots
            if LOCAL_DEBUG or (SOME_LOCAL_DEBUG and repetition < 0):
                if subverbose:
                    print("[+] Saving animation gif...")
                animation_filename = f"{system['name']}_{repetition:03d}.gif"
                animation_path = Path(output_debug_dir) / animation_filename
                S.save_traj_animation(
                    animation_path,
                    target.T,
                    annot_text=f"Score: {target_traj_loss[0]:.2f}"
                )
                if subverbose:
                    print("[+] Done!")

        def output(line, print_line=True):
            if output_results:
                with open(out_dir / "scores.txt", "a") as text_file:
                    text_file.write(line + "\n")
            if print_line:
                print(line)

        if len(RESULTS) > 0 and not PARTICIPANT_DEBUG:
            score = np.mean([k['loss'][0]
                             for k in RESULTS])
        else:
            score = 111
        output(f"Loss: {score:.3f}")

        result_systems = list(set(k['robotname']
                                  for k in RESULTS))
        result_bots = list(set(k['robotname'].split('-')[-1]
                               for k in RESULTS))

        # return aggregate scores
        for bot in result_bots:
            scores = np.vstack([
                k['loss']
                for k in RESULTS
                if k['robotname'].split('-')[-1] == bot]).mean(0)
            output(f"Loss-{bot}: {scores[0]:.3f}", verbose)
            output(f"Loss-{bot}-deviation: {scores[1]:.3f}", verbose)
            output(f"Loss-{bot}-deviation-scaled: {scores[3]:.3f}", verbose)
            output(f"Loss-{bot}-control: {scores[2]:.3f}", verbose)
            output(f"Loss-{bot}-control-scaled: {scores[4]:.3f}", verbose)

        # return system specific scores
        for system in result_systems:
            scores = np.vstack([
                k['loss']
                for k in RESULTS
                if k['robotname'] == system]).mean(0)
            output(f"Loss-{system}: {scores[0]:.3f}", subverbose)
            output(f"Loss-{system}-deviation: {scores[1]:.3f}", subverbose)
            output(f"Loss-{system}-deviation-scaled: {scores[3]:.3f}", subverbose)
            output(f"Loss-{system}-control: {scores[2]:.3f}", subverbose)
            output(f"Loss-{system}-control-scaled: {scores[4]:.3f}", subverbose)

        output(f"Controller-Runtime: {total_runtime:.1f}")

        # create detailed html results
        if output_results and LOCAL_DEBUG:
            animation_path = Path(output_debug_dir)
            if competition_phase == 'validation':
                gifs = sorted(animation_path.glob('*.gif'))
                imgs = {}
                for g in gifs:
                    with open(g, 'rb') as f:
                        imgs.update({
                            g.stem:
                            base64.b64encode(f.read()).decode().replace('\n', '')})
                    # remove gif
                    g.unlink()

            html_head = '''<!DOCTYPE html>
            <html>
            <head><title>Animations</title></head>
            <meta charset="utf-8">
            <body>
            '''
            html_foot = '</body></html>'

            with open(animation_path / "detailed_results.html", "w") as hf:
                hf.write(html_head)
                if competition_phase == 'validation':
                    for name, img in imgs.items():
                        hf.write(f'<h3>{name}</h3>')
                        hf.write(f'<img src="data:image/gif;base64,{img}">')
                hf.write(html_foot)

        # -----
        # ABOVE HERE IS WHERE THE EVALUATION IS HAPPENING
        # =====

    finally:
        wrapcontroller.terminate()
        wrapcontroller.kill()

    robo_eval_results = {
        "loss": score,
        "duration": total_runtime,
        "results": RESULTS,
    }

    return robo_eval_results
