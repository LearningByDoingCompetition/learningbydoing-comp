import numpy as np
from dgp.systems import Switch_2x2


np.random.seed(1)
scp = 1
wcp = 0.25
rates = (np.concatenate([np.repeat(0.5, 8),
                         np.array([0.05, 0.05])]))


systems = [
    # System 1
    {
        'name': 'system_01',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
    # System 2
    {
        'name': 'system_02',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 1)}
    },
    # System 3
    {
        'name': 'system_03',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 2)}
    },
    # System 4
    {
        'name': 'system_04',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 0)}
    },
    # System 5
    {
        'name': 'system_05',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 1)}
    },
# System 6
    {
        'name': 'system_06',
        'parameters': {'rates': (np.concatenate(
            [(np.random.randint(0, 2, 8)/10*4+0.1)*1.1,
             np.array([0.05, 0.05])])),
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 2)}
    },
    # System 7
    {
        'name': 'system_07',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (2, 0)}
    },
    # System 8
    {
        'name': 'system_08',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (2, 1)}
    },
    # System 9
    {
        'name': 'system_09',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (2, 2)}
    },
    # System 10
    {
        'name': 'system_10',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
]


def gset_evaluation_seed(sysid, repetition):
    np.random.seed(sysid * 71993 + repetition * 999331)
    return np.random.randint(2**32 - 1)
import numpy as np
from dgp.systems import Switch_2x2


np.random.seed(1)
scp = 1
wcp = 0.25
rates = (np.concatenate([np.repeat(0.5, 8),
                         np.array([0.05, 0.05])]))


systems = [
    # System 1
    {
        'name': 'system_01',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
    # System 2
    {
        'name': 'system_02',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 1)}
    },
    # System 3
    {
        'name': 'system_03',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 2)}
    },
    # System 4
    {
        'name': 'system_04',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 0)}
    },
    # System 5
    {
        'name': 'system_05',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 1)}
    },
    # System 6
    {
        'name': 'system_06',
        'parameters': {'rates': (np.concatenate(
            [(np.random.randint(0, 2, 8)/10*4+0.1)*1.1,
             np.array([0.05, 0.05])])),
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 2)}
    },
    # System 7
    {
        'name': 'system_07',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (2, 0)}
    },
    # System 8
    {
        'name': 'system_08',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (2, 1)}
    },
    # System 9
    {
        'name': 'system_09',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (2, 2)}
    },
    # System 10
    {
        'name': 'system_10',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
    # System 11
    {
        'name': 'system_10',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
    # System 12
    {
        'name': 'system_10',
        'parameters': {'rates': rates,
                       'B': np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                                      [0, scp, 0, 0, 0, wcp, 0, 0],
                                      [scp, 0, 0, 0, 0, 0, wcp, 0],
                                      [0, scp, 0, 0, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, 0, 0, wcp, 0],
                                      [0, 0, 0, scp, 0, 0, 0, wcp],
                                      [0, 0, scp, 0, wcp, 0, 0, 0],
                                      [0, 0, 0, scp, 0, wcp, 0, 0]])},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
]


def gset_evaluation_seed(sysid, repetition):
    np.random.seed(sysid * 71993 + repetition * 999331)
    return np.random.randint(2**32 - 1)
