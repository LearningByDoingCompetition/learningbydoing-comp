import numpy as np
from dgp.systems import Switch_2x2


np.random.seed(1)
scp = 1
wcp = 0.25
fr1 = 0.5
fr2 = 0.7
sr1 = 0.1
sr2 = 0.05
dr2 = 0.0125
dr3 = 0.05

# Slow rates
rates_slow1 = (np.concatenate([[sr1, sr1, sr2, sr2],
                               [sr1, sr2, sr1, sr2],
                               [dr3, dr3]]))
rates_slow2 = (np.concatenate([[sr2, sr2, sr1, sr1],
                               [sr2, sr1, sr2, sr1],
                               [dr3, dr3]]))
# Fast rates
rates_fast1 = (np.concatenate([[fr1, fr1, fr2, fr2],
                               [fr1, fr2, fr1, fr2],
                               [dr3, dr3]]))
rates_fast2 = (np.concatenate([[fr2, fr2, fr1, fr1],
                               [fr2, fr1, fr2, fr1],
                               [dr3, dr3]]))
# Mixed rates
rates_mixed1 = (np.concatenate([[fr1, sr2, fr1, sr2],
                                [fr2, sr1, fr2, sr1],
                                [dr2, dr3]]))
rates_mixed2 = (np.concatenate([[fr2, sr2, fr2, sr1],
                                [fr1, sr1, fr1, sr2],
                                [dr2, dr3]]))

# Random rates
rates_rand1 = (np.concatenate([np.random.uniform(0.2, 0.5, 8),
                               np.array([dr2, dr3])]))
rates_rand2 = (np.concatenate([np.random.uniform(0.2, 0.5, 8),
                               np.array([dr2, dr3])]))

# Bmat
Bmat = np.array([[scp, 0, 0, 0, wcp, 0, 0, 0],
                 [0, scp, 0, 0, 0, wcp, 0, 0],
                 [scp, 0, 0, 0, 0, 0, wcp, 0],
                 [0, scp, 0, 0, 0, 0, 0, wcp],
                 [0, 0, scp, 0, 0, 0, wcp, 0],
                 [0, 0, 0, scp, 0, 0, 0, wcp],
                 [0, 0, scp, 0, wcp, 0, 0, 0],
                 [0, 0, 0, scp, 0, wcp, 0, 0]])

# Confounding matrix
AA = np.zeros((23, 10))
# X3, X4, X5, X6, X10, X11, U5, U6
block1_ind = np.array([0, 0, 0, 1, 1, 1, 1, 0,
                       0, 0, 1, 1, 0, 0, 0, 0,
                       0, 0, 0, 1, 1, 0, 0])
# X1, X2, X7, X8, X9, X12, U7, U8
block2_ind = np.array([0, 1, 1, 0, 0, 0, 0, 1,
                       1, 1, 0, 0, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 1, 1])
# Assign correlation to first block
mm = sum(block1_ind)
zeros = np.random.choice(
    np.repeat((0, 1), [3*5, (mm-3)*5]), (mm, 5), replace=False)
AA[block1_ind == 1, :5] = zeros
# Assign correlation to second block
mm = sum(block2_ind)
zeros = np.random.choice(
    np.repeat((0, 1), [3*5, (mm-3)*5]), (mm, 5), replace=False)
AA[block2_ind == 1, 5:] = zeros
# Add minimal correlation and normalize AA
AA += 0.1
AA[(block1_ind + block2_ind) == 1, :] /= np.sqrt(
    np.diag(AA.dot(AA.T))[
        (block1_ind + block2_ind) == 1].reshape(-1, 1))
AA[:15, :] /= np.sqrt(2)


systems = [
    # System 1
    {
        'name': 'system_01',
        'parameters': {'rates': rates_slow1,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 1)}
    },
    # System 2
    {
        'name': 'system_02',
        'parameters': {'rates': rates_slow2,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 1)}
    },
    # System 3
    {
        'name': 'system_03',
        'parameters': {'rates': rates_slow1,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 1),
                         'AA': AA}
    },
    # System 4
    {
        'name': 'system_04',
        'parameters': {'rates': rates_slow2,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 1),
                         'AA': AA}
    },
    # System 5
    {
        'name': 'system_05',
        'parameters': {'rates': rates_fast1,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 1)}
    },
    # System 6
    {
        'name': 'system_06',
        'parameters': {'rates': rates_fast2,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 1)}
    },
    # System 7
    {
        'name': 'system_07',
        'parameters': {'rates': rates_fast1,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 1),
                         'AA': AA}
    },
    # System 8
    {
        'name': 'system_08',
        'parameters': {'rates': rates_fast2,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 1),
                         'AA': AA}
    },
    # System 9
    {
        'name': 'system_09',
        'parameters': {'rates': rates_mixed1,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 2),
                         'AA': AA}
    },
    # System 10
    {
        'name': 'system_10',
        'parameters': {'rates': rates_mixed2,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (1, 2),
                         'AA': AA}
    },
    # System 11
    {
        'name': 'system_11',
        'parameters': {'rates': rates_rand1,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
    # System 12
    {
        'name': 'system_12',
        'parameters': {'rates': rates_rand2,
                       'B': Bmat},
        'system': Switch_2x2,
        'control_args': {'corind': (0, 0)}
    },
]


def gset_evaluation_seed(sysid, repetition):
    np.random.seed(sysid * 71993 + repetition * 999331)
    return np.random.randint(2**32 - 1)
