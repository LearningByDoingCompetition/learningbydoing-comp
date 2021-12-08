from dgp.systems import (
    RobotRotational2LinkILinear,
    RobotPrismatic2LinkILinear,
    RobotRotational3LinkILinear,
)
import numpy as np


# we set up systems here, similar to track1/systems.py
# naming may reveal how systems are related without revealing the true nature
# i.e. instead of
# prismatic_linear_1, ..., rotational_underact_1, ...
# we map the individual slug parts to fantasy names

# set slugs for our system namings
# system names consist of three parts identifying
# * whether its a rotational or prismatic robot
# * the control interface being used
# * the parameter setting
# (generated with coolname package)
SLUGS = {
    'rotational2': 'beetle',
    'rotational3': 'butterfly',
    'prismatic': 'bumblebee',
    # ordered list of names for interfaces
    'interfaces': [
        'great',
        'lush',
        'rebel',
        'talented',
        'thoughtful',
        'little',
        'convivial',
        'singing',
        'dark',
        'wise',
        'garrulous',
        'dyanmioc',
        'attractive',
        'spiritual',
        'spicy',
        'exotic',
        'pastoral',
        'curious',
    ],
    # ordered list of names for parameter settings
    'params': [
        'piquant',
        'bipedal',
        'impartial',
        'proficient',
        'devious',
        'vivacious',
        'mauve',
        'wine',
        'ruddy',
        'steel',
        'zippy',
        'antique',
        'laughing',
        'yellow',
        'finicky',
        'whimsical',
        'charming',
        'independent',
    ],
}


def slug(robotype, interface, param):
    """
    robotype: 'rot' (rotational), 'prism' (prismatic)
    interface: consecutive numbers for interfaces,
               make sure same interface gets same number
               between different types of robots / params
    param: consecutive numbers for parameters,
           make sure same param setting gets same number
           between different types of robots / interfaces
    """
    post = SLUGS[robotype]
    return (f"{SLUGS['interfaces'][interface]}-"
            f"{SLUGS['params'][param]}-"
            f"{post}")


# modular parts of parameters for later reuse

# length links
# * for 3rot, 2rot, 2prism
# * roughly same maximum reach (~3)
#   * there is some area rot / prism cannot reach
#     (corners of square, vs circle overlap = same size)
#   * for simplicity, prismatic workspace always square (for now?)
#   * for simplicity, no link is larger than both other links in rot
# * --> can use same training trajectories
tmp = 3 * np.sqrt(np.pi) / 4
length_links = {
    'rotational': {
        3: [
            np.array([1., 1., 1.]),
            np.array([.5, 1., 1.5]),
            # this cannot reach some inner part
            np.array([.5, .5, 2.]),
        ],
        2: [
            np.array([1.5, 1.5]),
            # these cannot reach some inner part
            np.array([1., 2.]),
            np.array([2., 1.]),
        ],
    },
    'prismatic': {
        2: [
            np.array([tmp, tmp]),
            np.array([(9 / 11) * tmp, tmp / (9 / 11)]),
            np.array([tmp / (6 / 7), (6 / 7) * tmp]),
        ],
    },
}

common_paras = {
    3: [
        {
            'inertia_links': np.array([0.862, 1.045, 0.992]),
            'mass_links': np.array([1.012, 0.928, 1.083]),
        },
        {
            'inertia_links': np.array([0.553, 0.907, 1.723]),
            'mass_links': np.array([1.041, 1.533, 0.741]),
        },
    ]
}
common_paras[2] = [{a: b[:2]
                    for (a, b) in k.items()}
                   for k in common_paras[3]]

# assemble parameters for 3 different underlying robot types
prismatic_params = [{'length_links': ll, **cp}
                    for ll in length_links['prismatic'][2]
                    for cp in common_paras[2]]
rotational2_params = [{'length_links': ll, **cp}
                      for ll in length_links['rotational'][2]
                      for cp in common_paras[2]]
rotational3_params = [{'length_links': ll, **cp}
                      for ll in length_links['rotational'][3]
                      for cp in common_paras[3]]

# linear interface matrices
# (same sum(rownorms))
# 2 square, balanced
# 2 square, imbalanced
# 2 rectangular, imbalanced
interfaces = {
    2: [
        np.array([[0.976, 0.336],
                  [0.218, 0.948]]),
        np.array([[0.797, 0.603],
                  [0.611, -0.792]]),
        #
        np.array([[0.491, 0.372],
                  [-0.460, 1.307]]),
        np.array([[1.261, 0.236],
                  [0.624, 0.360]]),
        #
        np.array([[-0.422, -0.107, -0.457, 0.403],
                  [0.860, 0.218, -0.730, -0.507]]),
        np.array([[0.089, -0.871, -1.230, -0.083],
                  [-0.326, -0.262, 0.054, 0.254]]),
    ],
    3: [
        np.array([[0.245, 0.863, -0.441],
                  [0.596, 0.416, 0.687],
                  [-0.155, 0.37, -0.916]]),
        np.array([[-0.463, 0.831, -0.310],
                  [0.266, -0.711, 0.651],
                  [0.247, -0.725, 0.643]]),
        #
        np.array([[-0.703, -1.318, -0.131],
                  [0.934, -0.345, -0.086],
                  [-0.348, -0.277, 0.229]]),
        np.array([[-0.021, -0.491, -0.754],
                  [0.502, 1.183, 0.682],
                  [0.421, 0.250, 0.428]]),
        #
        np.array([[-0.339, -0.085, -0.365, 0.322, -0.415, -0.113],
                  [0.691, 0.175, -0.587, -0.407, -0.593, -0.742],
                  [0.299, 0.492, -0.284, -0.356, -0.195, 0.439]]),
        np.array([[-0.233, -0.107, -0.171, -0.006, -0.068, -0.282],
                  [0.212, 0.996, 0.095, -0.929, -1.299, -0.089],
                  [-0.319, -0.244, -0.362, -0.291, 0.060, 0.281]]),
    ],
}

prismatic_joint_limits_a = {
    'joint_limits': 2 * length_links['prismatic'][2][0]
}
prismatic_joint_limits_b = {
    'joint_limits': 2 * length_links['prismatic'][2][2]
}

system_paras = [
    # Prismatic
    # identity
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][0],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': np.eye(2)[[1, 0]],
     'iname': 'Prismatic-id',
     'name': slug('prismatic', 0, 0),
     'limits': prismatic_joint_limits_a,
     },
    # square-imbalanced
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][0],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': interfaces[2][3],
     'iname': 'Prismatic-square-imbalanced',
     'name': slug('prismatic', 0, 1),
     'limits': prismatic_joint_limits_a,
     },
    # rectangular-small
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][0],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': interfaces[2][4][:, 1:],
     'iname': 'Prismatic-rectangular-small',
     'name': slug('prismatic', 0, 2),
     'limits': prismatic_joint_limits_a,
     },
    # rectangular
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][0],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': interfaces[2][5],
     'iname': 'Prismatic-rectangular',
     'name': slug('prismatic', 0, 3),
     'limits': prismatic_joint_limits_a,
     },
    # identity
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][2],
     'mass': common_paras[2][0],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': np.eye(2)[[1, 0]],
     'iname': 'Prismatic-id-R',
     'name': slug('prismatic', 1, 0),
     'limits': prismatic_joint_limits_b,
     },
    # square-imbalanced
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][2],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': interfaces[2][3],
     'iname': 'Prismatic-square-imbalanced-R',
     'name': slug('prismatic', 1, 1),
     'limits': prismatic_joint_limits_b,
     },
    # rectangular-small
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][2],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': interfaces[2][4][:, 1:],
     'iname': 'Prismatic-rectangular-small-R',
     'name': slug('prismatic', 1, 2),
     'limits': prismatic_joint_limits_b,
     },
    # rectangular
    {'system': RobotPrismatic2LinkILinear,
     'lengths': length_links['prismatic'][2][2],
     'mass': {
         'inertia_links': common_paras[2][0]['inertia_links'],
         'mass_links': common_paras[2][0]['mass_links'] / 2,
     },
     'i_G': interfaces[2][5],
     'iname': 'Prismatic-rectangular-R',
     'name': slug('prismatic', 1, 3),
     'limits': prismatic_joint_limits_b,
     },
    # Rotational2
    # identity
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][0],
     'mass': common_paras[2][0],
     'i_G': np.eye(2),
     'iname': 'Rotational2-id',
     'name': slug('rotational2', 0, 4),
     },
    # square-imbalanced
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][0],
     'mass': common_paras[2][0],
     'i_G': interfaces[2][2],
     'iname': 'Rotational2-square-imbalanced',
     'name': slug('rotational2', 0, 5),
     },
    # rectangular-small
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][0],
     'mass': common_paras[2][0],
     'i_G': interfaces[2][4][:, 1:],
     'iname': 'Rotational2-rectangular-small',
     'name': slug('rotational2', 0, 6),
     },
    # rectangular
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][0],
     'mass': common_paras[2][0],
     'i_G': interfaces[2][5],
     'iname': 'Rotational2-rectangular',
     'name': slug('rotational2', 0, 7),
     },
    # cannot reach inner part
    # identity
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][2],
     'mass': common_paras[2][1],
     'i_G': np.eye(2),
     'iname': 'Rotational2-id-R',
     'name': slug('rotational2', 2, 4),
     },
    # square-imbalanced
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][2],
     'mass': common_paras[2][1],
     'i_G': interfaces[2][2],
     'iname': 'Rotational2-square-imbalanced-R',
     'name': slug('rotational2', 2, 5),
     },
    # rectangular-small
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][2],
     'mass': common_paras[2][1],
     'i_G': interfaces[2][4][:, 1:],
     'iname': 'Rotational2-rectangular-small-R',
     'name': slug('rotational2', 2, 6),
     },
    # rectangular
    {'system': RobotRotational2LinkILinear,
     'lengths': length_links['rotational'][2][2],
     'mass': common_paras[2][1],
     'i_G': interfaces[2][5],
     'iname': 'Rotational2-rectangular-R',
     'name': slug('rotational2', 2, 7),
     },
    # Rotational3Link
    # identity
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][0],
     'mass': common_paras[3][0],
     'i_G': np.eye(3)[[1, 0, 2]],
     'iname': 'Rotational3-id',
     'name': slug('rotational3', 3, 8),
     },
    # square-imbalanced
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][0],
     'mass': common_paras[3][0],
     'i_G': interfaces[3][3],
     'iname': 'Rotational3-square-imbalanced',
     'name': slug('rotational3', 3, 9),
     },
    # rectangular-small
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][0],
     'mass': common_paras[3][0],
     'i_G': interfaces[3][5][:, 2:],
     'iname': 'Rotational3-rectangular-small',
     'name': slug('rotational3', 3, 10),
     },
    # rectangular
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][0],
     'mass': common_paras[3][0],
     'i_G': interfaces[3][4],
     'iname': 'Rotational3-rectangular',
     'name': slug('rotational3', 3, 11),
     },
    # identity - cannt reach inner part
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][2],
     'mass': common_paras[3][1],
     'i_G': np.eye(3)[[1, 0, 2]],
     'iname': 'Rotational3-id-R',
     'name': slug('rotational3', 4, 8),
     },
    # square-imbalanced,
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][2],
     'mass': common_paras[3][1],
     'i_G': interfaces[3][3],
     'iname': 'Rotational3-square-imbalanced-R',
     'name': slug('rotational3', 4, 9),
     },
    # rectangular-small
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][2],
     'mass': common_paras[3][1],
     'i_G': interfaces[3][5][:, 2:],
     'iname': 'Rotational3-rectangular-small-R',
     'name': slug('rotational3', 4, 10),
     },
    # rectangular
    {'system': RobotRotational3LinkILinear,
     'lengths': length_links['rotational'][3][2],
     'mass': common_paras[3][1],
     'i_G': interfaces[3][4],
     'iname': 'Rotational3-rectangular-R',
     'name': slug('rotational3', 4, 11),
     },
]

# systematically assemble all systems
systems = []
internal_names = {}

for paras in system_paras:
    # keep track of name mapping
    internal_names.update({paras['name']: paras['iname']})
    p = dict(paras['mass'])
    p.update({'length_links': paras['lengths']})
    p.update({'i_G': paras['i_G']})
    if 'limits' in paras.keys():
        p.update(paras['limits'])
    systems.append({
        'name': paras['name'],
        'parameters': p,
        'system': paras['system'],
    })


def gset_evaluation_seed(sysid, repetition):
    np.random.seed(sysid * 71993 + repetition * 999331)
    return np.random.randint(2**32 - 1)
