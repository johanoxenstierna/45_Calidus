import P


def calidus_gi_():

    gi = {}
    gi['r'] = 0
    gi['pi_offset'] = 0
    gi['speed_gi'] = 0

    return gi


def black_gi_():  # 20

    gi = {}
    gi['scale'] = 0.26
    gi['alpha'] = 0.99
    gi['zorder'] = 1000

    if P.REAL_SCALE:
        gi['scale'] = 0.03

    return gi


def sun_gi_():  # 20

    gi = {}
    gi['scale'] = 0.26
    gi['alpha'] = 0.99
    gi['zorder'] = 1001

    if P.REAL_SCALE:
        gi['scale'] = 0.02

    return gi


def red_gi_():  # 20

    gi = {}
    gi['scale'] = 0.3
    gi['pi_offset'] = 0
    gi['speed_gi'] = 1.5
    gi['tilt'] = 0
    gi['max_alpha'] = 0.8
    gi['min_alpha'] = 0.1
    gi['zorder'] = 1002

    return gi


def mid_gi_():  # 20

    gi = {}
    gi['scale'] = 0.3
    gi['pi_offset'] = 0
    gi['speed_gi'] = -2
    gi['tilt'] = 0
    gi['max_alpha'] = 0.6
    gi['min_alpha'] = 0.1
    gi['zorder'] = 1003

    return gi


def light_gi_():  # 20

    gi = {}
    gi['scale'] = 0.3
    gi['pi_offset'] = 0
    gi['speed_gi'] = 2
    gi['tilt'] = 0
    gi['max_alpha'] = 0.3
    gi['min_alpha'] = 0.2
    gi['zorder'] = 1004

    return gi


def h_red_gi_():  # 20

    gi = {}
    gi['scale'] = 0.27
    gi['pi_offset'] = 0
    gi['speed_gi'] = 4
    gi['tilt'] = 0
    gi['max_alpha'] = 0.15
    gi['min_alpha'] = 0.04
    gi['zorder'] = 1002

    return gi


def h_mid_gi_():  # 20

    gi = {}
    gi['scale'] = 0.27
    gi['pi_offset'] = 0
    gi['speed_gi'] = -4
    gi['tilt'] = 0
    gi['max_alpha'] = 0.15
    gi['min_alpha'] = 0.04
    gi['zorder'] = 1001

    return gi


def h_light_gi_():  # 20

    gi = {}
    gi['scale'] = 0.26
    gi['pi_offset'] = 0
    gi['speed_gi'] = 5
    gi['tilt'] = 0
    gi['max_alpha'] = 0.18
    gi['min_alpha'] = 0.05
    gi['zorder'] = 1005

    return gi