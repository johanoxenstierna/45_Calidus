import P

# from sh_info import shInfoAbstract, _0_info
from O0s_info import calidus_gi, R_gi
from O0s_info.O1 import a_ogun_gi, b_venus_gi, c_nauvis_gi, d_mars_gi, f_jupiter_gi, z_astro0_gi


def _genesis():

    '''
    Creates instance of each info and stores in dict
    '''

    gis = {}

    # if 'Calidus' in P.OBJ_TO_SHOW:  # EXPL
    gis['Calidus'] = calidus_gi.calidus_gi_()
    gis['0_black'] = calidus_gi.black_gi_()
    gis['0_sun'] = calidus_gi.sun_gi_()
    gis['0_red'] = calidus_gi.red_gi_()
    gis['0_mid'] = calidus_gi.mid_gi_()
    gis['0_light'] = calidus_gi.light_gi_()
    gis['0h_red'] = calidus_gi.h_red_gi_()
    gis['0h_mid'] = calidus_gi.h_mid_gi_()
    gis['0h_light'] = calidus_gi.h_light_gi_()
    # gis['0_red'] = calidus_gi.red_gi_()
    # gis['0_red'] = calidus_gi.red_gi_()
    # gis['0_red'] = calidus_gi.red_gi_()

    if 'Ogun' in P.OBJ_TO_SHOW:
        gis['Ogun'] = a_ogun_gi.ogun_gi_()

    if 'Venus' in P.OBJ_TO_SHOW:
        gis['Venus'] = b_venus_gi.venus_gi_()

    if 'Nauvis' in P.OBJ_TO_SHOW:
        gis['Nauvis'] = c_nauvis_gi.nauvis_gi_()

    if 'GSS' in P.OBJ_TO_SHOW:
        gis['GSS'] = c_nauvis_gi.gss_gi_()

    if 'Molli' in P.OBJ_TO_SHOW:
        gis['Molli'] = c_nauvis_gi.molli_gi_()

    if 'Mars' in P.OBJ_TO_SHOW:
        gis['Mars'] = d_mars_gi.mars_gi_()

    if 'Jupiter' in P.OBJ_TO_SHOW:
        gis['Jupiter'] = f_jupiter_gi.jupiter_gi_()

    if 'Everglade' in P.OBJ_TO_SHOW:
        gis['Everglade'] = f_jupiter_gi.everglade_gi_()

    if 'Petussia' in P.OBJ_TO_SHOW:
        gis['Petussia'] = f_jupiter_gi.petussia_gi_()

    if 'Astro0' in P.OBJ_TO_SHOW:
        gis['Astro0'] = z_astro0_gi.astro0_gi_()

    if 'Astro0b' in P.OBJ_TO_SHOW:
        gis['Astro0b'] = z_astro0_gi.astro0b_gi_()

    if 'Rockets' in P.OBJ_TO_SHOW:
        gis['Rockets'] = R_gi.R_gi_()

    for gi_id, gi in gis.items():

        if 'speed_gi' in gi:
            gi['speed_gi'] *= P.SPEED_MULTIPLIER
            if P.WRITE > 90:
                assert(gi['speed_gi'] < 30)
        if 'speed_max' in gi:
            gi['speed_max'] *= P.SPEED_MULTIPLIER
            if P.WRITE > 90:
                assert(gi['speed_max'] < 30)

    # for rocket_gi in gis['Rockets']:
    #     rocket_gi['speed_max'] *= P.SPEED_MULTIPLIER

    return gis