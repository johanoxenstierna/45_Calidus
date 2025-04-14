import os

import numpy as np

import P as P
from matplotlib.pyplot import imread

def load_pics():
    """LOADS BGR
    ch needed to see if smoka_hardcoded is used """

    pics = {}
    pics['O0'] = {}

    # if P.MAP_DIMS[0] == 1280:
    #     pics['backgr'] = imread('./pictures/backgr_b.png')  # 482, 187
    #     # pics['backgr'] = np.flipud(pics['backgr'])
    # elif P.MAP_DIMS[0] == 1920:
    pics['backgr_b'] = imread('./pictures/backgr_b.png')
    pics['backgr_55'] = imread('./pictures/backgr_55.jpg')

    # if 'Calidus' in P.OBJ_TO_SHOW:
    pics['0_black'] = imread('./pictures/Calidus0/0_cal/0_black.png')
    pics['0_sun'] = imread('./pictures/Calidus1/0_cal/0_sunR.png')
    pics['0_red'] = imread('./pictures/Calidus1/0_cal/0_red_f.png')
    pics['0_mid'] = imread('./pictures/Calidus1/0_cal/0_mid.png')
    pics['0_light'] = imread('./pictures/Calidus1/0_cal/0_light.png')
    pics['0h_red'] = imread('./pictures/Calidus1/0_cal/0h_red.png')
    pics['0h_mid'] = imread('./pictures/Calidus1/0_cal/0h_mid.png')
    pics['0h_light'] = imread('./pictures/Calidus1/0_cal/0h_light.png')

    if 'Ogun' in P.OBJ_TO_SHOW:
        pics['Ogun'] = [imread('./pictures/Calidus1/planets/1_OgunD.png'),
                          imread('./pictures/Calidus1/planets/1_Ogun.png'),
                          imread('./pictures/Calidus1/planets/1_OgunL.png')]

    if 'Venus' in P.OBJ_TO_SHOW:
        pics['Venus'] = [imread('./pictures/Calidus1/planets/2_VenusD.png'),
                          imread('./pictures/Calidus1/planets/2_Venus.png'),
                          imread('./pictures/Calidus1/planets/2_VenusL.png')]

    # ------------------------

    if 'Nauvis' in P.OBJ_TO_SHOW:
        pics['Nauvis'] = [imread('./pictures/Calidus1/planets/3_NauvisD.png'),
                          imread('./pictures/Calidus1/planets/3_Nauvis.png'),
                          imread('./pictures/Calidus1/planets/3_NauvisL.png')]

    if 'GSS' in P.OBJ_TO_SHOW:
        pics['GSS'] = [imread('./pictures/Calidus1/planets/3_GSSD.png'),
                       imread('./pictures/Calidus1/planets/3_GSS.png')]
        '''IN o1.py: OBS _s.scale = np.ones((P.FRAMES_TOT_BODIES,))'''

    if 'Molli' in P.OBJ_TO_SHOW:
        pics['Molli'] = [imread('./pictures/Calidus1/planets/3_MolliD.png'),
                         imread('./pictures/Calidus1/planets/3_Molli.png'),
                         imread('./pictures/Calidus1/planets/3_MolliL.png')]

    # ----------------------

    if 'Mars' in P.OBJ_TO_SHOW:
        pics['Mars'] = [imread('./pictures/Calidus1/planets/4_MarsD.png'),
                          imread('./pictures/Calidus1/planets/4_Mars.png'),
                          imread('./pictures/Calidus1/planets/4_MarsL.png')]

    if 'Jupiter' in P.OBJ_TO_SHOW:
        pics['Jupiter'] = [imread('./pictures/Calidus1/planets/6_JupiterD.png'),
                          imread('./pictures/Calidus1/planets/6_Jupiter.png'),
                          imread('./pictures/Calidus1/planets/6_JupiterL.png')]

    if 'Everglade' in P.OBJ_TO_SHOW:
        pics['Everglade'] = [imread('./pictures/Calidus1/planets/6_EvergladeD.png'),
                          imread('./pictures/Calidus1/planets/6_Everglade.png'),
                          imread('./pictures/Calidus1/planets/6_EvergladeL.png')]

    if 'Petussia' in P.OBJ_TO_SHOW:

        pics['Petussia'] = [imread('./pictures/Calidus1/planets/6_PetussiaD.png'),
                             imread('./pictures/Calidus1/planets/6_Petussia.png'),
                             imread('./pictures/Calidus1/planets/6_PetussiaL.png')]

    if 'Astro0' in P.OBJ_TO_SHOW:
        pics['Astro0'] = imread('./pictures/Calidus1/z_Astro0_masked.png')

    if 'Astro0b' in P.OBJ_TO_SHOW:
        pics['Astro0b'] = [imread('./pictures/Calidus1/planets/3_GSSD.png'),
                           imread('./pictures/Calidus1/planets/3_GSS.png')]

    if 'Saturn' in P.OBJ_TO_SHOW:
        pics['Saturn'] = [imread('./pictures/Calidus1/planets/7_SaturnD.png'),
                          imread('./pictures/Calidus1/planets/7_Saturn.png'),
                          imread('./pictures/Calidus1/planets/7_SaturnL.png')]

    if 'Uranus' in P.OBJ_TO_SHOW:
        pics['Uranus'] = [imread('./pictures/Calidus1/planets/8_UranusD.png'),
                          imread('./pictures/Calidus1/planets/8_Uranus.png'),
                          imread('./pictures/Calidus1/planets/8_UranusL.png')]

    if 'Neptune' in P.OBJ_TO_SHOW:
        pics['Neptune'] = [imread('./pictures/Calidus1/planets/9_NeptuneD.png'),
                          imread('./pictures/Calidus1/planets/9_Neptune.png'),
                          imread('./pictures/Calidus1/planets/9_NeptuneL.png')]



    return pics
