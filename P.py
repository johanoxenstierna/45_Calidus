
'''These are dimensions for backr pic. Has a huge impact on cpu-time'''
MAP_DIMS = (1920, 1080)  #(233, 141)small  # NEEDED FOR ASSERTIONS

COMPLEXITY = 0  # OBS REMEMBER SET P.FRAMES in waves_helper

FRAMES_START = 0
FRAMES_STOP = 9000
FRAMES_TOT_BODIES = FRAMES_STOP - 25
FRAMES_TOT = FRAMES_STOP - FRAMES_START

FPS = 100  # 17:31
SPEED_MULTIPLIER = 0.4 # 0.4  latest: 0.4
WRITE = 0
REAL_SCALE = 0

OBJ_TO_SHOW = []
OBJ_TO_SHOW.append('Calidus')  # JUST USE ALPHA in main
OBJ_TO_SHOW.append('Ogun')
OBJ_TO_SHOW.append('Venus')
OBJ_TO_SHOW.append('Nauvis')
OBJ_TO_SHOW.append('GSS')
OBJ_TO_SHOW.append('Molli')
OBJ_TO_SHOW.append('Mars')
# OBJ_TO_SHOW.append('Astro0')
OBJ_TO_SHOW.append('Astro0b')
OBJ_TO_SHOW.append('Jupiter')
OBJ_TO_SHOW.append('Everglade')
OBJ_TO_SHOW.append('Petussia')
OBJ_TO_SHOW.append('Rockets')

VID_SINGLE_WORD = '_'
for word in OBJ_TO_SHOW:
    VID_SINGLE_WORD += word[0] + '_'

