"""
Matplotlib animation of projectiles, waves and clouds

Only create 1 stns and Y map (or whatever TF its called).

YT still needed!

-*Foam f twist after h (probably just shift)
-zorder.
-increase c
-rotate_around
-scale
-tune foam in 1 and 2
-MORERandom b foam flying into air.
"""

import matplotlib.pyplot as plt
from sys import platform
if platform == 'win32':
    plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

import numpy as np
import random
random.seed(3)  # ONLY HERE
np.random.seed(3)  # ONLY HERE
import time
import pickle

import matplotlib.animation as animation
from src.gen_objects import GenObjects
from src.load_pics import load_pics
from src.genesis import _genesis
from src.ani_helpers import *
import P as P

Writer = animation.writers['ffmpeg']
writer = Writer(fps=P.FPS, bitrate=3600)  #, extra_args=['-vcodec', 'h264_nvenc'])

# fig = plt.figure(figsize=(16, 9), dpi=1920/16)
fig = plt.figure(figsize=(12.8, 7.2))
# fig = plt.figure(figsize=(10, 6))
if P.WRITE != 0:
    fig, ax_b = plt.subplots(figsize=(19.2, 10.8), dpi=100)

ax_b = plt.gca()
# fig, ax_b = plt.subplots()

fig.subplots_adjust(bottom=0)
fig.subplots_adjust(top=1)
fig.subplots_adjust(right=1)
fig.subplots_adjust(left=0)

axs0 = []
axs1 = []

g = GenObjects()
g.pics = load_pics()
g.gis = _genesis()
g.real_scale()
g.gen_backgr(ax_b)
o0calidus = g.gen_calidus(ax_b)
g.gen_planets_moons(o0calidus)

if 'Rockets' in P.OBJ_TO_SHOW:
    R = g.gen_rockets(o0calidus)

with open('./vids/gis', 'wb') as f:
    pickle.dump(g.gis, f)

# for o1_id, o1 in o0calidus.O1.items():
#     _ = o1.ani_update_step(ax_b, axs0)  # doesnt help

plt.gca().invert_yaxis()

print("Done prep =============================\n")
'''VIEWER ==========================================='''

def init():
    return axs0 #+ axs1


def animate(i):

    prints = "i: " + str(i) + "  len_axs0: " + str(len(axs0)) + "  len_axs1: " + str(len(axs1))

    for o1_id, o1 in o0calidus.O1.items():  # this is where most of the CPU time goes
        if o1.drawn in [1, 2]:  # either start or continue
            o1.set_clock(i)  # sets drawn
            _ = o1.ani_update_step(ax_b, axs0)
            if o1.drawn == 2:  # NOW, THESE OBJ WILL JUST DISSAPEAR WHEN IND REACHED
                set_data_O1(o1, ax_b)

    if 'Rockets' in P.OBJ_TO_SHOW:
        for rocket in R:
            if i == rocket.frame_ss[0]:
                prints += "  adding rocket"
                rocket.drawn = 1

            # for o1_id, o1 in o0.O1.items():  # this loop is super fast
            #
            #     if i in o1.gi['init_frames']:
            #
            #         if o1.drawn == 0:
            #             prints += "  adding f"
            #             exceeds_frame_max, how_many = o1.check_frame_max(i, o1.gi['frames_tot'])
            #             if exceeds_frame_max == True:
            #                 print("EXCEEDS MAX. This means objects at end of animation will go faster. \n")
            #                 o1.gi['frames_tot'] = how_many
            #
            #             # o1.dyn_gen()
            #             o1.drawn = 1  # this variable can serve multiple purposes (see below, and in set_clock)
            #             o1.set_frame_ss(i, o1.gi['frames_tot'])  # uses AbstractSSS
            #
            #             ''' EVIL BUG HERE. An o1 cannot be allowed to init new O2 children if old children
            #             are still being drawn!!! THIS MEANS o1 FRAMES_TOT MUST > O2 FRAMES TOT
            #             UPDATE: Try releasing o1 once max frame stop of its sps reached. '''
            #
            #         else:
            #             prints += "  no free o1"

        for rocket in R:
            if rocket.drawn in [1, 2]:
                rocket.set_clock(i)  # sets drawn
                index_removed = rocket.ani_update_step(ax_b, axs0)
                if rocket.drawn == 2:
                    set_data_rocket(rocket, ax_b)
                elif rocket.drawn == 3:
                    prints += "  removing rocket"
                    decrement_all_index_axs0(index_removed, R)

    if i % 100 == 0:
        print(prints)

    return axs0  # + axs1  #


sec_vid = ((P.FRAMES_STOP - P.FRAMES_START) / P.FPS)
min_vid = ((P.FRAMES_STOP - P.FRAMES_START) / P.FPS) / 60
print("len of vid: " + str(sec_vid) + " s" + "    " + str(min_vid) + " min")

start_t = time.time()
ani = animation.FuncAnimation(fig, animate,
                              frames=range(P.FRAMES_START, P.FRAMES_STOP),
                              blit=True, interval=1, init_func=init,
                              repeat=False)  # interval only affects live ani. blitting seems to make it crash

if P.WRITE == 0:
    plt.show()
else:
    ani.save('./vids/vid_' + str(P.WRITE) + '_' + P.VID_SINGLE_WORD + '_' + '.mp4', writer=writer)

    # ani.save('./vids/vid_' + str(WRITE) + '.mov',
    #          codec="png",
    #          dpi=100,
    #          fps=40,
    #          bitrate=3600)

    # THIS ONE!!!
    # ani.save('./vids/vid_' + str(WRITE) + '.mov',
    #          codec="png",
    #          dpi=100,
    #          fps=FPS,
    #          bitrate=3600,
    #          savefig_kwargs={"transparent": True, "facecolor": "none"})

tot_time = round((time.time() - start_t) / 60, 4)
print("minutes to make animation: " + str(tot_time) + " |  min_gen/min_vid: " + str(tot_time / min_vid))  #
