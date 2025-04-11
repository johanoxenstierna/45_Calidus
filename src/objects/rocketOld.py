import numpy as np
# import scipy.stats
from scipy.stats import beta, norm
# from scipy.stats._multivariate import multivariate_normal

import P as P
from src.m_functions import *
from src.objects.abstract import AbstractObject, AbstractSSS


# from src.wave_funcs import *


class Rocket(AbstractObject, AbstractSSS):

    def __init__(_s, init_frame, gi, p0, p1, destination_type):
        AbstractObject.__init__(_s)

        _s.init_frame = init_frame
        _s.id = str(init_frame)
        _s.gi = gi
        _s.zorder = 1000
        _s.type = 'rocket'
        _s.p0 = p0
        _s.p1 = p1
        _s.destination_type = destination_type

        AbstractSSS.__init__(_s)

        '''Set after motion decided'''
        _s.xy = None
        _s.xy_t = None
        _s.vxy = None
        _s.zorders = None
        _s.alphas = None
        _s.color = None
        _s.ok_rocket = ''

        kp = 0.99  # compulsory
        ki = 0.01  # 0.02 goes crazy when ob
        kd = 0.05  # 0.01
        pos_factor = 1
        _s.pid_x = _s.PIDController(kp=pos_factor * kp, ki=pos_factor * ki, kd=pos_factor * kd)
        _s.pid_y = _s.PIDController(kp=pos_factor * kp, ki=pos_factor * ki, kd=pos_factor * kd)
        _s.pid_vx = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vy = _s.PIDController(kp=kp, ki=ki, kd=kd)

        # _s.max_turn_rate = 0.1 * np.pi

    def gen_rocket_motion(_s):

        _s.takeoff()
        _s.mid_flight()
        _s.landing()

        _s.zorders = np.asarray(_s.zorders)

        # _s.zorders = np.full((len(_s.xy)), fill_value=9999)
        # _s.alphas = norm.pdf(x=np.arange(0, len(_s.xy)), loc=len(_s.xy) / 2, scale=len(_s.xy) / 5)
        # y_range_min, y_range_max = 0.3, 0.7
        # if P.WRITE != 0:
        #     y_range_min, y_range_max = 0.3, 0.7
        # _s.alphas = min_max_normalize_array(_s.alphas, y_range=[y_range_min, y_range_max])  # 0.2, 0.7

        if P.WRITE != 0:
            _s.gen_color()
            if len(_s.xy) > 400 and P.WRITE != 0 and \
                    _s.p0.id not in ['Jupiter', 'Everglade', 'Petussia', 'Astro0b'] and \
                    _s.p1.id not in ['Jupiter', 'Everglade', 'Petussia', 'Astro0b']:
                _s.gen_rand_color()
        else:
            _s.color = np.linspace(1, 0.99, len(_s.xy))

        # _s.color[first_frame_land:] = 0.99
        # _s.alphas[first_frame_land:] = 0.99

        _s.set_frame_ss(_s.init_frame, len(_s.xy))

        print("id: " + str(_s.id).ljust(5) +
              # " | num_frames: " + str(i).ljust(5) +
              # " | speed_max: " + str(speed_max)[0:4] +
              # " | attempts_tot: " + str(attempt + 1).ljust(4) +
              " | ok_rocket: " + str(_s.ok_rocket).ljust(20)
              )

    def takeoff(_s):

        # TODO: GEN

        num_frames = 50  # int(200 / (_s.gi['speed_max']))
        # num_frames = 20
        # if P.WRITE != 0:
        #     num_frames = int(200)

        '''
        TODO HERE: DO IT AS MUCH AS O1 AS POSSIBLE. NO DIRECTIONS IN XY_T
        ALSO FIX ZORDERS AFTER 
        '''
        if _s.destination_type == 'inter':
            # r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5, _s.p0.centroids[_s.init_frame] * _s.p0.gi['centroid_mult'], num_frames)
            r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5,
                            _s.p0.centroids[_s.init_frame] * _s.p0.gi['centroid_mult'], num_frames)
            # r = 50  # 50 GIVES EXIT SPEED=1
            num_rot = 1
        else:  # _s.destination_type == 'orbit':
            # r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5, _s.p0.centroids[_s.init_frame] * 4, num_frames)
            r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5, _s.p1.gi['r'], num_frames)
            num_rot = 1  # gonna mess up if weird number here

        y_squeeze = 0.08
        xy_t = np.zeros((num_frames, 2), dtype=np.float32)
        xy_t[:, 0] = np.sin(np.linspace(0, num_rot * 2 * np.pi, num_frames)) * r
        xy_t[:, 1] = -np.cos(np.linspace(0, num_rot * 2 * np.pi, num_frames)) * r * y_squeeze

        # # Apply tilt by rotating the coordinates
        direction_to_p1 = _s.p1.xy[_s.init_frame + num_frames - 1] - _s.p0.xy[
            _s.init_frame + num_frames - 1]  # p1 must be first
        direction_to_p1 /= np.linalg.norm(direction_to_p1)  # Normalize 0-1
        tilt = np.arctan2(direction_to_p1[1], direction_to_p1[0])

        cos_theta = np.cos(tilt)
        sin_theta = np.sin(tilt)
        x_rot = cos_theta * xy_t[:, 0] - sin_theta * xy_t[:, 1]
        y_rot = sin_theta * xy_t[:, 0] + cos_theta * xy_t[:, 1]

        xy_t_rot = np.copy(xy_t)  # OBS: _s.xy_t no longer rotated!
        xy_t_rot[:, 0] = x_rot
        xy_t_rot[:, 1] = y_rot

        _s.xy = _s.p0.xy[_s.init_frame:_s.init_frame + num_frames] + xy_t_rot
        _s.p1_xy_temp0 = _s.p1.xy[_s.init_frame + len(_s.xy) - 1]

        _s.alphas = np.linspace(0.3, 0.5, num_frames)

        _s.zorders = np.full((num_frames,), dtype=int, fill_value=10)
        vxy_t = np.gradient(xy_t, axis=0)
        inds_neg = np.where(vxy_t[:, 0] >= 0)[0]
        _s.zorders[inds_neg] *= -10
        _s.zorders += _s.p0.zorders[_s.init_frame:_s.init_frame + num_frames]

        speed_i_debug = np.linalg.norm(np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]]))

        aa = 5

    def mid_flight(_s):

        '''
        '''

        vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])  # dictates max speed currently
        v0 = np.copy(vxy_i)

        # max_speed = np.linalg.norm(vxy_i) * 1
        num_frames = int(0.8 * _s.gi['frames_max'])  # No need for speed here cuz break
        num_frames_acc = 50  # int(0.1 * num_frames)

        use_v0 = np.zeros((num_frames,))
        use_v0[0:num_frames_acc] = np.linspace(1, 0, num_frames_acc)
        use_pid = np.full((num_frames,), fill_value=1.)
        use_pid[0:num_frames_acc] = np.linspace(0, 1, num_frames_acc)

        '''OBS len(xy) - 1 GIVES LAST xy ADDED i.e. CURRENT, BUT LOOP BELOW SHOULD USE NEXT VALUES ie range(1, num)
        BCS OTHERWISE AFTER xy.append() the latest xy will be one step ahead of p1, 
        and that prevents clean way to retrieve values after loop
        '''
        num_temp = 10
        p1_xy = _s.p1.xy[
                _s.init_frame + len(_s.xy) - 1 + num_temp:_s.init_frame + len(_s.xy) - 1 + num_frames + num_temp]
        p1_vxy = _s.p1.vxy[
                 _s.init_frame + len(_s.xy) - 1 + num_temp:_s.init_frame + len(_s.xy) - 1 + num_frames + num_temp]
        # p1_centroids = _s.p1.centroids[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]

        '''Num frames forward to predict'''
        dist_cond0 = 200
        dist_cond1 = 10
        pdf_forward = beta.pdf(x=np.arange(0, dist_cond0), loc=dist_cond1, a=6, b=10, scale=dist_cond0)
        pdf_forward = min_max_normalize_array(pdf_forward, y_range=[0, 30])
        num_forward = 0

        speed_dist = np.linspace(1.5, _s.p1.speed_max * 3, num=dist_cond0)
        speed_max = _s.p1.speed_max * 3  # used until dist_cond0

        xy_i = np.copy(_s.xy[-1, :])
        xy = []
        zorders = []

        for i in range(1, num_frames):  # where will you be next frame? Then I will append myself to that.

            error_x = p1_xy[i + num_forward, 0] - xy_i[0]
            error_y = p1_xy[i + num_forward, 1] - xy_i[1]
            error_vx = p1_vxy[i + num_forward, 0] - vxy_i[0]
            error_vy = p1_vxy[i + num_forward, 1] - vxy_i[1]

            _pid_x = _s.pid_x.update(error_x)
            _pid_y = _s.pid_y.update(error_y)
            _pid_vx = _s.pid_vx.update(error_vx)
            _pid_vy = _s.pid_vy.update(error_vy)
            _pid_speed = np.linalg.norm([_pid_vx, _pid_vy])

            vxy_new = np.array([_pid_x + _pid_vx, _pid_y + _pid_vy])
            # speed = np.linalg.norm(vxy_new)  # Compute the speed (magnitude of velocity vector)
            # if speed > speed_max:
            #     vxy_new = vxy_new / speed * speed_max  # first div gives 0-1
            # elif speed < 1:
            #     vxy_new = vxy_new / speed

            vxy_i = use_v0[i] * v0 + use_pid[i] * vxy_new
            xy_i += vxy_i
            xy.append(np.copy(xy_i))  # xy is now
            zorders.append(9999)

            dist = np.linalg.norm(p1_xy[i] - xy_i)
            if dist < dist_cond0:
                num_forward = 0  # int(pdf_forward[int(dist)])
                speed_max = speed_dist[int(dist)]
                if dist < dist_cond1:
                    break

        _s.xy = np.concatenate((_s.xy, np.asarray(xy)))
        _s.alphas = np.concatenate((_s.alphas, np.full((len(xy),), fill_value=0.5)))
        _s.zorders = np.concatenate((_s.zorders, np.asarray(zorders)))
        p1_xy_debug = _s.p1.xy[_s.init_frame + len(_s.xy) - 1]  # THIS IS NOW THE CURRENT VALUE
        speed_i_debug = np.linalg.norm(np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]]))

        ads = 5

    def landing(_s):

        xy_i = np.copy(_s.xy[-1, :])
        vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])
        speed_i_debug = np.linalg.norm(vxy_i)
        xy_i += vxy_i  # can now be used as first value
        dist_xy = _s.p1.xy[_s.init_frame + len(_s.xy)] - xy_i  # p1 must be first
        # vxy_lagged = np.array([_s.xy[-30, 0] - _s.xy[-36, 0], _s.xy[-30, 1] - _s.xy[-36, 1]])

        # speed_i = np.linalg.norm(vxy_i)
        # speed_entry = 1 * P.SPEED_MULTIPLIER  # MAYBE PUT IN o1 gi
        #
        # dist_xy = _s.p1.xy[_s.init_frame + len(_s.xy)] - xy_i  # p1 must be first
        # dist = np.linalg.norm(dist_xy)
        #
        # num_frames_dec0 = abs(speed_i - speed_entry) * 10 / P.SPEED_MULTIPLIER
        # num_frames_dec1 = dist * 0.7
        # num_frames_dec = max(3, int(num_frames_dec0 + num_frames_dec1))
        #
        # '''
        # STRAIGHT LINE TO STATIONARY ST TARGET IN NEXT FRAME
        # start and stop in linspace below are both for NEXT frame.
        # OBS The stopping position must be 1 frame away from p1 so that landing can start at exact origin
        # Also, extra needed for GSS cuz parent is also moving
        # '''
        #
        # xy_dec = np.zeros((num_frames_dec, 2), dtype=np.float32)
        # p1_parent_vxy = _s.p1.parent.vxy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_dec]
        # zz_stop = _s.p1.xy[_s.init_frame + len(_s.xy)] #+ _s.p1.parent.xy_t[_s.init_frame + len(_s.xy) + num_frames_dec]
        # xy_dec[:, 0] = np.linspace(start=xy_i[0], stop=zz_stop[0] + np.sum(p1_parent_vxy[:, 0]), num=num_frames_dec)  # BOTH ARE NEXT HERE
        # xy_dec[:, 1] = np.linspace(start=xy_i[1], stop=zz_stop[1] + np.sum(p1_parent_vxy[:, 1]), num=num_frames_dec)
        # # xy_dec[:, 0] = np.linspace(start=xy_i[0], stop=zz_stop[0], num=num_frames_dec)
        # # xy_dec[:, 1] = np.linspace(start=xy_i[1], stop=zz_stop[1], num=num_frames_dec)
        # # xy_dec += _s.p1.parent.vxy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_dec]
        # xy_dec = xy_dec[0:len(xy_dec) - 1]  # OBS THE THING AFTER COLON IS UBE if xy=[0, 1, 2, 3] then this gets [0, 1, 2]
        # xy = xy_dec

        # ================================================================
        # xy_i = np.copy(_s.xy[-1, :])
        # vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])  # NEEDED FOR TILT
        # vxy_i = np.array([xy_dec[-1, 0] - xy_dec[-2, 0], xy_dec[-1, 1] - xy_dec[-2, 1]])  # NEEDED FOR TILT

        # xy_i += vxy_i  # can now be used as first value

        num_frames_t = int(200)
        num_rot = 1

        # p1_centroids = _s.p1.centroids[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]

        y_squeeze = 0.00011

        r = np.linspace(25, 6, num_frames_t)  # TODO: CHANGE TO NORMAL BCS STARTS AT ORIG

        xy_t = np.zeros((num_frames_t, 2))
        xy_t[:, 0] = np.sin(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * r
        xy_t[:, 1] = -np.cos(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * r * y_squeeze

        # # # Apply tilt by rotating the coordinates
        tilt = np.arctan2(vxy_i[1], vxy_i[0])

        cos_theta = np.cos(tilt)
        sin_theta = np.sin(tilt)
        x_rot = cos_theta * xy_t[:, 0] - sin_theta * xy_t[:, 1]
        y_rot = sin_theta * xy_t[:, 0] + cos_theta * xy_t[:, 1]

        xy_t_rot = np.copy(xy_t)  # OBS: _s.xy_t no longer rotated!
        xy_t_rot[:, 0] = x_rot
        xy_t_rot[:, 1] = y_rot

        p1_shifted = - dist_xy + _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_t]
        p1_actuall = _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_t]
        # xy = xy_t_rot + _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_t] - dist_xy
        # xy +=

        # xy_t_rot += xy_dec[-1]
        # xy = np.concatenate((xy_dec, xy_t_rot))

        # '''WEIGHTHING ST WITH ACTUAL P1 POSITION'''
        # p1_xy = _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + len(xy)]
        # use_st = np.linspace([1, 1], [0.99, 0.99], len(xy))
        # use_p1 = np.linspace([0, 0], [0.01, 0.01], len(xy))
        use_v0 = np.zeros((num_frames_t,))
        use_v0[0:50] = np.linspace(1, 0, 50)
        use_v0 = np.stack((use_v0, use_v0), axis=1)
        use_landing = np.zeros((num_frames_t,))
        use_landing[0:50] = np.linspace(0, 1, 50)
        use_landing = np.stack((use_landing, use_landing), axis=1)
        use_shifted = np.linspace(1, 0, num=len(p1_shifted))
        use_shifted = np.stack((use_shifted, use_shifted), axis=1)
        use_actuall = np.linspace(0, 1, num=len(p1_shifted))
        use_actuall = np.stack((use_actuall, use_actuall), axis=1)
        xy = xy_t_rot + (use_v0 * vxy_i)  # + use_landing * (use_shifted * p1_shifted + use_actuall * p1_actuall)

        zorders = np.full((len(xy),), dtype=int, fill_value=9999)
        _s.xy = np.concatenate((_s.xy, np.asarray(xy)))
        _s.alphas = np.concatenate((_s.alphas, np.full((len(xy),), fill_value=0.99)))
        _s.zorders = np.concatenate((_s.zorders, np.asarray(zorders)))

    def get_dummy_orbit(_s, p1_xy):
        """Get the orbit target with a shrinking radius."""
        _s.r = max(0.1, _s.r * 0.99)  # , p1_centroids[num_frames - 1], num_frames_t)
        _s.theta += 0.05
        x = p1_xy[0] + _s.r * np.cos(_s.theta)
        y = p1_xy[1] + _s.r * np.sin(_s.theta) * 0.5
        return np.array([x, y])

    def limit_turn_rate(_s, vxy_i, desired_direction, speed_min, speed_max, turn_rate_max):

        """ Limit turn rate and enforce max speed. """
        current_angle = np.arctan2(vxy_i[1], vxy_i[0])  # Flipped Y-axis fix
        desired_angle = np.arctan2(desired_direction[1], desired_direction[0])

        angle_diff = np.arctan2(np.sin(desired_angle - current_angle), np.cos(desired_angle - current_angle))

        # Apply turn rate limit
        # max_turn_step = np.clip(angle_diff, -turn_rate_max, turn_rate_max)
        # max_turn_step = angle_diff * (turn_rate_max / (abs(angle_diff) + 1e-6))  # Prevent div by zero
        max_turn_step = np.sign(angle_diff) * min(abs(angle_diff), turn_rate_max)

        new_angle = current_angle + max_turn_step

        # Maintain speed while limiting turn
        speed = np.linalg.norm(vxy_i)
        if speed > speed_max:
            speed = np.linalg.norm(vxy_i * 0.95)
        elif speed <= speed_min:
            speed = np.linalg.norm(vxy_i * 1.05)

        return np.array([np.cos(new_angle) * speed, np.sin(new_angle) * speed])  # Flipped Y correction

    def gen_color(_s):
        color = np.linspace(1, 0.99, len(_s.xy))
        indicies = np.where((_s.xy[:, 0] < 1010) & (_s.xy[:, 0] > 910) &
                            (_s.xy[:, 1] < 590) & (_s.xy[:, 1] > 490))[0]
        if len(indicies) > 10 and indicies[1] == indicies[0] + 1 and (indicies[0] + 60 < len(_s.xy)):
            pdf = -beta.pdf(x=np.arange(0, 60), a=2, b=2, loc=0, scale=60)
            pdf = min_max_normalize_array(pdf, y_range=[0, 1])
            color[indicies[0]:indicies[0] + 60] = pdf

        _s.color = color

    def gen_rand_color(_s):

        inds = np.random.randint(low=10, high=len(_s.xy) - 100, size=2)
        for ind in inds:
            num = np.random.randint(low=20, high=50)
            pdf = -beta.pdf(x=np.arange(0, num), a=2, b=2, loc=0, scale=num)
            pdf = min_max_normalize_array(pdf, y_range=[0, 1])
            _s.color[ind:ind + num] = pdf
            adf = 5

    # def p1_set_target(_s, error_x, error_y, p1_xy_, ratio_k):
    #
    #     p1t_xy_ = np.copy(p1_xy_)
    #     p1t_xy_[0] += error_x * ratio_k
    #     p1t_xy_[1] += error_y * ratio_k
    #
    #     return p1t_xy_

    class PIDController:
        def __init__(self, kp, ki, kd):
            """
            Proportional (Kp): Controls the immediate response to position and velocity errors.
            Integral (Ki): Helps eliminate steady-state errors by accounting for accumulated past errors.
            Derivative (Kd): Damps the system by reacting to the rate of change of errors.
            """
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.prev_error = 0
            self.integral = 0

        def update(self, error):
            ''' Proportional term: If error=2000 pixels'''
            proportional = self.kp * error

            # Integral term
            self.integral += error
            integral = self.ki * self.integral

            # Derivative term
            derivative = self.kd * (error - self.prev_error)
            self.prev_error = error
            return proportional + integral + derivative

    # def mid_flightOLD(_s):
    #
    #     '''
    #     Hi, this question concerns the smooth matching of positions and velocities in two dimensions using Python and PID or similar.
    #     I have two objects, p0 and p1. p0 has a position vector called p0_xy given as a numpy array with 200 positions as rows and 2 columns (x and y)
    #     The corresponding velocity vector is p0_vxy = np.gradient(p0_xy, axis=0). p0 follows an elliptical motion.
    #     p1 has a starting xy position and a starting velocity. I now wish to move p1 such that it matches the position and
    #     velocity of p0 in a smooth way. p1 is allowed to move faster than p0. The solution needs to include all cases where
    #     p1 overshoots or undershoots p1, due to small errors.
    #
    #     I think PID is the way to do it. Could you provide a simple Python example where PID is used with x and y coordinates to achieve this?
    #
    #
    #     '''
    #
    #     v0 = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])
    #
    #     num_frames_max = 100
    #     use_v0 = np.zeros((num_frames_max,))
    #     use_v0[0:50] = np.linspace(1, 0, 50)
    #     p1_xy = _s.p1.xy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames_max]  # OBS copy NEEDED!
    #     p1_centroids = _s.p1.centroids[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_max]
    #
    #     xy_i = np.copy(_s.xy[-1, :])
    #     xy = []
    #     dist0 = np.linalg.norm(_s.p1.xy[_s.init_frame + len(_s.xy) - 1] - _s.xy[-1, :])
    #
    #     for i in range(0, num_frames_max):
    #         direction_to_p1 = p1_xy[i] - xy_i  # p1 must be first
    #         direction_to_p1 /= np.linalg.norm(direction_to_p1)  # Normalize
    #         dist1 = np.linalg.norm(p1_xy[i] - xy_i)
    #
    #         if dist1 < dist0:
    #             pass
    #         vxy_i = v0 * use_v0[i]
    #         xy_i = xy_i + vxy_i
    #         xy.append(xy_i)
    #
    #     # _s.xy = np.concatenate((_s.xy, np.asarray([[444.5, 4]])))
    #     _s.xy = np.concatenate((_s.xy, np.asarray(xy)))
    #     adf = 5

    def gen_rocket_motionOLD(_s, init_frame):

        xy = []  # just init to stop intellisense
        tilt = 0  # TODO. Based on current xy dist

        speed_min = np.copy(_s.gi['speed_min'])  # 120 pixels/s
        speed_max = np.copy(_s.gi['speed_max'])  # 120 pixels/s

        ok_rocket = ''  # true if ''
        # reached_check = False
        # overshoots_tot = 0
        vxy_i = None

        num_frames = np.copy(_s.gi['min_frames'])  # VARIABLE

        if _s.id == '3343':
            adf = 5

        '''
        Rocket will always reach geostationary orbit unless it runs out of frames.
        If it reaches with 0.0001 speed doesnt matter: Landing generates orbit
        around p1. PROBLEM: What if p1 is moving away fast... not gonna reach it. 
        SOL: Add cond: if p1 is moving away, then speed at the end needs to be at least as fast as p1
        '''
        for attempt in range(1):

            xy_i = _s.p0.xy[init_frame]  # + np.random.randint(-5, 5, size=2)
            xy = [xy_i]  # KEEP THIS! DO NOT WANT TO COMBINE ndarray with normal list
            zorders = [99999]
            # speed_max = max(speed_min, speed_max + np.random.uniform(low=-0.5, high=0.5))
            speed_max = max(speed_min, speed_max + np.random.uniform(low=-0.1, high=0.1))
            v0 = _s.p0.vxy[init_frame]

            d_cond_check = False
            d_orbit_cond = 50
            d_orbit_cond_bool = False
            i_orbit = 0
            speed0land = 0
            v0land = 0
            centroid0land = 0

            # v_cond_check = False
            # overshoot = False

            d = 99999

            '''
            p1 centroid is same as radius. CentroidS because they are also scaled.
            '''

            p1_xy = np.copy(_s.p1.xy[init_frame:init_frame + num_frames])  # OBS copy NEEDED!
            p1_centroids = _s.p1.centroids[init_frame:init_frame + num_frames]

            speed = np.sin(np.linspace(0, np.pi, num_frames)) * speed_max
            # use_p0 = np.cos(np.linspace(0, np.pi / 2, num_frames))
            use_p0 = np.linspace(1, 0.01, num_frames)
            # use_p1 = np.sin(np.linspace(0, np.pi / 2, num_frames))
            use_roc = np.linspace(0.01, 1, num_frames)  # this should be 0.8 and p1 should be 0.2 at end

            # use_roc = np.sin(np.linspace(0, np.pi, num_frames))
            # use_p = -use_roc + 1

            i = 0  # just init to stop intellisense

            for i in range(1, num_frames):  # 1 needed to get previous d_xy. num_frames = min_frames

                direction_to_p1 = p1_xy[i] - xy_i
                direction_to_p1 /= np.linalg.norm(direction_to_p1)  # Normalize

                v_roc = speed[i] * direction_to_p1
                vxy_i = v0 * use_p0[i] + v_roc * use_roc[i]
                xy_i = xy_i + vxy_i

                d = np.linalg.norm(p1_xy[i] - xy_i)

                # d_xy_prev = [np.sign(p1_xy[i - 1, 0] - xy[i - 1][0]), np.sign(p1_xy[i - 1, 1] - xy[i - 1][1])]
                # d_xy_this = [np.sign(p1_xy[i, 0] - xy_i[0]), np.sign(p1_xy[i, 1] - xy_i[1])]
                # if d_xy_this[0] != d_xy_prev[0] and d_xy_this[1] != d_xy_prev[1]:
                #     TODO: dont add xy in this case

                p1_centroid_i = p1_centroids[i]

                if d_orbit_cond_bool == False:
                    if d < p1_centroid_i:
                        raise Exception("Asdfasdf")
                    if d < d_orbit_cond:
                        v0land = vxy_i  # creates copy
                        d_orbit_cond_bool = True

                if d < p1_centroid_i:  # or (d_xy_this[0] != d_xy_prev[0] and d_xy_this[1] != d_xy_prev[1]):
                    i_orbit = init_frame + i
                    speed0land = speed[i]  # creates copy
                    centroid0land = p1_centroid_i
                    d_cond_check = True
                    break
                # else:

                xy.append(xy_i)  # OBS, HENCE NOT ADDED FOR LANDING
                zorders.append(9999)

                # if i_orbit == 0:  # orbit not reached yet
                #     if d < d_orbit_cond:  # ONLY RUNS ONCE (bcs above if stops)
                #         i_orbit = init_frame + i
                #         speed0land = speed[i]  # creates copy
                #         v0land = vxy_i  # creates copy

                # if d < d_cond:  # needed to make sure rocket CAN reach destination (in case p1 is moving away).
                #     d_cond_check = True
                #     break

            if d_cond_check == True:  # d reached before frames ran out
                xy, zorders = _s.gen_landing(xy, zorders, v0land, speed0land, i_orbit, centroid0land)
                break
            else:  # not enough frames
                num_frames += int(d / 3)  # could use speed_max maybe
                if init_frame + num_frames > P.FRAMES_TOT - 100:
                    ok_rocket = 'num_frames > P.FRAMES_TOT '
                    break
                if num_frames > _s.gi['max_frames']:
                    ok_rocket = 'num_frames > max_frames'

        _s.xy = np.asarray(xy)
        _s.zorders = np.asarray(zorders)
        _s.alphas = norm.pdf(x=np.arange(0, len(_s.xy)), loc=len(_s.xy) / 2, scale=len(_s.xy) / 5)
        _s.alphas = min_max_normalize_array(_s.alphas, y_range=[0.7, 0.8])
        # _s.zorders = np.full(shape=(len(_s.xy)), fill_value=99999)

        _s.set_frame_ss(init_frame, len(xy))

        print("id: " + str(_s.id).ljust(5) +
              " | num_frames: " + str(i).ljust(5) +
              " | speed_max: " + str(speed_max)[0:4] +
              " | attempts_tot: " + str(attempt + 1).ljust(4) +
              " | ok_rocket: " + str(ok_rocket).ljust(20)
              )

        return ok_rocket

    def landingOLD(_s):

        kp = 0.8  # compulsory
        ki = 0.00  # goes crazy when ob
        kd = 0.01
        _s.pid_x = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_y = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vx = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vy = _s.PIDController(kp=kp, ki=ki, kd=kd)

        vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])  # dictates max speed currently
        v0 = np.copy(vxy_i)

        # max_speed = np.linalg.norm(vxy_i) * 1
        turn_rate_max = 0.1 * np.pi  # FLIPPED: Large value=no turn
        num_frames = int(0.8 * _s.gi['frames_max'])

        # _s.r = 30
        _s.theta = np.arctan(vxy_i[1] / vxy_i[0])  # 0.12 * 2 * np.pi  #

        p1_xy = np.copy(_s.p1.xy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames])
        p1_centroids = _s.p1.centroids[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]

        if _s.p1.id in ['GSS']:
            _s.r = p1_centroids[0] * 10  # GSS centroid=2
        else:
            _s.r = p1_centroids[0] * 0.5 * _s.p1.gi['centroid_mult']

        p1d_xy_prev = _s.get_dummy_orbit(p1_xy[0])
        p1_zorders = _s.p1.zorders[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]

        # use_v0 = np.zeros((num_frames,))
        # use_v0[0:50] = np.linspace(1, 0, 50)
        # use_pid = np.full((num_frames,), fill_value=1.)
        # use_pid[0:50] = np.linspace(0, 1, 50)

        xy_i = np.copy(_s.xy[-1, :])

        consequtive_num = 0

        xy = []
        zorders = []

        for i in range(1, num_frames):
            p1d_xy_i = _s.get_dummy_orbit(p1_xy[i])
            p1d_vxy_i = (p1d_xy_i - p1d_xy_prev)  # / 0.05
            speed_min = np.linalg.norm(p1_xy[i] - p1_xy[i - 1])
            speed_max = max(1, 2 * speed_min)
            p1d_xy_prev = p1d_xy_i

            # Position errors
            error_x = p1d_xy_i[0] - xy_i[0]
            error_y = p1d_xy_i[1] - xy_i[1]

            # Velocity errors
            error_vx = p1d_vxy_i[0] - vxy_i[0]
            error_vy = p1d_vxy_i[1] - vxy_i[1]

            # PID Updates
            _pid_x = _s.pid_x.update(error_x)
            _pid_y = _s.pid_y.update(error_y)
            _pid_vx = _s.pid_vx.update(error_vx)
            _pid_vy = _s.pid_vy.update(error_vy)

            vxy_desired = np.array([_pid_x + _pid_vx, _pid_y + _pid_vy])
            vxy_new = _s.limit_turn_rate(vxy_i, vxy_desired, speed_min, speed_max, turn_rate_max)

            # vxy_i = use_v0[i] * v0 + use_pid[i] * vxy_new  # vxy_new
            vxy_i = vxy_new  # vxy_new

            # xy_i += vxy_i  # * 0.05
            xy_i = p1d_xy_i  # DEBUG

            xy.append(np.copy(xy_i))
            zorders.append(999)

            # if i > 100:
            #     dist = np.linalg.norm(p1_xy[i] - xy_i)
            #     dist_factor = 1.05
            #     if _s.p1.id in ['GSS']:
            #         dist_factor = 5
            #
            #     if dist < p1_centroids[i] * dist_factor:
            #         '''This MUST be outside landing r '''
            #         consequtive_num += 1
            #         if consequtive_num >= 50:
            #             break
            #     else:
            #         consequtive_num = 0

        _s.xy = np.concatenate((_s.xy, np.asarray(xy)))
        _s.zorders = np.concatenate((_s.zorders, np.asarray(zorders)))

    def landingOLDOLD(_s):

        kp = 0.7  # compulsory
        ki = 0.01  # goes crazy when ob
        kd = 0.005
        _s.pid_x = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_y = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vx = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vy = _s.PIDController(kp=kp, ki=ki, kd=kd)

        vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])
        v0 = np.copy(vxy_i)

        '''
        TODO: num_frames should depend on entry v: FIXED
        ALSO it should depend on DIFFERENCE IN v between p1 and rocket. 
        '''
        speed_max = P.SPEED_MULTIPLIER * _s.gi[
            'speed_max']  # Shouldnt be different for mid_flight and landing np.linalg.norm(vxy_i) * 1
        num_frames = int(0.2 * _s.gi['frames_max'])
        num_frames_t = int(min(num_frames, np.linalg.norm(vxy_i) * 100))

        num_rot = 1
        # if num_frames_t > 150:
        #     num_rot = 2

        p1_xy = np.copy(_s.p1.xy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames])
        p1_vxy = _s.p1.vxy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]
        p1_centroids = _s.p1.centroids[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]
        p1_zorders = _s.p1.zorders[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]

        if _s.destination_type == 'inter':
            r = np.linspace(p1_centroids[0] * 0.5 * _s.p1.gi['centroid_mult'], p1_centroids[num_frames - 1],
                            num_frames_t)
        else:
            r = np.linspace(p1_centroids[0] * 2, p1_centroids[num_frames - 1], num_frames_t)

        # xy_t = np.zeros((num_frames, 2))  # ROTATION FROM THE POINT OF ORBIT ENTRY: PROBLEM: YOU DONT KNOW num_frames!
        xy_t = np.zeros((num_frames_t, 2))  # ROTATION FROM THE POINT OF ORBIT ENTRY: PROBLEM: YOU DONT KNOW num_frames!
        xy_t[:, 0] = np.sin(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * np.sign(vxy_i[0]) * r
        xy_t[:, 1] = -np.cos(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * np.sign(
            vxy_i[1]) * r * 0.15  # OBS TILT AFFECTED BY THIS. MAYBE RANDOM

        # # Apply tilt by rotating the coordinates
        tilt = np.arctan(vxy_i[1] / vxy_i[0])  # 0.12 * 2 * np.pi  #
        if vxy_i[1] > 0 and vxy_i[0] > 0:
            tilt += 0

        cos_theta = np.cos(tilt)
        sin_theta = np.sin(tilt)
        x_rot = cos_theta * xy_t[:, 0] - sin_theta * xy_t[:, 1]
        y_rot = sin_theta * xy_t[:, 0] + cos_theta * xy_t[:, 1]

        xy_t_rot = np.copy(xy_t)  # OBS: _s.xy_t no longer rotated!
        xy_t_rot[:, 0] = x_rot
        xy_t_rot[:, 1] = y_rot

        p1_xy_rot = p1_xy
        p1_xy_rot[0:num_frames_t, :] += xy_t_rot
        p1_vxy_rot = np.gradient(p1_xy_rot, axis=0)

        use_v0 = np.zeros((num_frames,))
        use_v0[0:50] = np.linspace(1, 0, 50)
        use_pid = np.full((num_frames,), fill_value=1.)
        use_pid[0:50] = np.linspace(0, 1, 50)

        xy_i = np.copy(_s.xy[-1, :])

        consequtive_num = 0
        # consequtive_prev = False
        xy = []
        zorders = []

        for i in range(1, num_frames):  # where will you be next frame? Then I will append myself to that.

            error_x = p1_xy_rot[i, 0] - xy_i[0]
            error_y = p1_xy_rot[i, 1] - xy_i[1]
            error_vx = p1_vxy_rot[i, 0] - vxy_i[0]
            error_vy = p1_vxy_rot[i, 1] - vxy_i[1]

            _pid_x = _s.pid_x.update(error_x)
            _pid_vx = _s.pid_vx.update(error_vx)
            _pid_y = _s.pid_y.update(error_y)
            _pid_vy = _s.pid_vy.update(error_vy)

            vxy_new = vxy_i + np.array([_pid_x + _pid_vx, _pid_y + _pid_vy])

            # Calculate speed and clip if necessary
            speed = np.linalg.norm(vxy_new)  # Compute the speed (magnitude of velocity vector)
            # p1_speed = np.linalg.norm(p1_vxy[i])
            if speed > speed_max:
                vxy_new = vxy_new / speed * speed_max  # first division normalizes to [0, 1]

            vxy_i = use_v0[i] * v0 + use_pid[i] * vxy_new  # vxy_new

            xy_i += vxy_i
            # xy_i = p1_xy_rot[i]

            xy.append(np.copy(xy_i))  # MUST BE BEFORE BELOW CHECK (xy cant be empty)
            if vxy_i[0] >= 0:
                zorders.append(p1_zorders[i] - 10)
            else:
                zorders.append(p1_zorders[i] + 10)

            if i > int(0.8 * num_frames_t):
                dist = np.linalg.norm(p1_xy[i] - xy_i)
                if _s.destination_type == 'inter':
                    if dist < p1_centroids[i] * 1.05:
                        '''This MUST be outside landing r '''
                        consequtive_num += 1
                        if consequtive_num >= 50:
                            break
                    else:
                        consequtive_num = 0
                elif _s.destination_type == 'orbit':
                    if dist < p1_centroids[i]:
                        break

        _s.xy = np.concatenate((_s.xy, np.asarray(xy)))
        _s.zorders = np.concatenate((_s.zorders, np.asarray(zorders)))

    def landingOLDOLDOLD(_s):

        kp = 0.6  # compulsory
        ki = 0.02  # goes crazy when ob
        kd = 0.01
        _s.pid_x = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_y = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vx = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vy = _s.PIDController(kp=kp, ki=ki, kd=kd)

        vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])  # dictates max speed currently

        max_speed = np.linalg.norm(vxy_i) * 1
        min_speed = 0

        num_frames_max = 200
        # use_v0 = np.zeros((num_frames_max,))
        # use_v0[0:50] = np.linspace(1, 0, 50)

        '''First p1 will not have been used in mid_flight'''
        p1_xy = _s.p1.xy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames_max]
        p1_vxy = _s.p1.vxy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames_max]
        p1_centroids = _s.p1.centroids[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames_max]

        xy_i = np.copy(_s.xy[-1, :])

        # p1_xy_ = p1_xy[1]
        error_x = p1_xy[1, 0] - xy_i[0]
        error_y = p1_xy[1, 1] - xy_i[1]
        p1_centroid_ = p1_centroids[1]
        ratios_k = [1, -0.5, 0.1]
        shifts = np.zeros((len(ratios_k), 2), dtype=np.float32)
        for i in range(len(ratios_k)):
            shifts[i, 0] = error_x * ratios_k[i]
            shifts[i, 1] = error_y * ratios_k[i]
        dist_cond = 5
        shift = shifts[0]
        shift_i = 0

        xy = []
        for i in range(1, num_frames_max):

            error_x = p1_xy[i, 0] + shift[0] - xy_i[0]
            error_y = p1_xy[i, 1] + shift[1] - xy_i[1]
            error_vx = p1_vxy[i, 0] - vxy_i[0]
            error_vy = p1_vxy[i, 1] - vxy_i[1]

            _pid_x = _s.pid_x.update(error_x)
            _pid_vx = _s.pid_vx.update(error_vx)
            _pid_y = _s.pid_y.update(error_y)
            _pid_vy = _s.pid_vy.update(error_vy)

            vxy_new = vxy_i + np.array([_pid_x + _pid_vx, _pid_y + _pid_vy])

            # Calculate speed and clip if necessary
            speed = np.linalg.norm(vxy_new)  # Compute the speed (magnitude of velocity vector)

            if speed > max_speed:
                vxy_new = vxy_new / speed * max_speed  # Scale down velocity to max speed
            elif speed < min_speed:
                vxy_new = vxy_new / speed * min_speed  # Scale up velocity to min speed

            vxy_i = vxy_new
            xy_i += vxy_i
            xy.append(np.copy(xy_i))

            dist = np.linalg.norm(p1_xy[i] + shift - xy_i)

            if dist < dist_cond:

                shift_i += 1
                if shift_i == len(shifts):
                    break

                shift = shifts[shift_i]
                max_speed *= 0.8

                _s.pid_x.prev_error = 0
                _s.pid_y.prev_error = 0
                _s.pid_vx.prev_error = 0
                _s.pid_vy.prev_error = 0

        _s.xy = np.concatenate((_s.xy, np.asarray(xy)))

    # def landingOLDOLDOLDOLD(_s, xy, zorders, v0land, speed0land, j_orbit, centroid0land):
    #     """
    #     v0land used for pi_offset gen
    #
    #     """
    #     xy_j = np.copy(xy[-1])  # shouldnt matter that much whether last xy is before or after centroid hit
    #
    #     speed0 = np.linalg.norm(v0)
    #     num_frames = int(speed0land * 30)   # base it on vxy_0
    #     if j_orbit + num_frames > P.FRAMES_TOT - 100:
    #         ok_rocket = 'num_frames > P.FRAMES_TOT '
    #
    #     speed = np.linspace(speed0land, 0, num_frames)  # Might have to be in loop
    #     num_rot = int(num_frames / 50)  # speed * _s.p1.centroids[len(xy)]
    #     pi_offset = 0  # 0.2 * 2 * np.pi  # OBS THIS NEEDS TO BE SET DEPENDING ON WHERE IT COMES FROM
    #
    #     p1_xy = np.copy(_s.p1.xy[j_orbit:j_orbit + num_frames])  # OBS copy NEEDED!
    #     # p1_xy[:, 0] += _s.p1.centroids[j_orbit:j_orbit + num_frames]  # Not sure these should be added. p1_xy SEEMS TO BE CENTER
    #     # p1_xy[:, 1] += _s.p1.centroids[j_orbit:j_orbit + num_frames]  # centroids are symmetrical
    #     # p1_vxy = _s.p1.vxy[j_orbit:j_orbit + num_frames]
    #     p1_zorders = _s.p1.zorders[j_orbit:j_orbit + num_frames]
    #
    #     # d0 = np.linalg.norm(p1_xy[1] - xy_j)  # obs the destination is now the center of the planet!
    #     # d = np.linspace(d0 * 2, 0, num_frames)
    #     radius = np.linspace(centroid0land * 3, 0, num_frames)  # starts
    #
    #     xy_t = np.zeros((num_frames, 2))  # ROTATION FROM THE POINT OF ORBIT ENTRY: PROBLEM: YOU DONT KNOW num_frames!
    #     xy_t[:, 0] = np.sin(np.linspace(0 + pi_offset, num_rot * 2 * np.pi + pi_offset, num_frames)) * centroid0land
    #     xy_t[:, 1] = -np.cos(np.linspace(0 + pi_offset, num_rot * 2 * np.pi + pi_offset, num_frames)) * centroid0land # NEEDS TO DEPEND ON TILT
    #
    #     # # Apply tilt by rotating the coordinates
    #     tilt = 0  #np.arctan(v0land[1] / v0land[0])
    #     if v0land[1] > 0 and v0land[0] > 0:
    #         tilt += 0
    #
    #     cos_theta = np.cos(tilt)
    #     sin_theta = np.sin(tilt)
    #     x_rot = cos_theta * xy_t[:, 0] - sin_theta * xy_t[:, 1]
    #     y_rot = sin_theta * xy_t[:, 0] + cos_theta * xy_t[:, 1]
    #
    #     xy_t_rot = np.copy(xy_t)  # OBS: _s.xy_t no longer rotated!
    #     xy_t_rot[:, 0] = x_rot
    #     xy_t_rot[:, 1] = y_rot
    #
    #     p1_xy_rot = p1_xy + xy_t_rot
    #     # p1_xy_rot = p1_xy #+ xy_t
    #     # p1_xy += xy_t
    #     # p1_vxy = np.gradient(p1_xy, axis=0)  # OBS
    #     # p1_vxy_rot = np.gradient(p1_xy, axis=0)
    #     p1_vxy_rot = np.gradient(p1_xy_rot, axis=0)
    #
    #     use_p1 = np.linspace(0.99, 1, num_frames)
    #     use_roc = np.linspace(0.2, 0.1, num_frames)
    #
    #     xy_land = []
    #     zorders_land = []
    #
    #     '''
    #     QUESTION: Should rocket aim for p1 rot or p1 centroid??? For p1!!!
    #     If rocket drifts after a while, its bcs speed[-1] = 0
    #     NEW: FUCK THE ROCKET: Just make sure centroid reached somehow and then create orbit somehow.
    #     '''
    #     for j in range(1, num_frames):
    #         direction_to_p1 = p1_xy[j] - xy_j  # xy_j last frame from prev function call i.e. i=0
    #         direction_to_p1 /= np.linalg.norm(direction_to_p1)  # Normalize
    #         d = np.linalg.norm(p1_xy[j] - xy_j)
    #
    #         v_roc = speed[j] * direction_to_p1  # moves toward center
    #         # vxy_j = v_roc * use_roc[j] + p1_vxy_rot[j] * use_p1[j]
    #         vxy_j = p1_vxy_rot[j] * use_p1[j]
    #         # vxy_j = v_roc * use_roc[j]
    #         xy_j = xy_j + vxy_j
    #         xy_land.append(xy_j)
    #
    #         if v_roc[0] > 0:
    #             zorders_land.append(p1_zorders[j] + 10)
    #         else:
    #             zorders_land.append(p1_zorders[j] + 10)
    #
    #     xy += xy_land
    #     zorders += zorders_land
    #
    #     adf = 5
    #
    #     # p1_rot = -_s.p1.rotation[init_frame:init_frame + num_frames]
    #     # p1_opp_corns = np.zeros((len(p1_rot), 2))  # OBS ONLY FOR ROCKETS
    #     # p1_opp_corns[:, 0] = _s.p1.centroids[init_frame:init_frame + num_frames, 0]
    #     # p1_opp_corns[:, 1] = _s.p1.centroids[init_frame:init_frame + num_frames, 1]
    #     # p1_opp_corns[:, 0] = -np.cos(p1_rot) * 1.7 * _s.p1.centroids[init_frame:init_frame + num_frames, 0]
    #     # p1_opp_corns[:, 1] = np.sin(p1_rot) * 1.7 * _s.p1.centroids[init_frame:init_frame + num_frames, 1]
    #
    #     # p1_xy += p1_opp_corns
    #     # p1_xy[:, 0] += _s.p1.centroid[0]
    #     # p1_xy[:, 1] += _s.p1.centroid[1]
    #
    #     # p1_vxy = _s.p1.vxy[init_frame:init_frame + num_frames]
    #
    #     '''Overshoot '''
    #     # d_xy_prev = [np.sign(p1_xy[i - 1, 0] - xy[i - 1][0]), np.sign(p1_xy[i - 1, 1] - xy[i - 1][1])]
    #     # d_xy_this = [np.sign(p1_xy[i, 0] - xy_i[0]), np.sign(p1_xy[i, 1] - xy_i[1])]
    #     #
    #     # if d_xy_this[0] != d_xy_prev[0] and d_xy_this[1] != d_xy_prev[1]:
    #     #     overshoots_tot += 1  # CANNOT USE INT BCS it needs resetting each attempt!
    #     #     overshoot = True
    #     #     break
    #
    #     # use_p1 = np.linspace(0, 0.99, num_frames)
    #     # vxy_i = (v0 * use_p0[i] + p1_vxy[i] * use_p1[i]) * use_p[i] + v_roc * use_roc[i]
    #
    #     # v_diff = np.linalg.norm(p1_vxy[i] - vxy_i)
    #
    #     return xy, zorders


