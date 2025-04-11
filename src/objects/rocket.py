
from scipy.stats import beta, norm
from src.m_functions import *
from src.objects.abstract import AbstractObject, AbstractSSS


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
        _s.pid_pos = None
        _s.pid_vel = None

    def gen_rocket_motion(_s):

        _s.takeoff()
        _s.mid_flight()
        _s.landing()

        _s.zorders = np.asarray(_s.zorders)

        if P.WRITE != 0:
            _s.alphas = norm.pdf(x=np.arange(0, len(_s.xy)), loc=len(_s.xy) / 2, scale=len(_s.xy) / 5)
            y_range_min, y_range_max = 0.8, 0.9
            if P.WRITE != 0:
                y_range_min, y_range_max = 0.1, 0.4
            _s.alphas = min_max_normalize_array(_s.alphas, y_range=[y_range_min, y_range_max])  # 0.2, 0.7

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

        num_frames = 100 #int(200 / (_s.gi['speed_max'])
        if P.WRITE != 0:
            num_frames = 400

        '''
        TODO HERE: DO IT AS MUCH AS O1 AS POSSIBLE. NO DIRECTIONS IN XY_T
        ALSO FIX ZORDERS AFTER 
        '''
        if _s.destination_type == 'inter':
            # r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5, _s.p0.centroids[_s.init_frame] * _s.p0.gi['centroid_mult'], num_frames)
            r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5, _s.p0.centroids[_s.init_frame] * _s.p0.gi['centroid_mult'], num_frames)
            # r = 50  # 50 GIVES EXIT SPEED=1
            num_rot = 1
        else: #_s.destination_type == 'orbit':
            # r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5, _s.p0.centroids[_s.init_frame] * 4, num_frames)
            r = np.linspace(_s.p0.centroids[_s.init_frame] * 0.5, _s.p1.gi['r'], num_frames)
            num_rot = 1  # gonna mess up if weird number here

        y_squeeze = 0.08
        xy_t = np.zeros((num_frames, 2), dtype=np.float32)
        xy_t[:, 0] = np.sin(np.linspace(0, num_rot * 2 * np.pi, num_frames)) * r
        xy_t[:, 1] = -np.cos(np.linspace(0, num_rot * 2 * np.pi, num_frames)) * r * y_squeeze

        # # Apply tilt by rotating the coordinates
        direction_to_p1 = _s.p1.xy[_s.init_frame + num_frames - 1] - _s.p0.xy[_s.init_frame + num_frames - 1]  # p1 must be first
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

        kp = 0.99
        ki = 0.00
        kd = 0.05
        _s.pid_pos = _s.PIDController(kp=kp, ki=ki, kd=kd)
        _s.pid_vel = _s.PIDController(kp=kp, ki=ki, kd=kd)

        xy_i = np.copy(_s.xy[-1, :])
        vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])  # dictates max speed currently

        num_frames = int(0.8 * _s.gi['frames_max'])  # No need for speed here cuz break

        '''OBS len(xy) - 1 GIVES LAST xy ADDED i.e. CURRENT, BUT LOOP BELOW SHOULD USE NEXT VALUES ie range(1, num)
        BCS OTHERWISE AFTER xy.append() the latest xy will be one step ahead of p1, 
        and that prevents clean way to retrieve values after loop
        '''

        p1_xy = _s.p1.xy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]
        p1_vxy = _s.p1.vxy[_s.init_frame + len(_s.xy) - 1:_s.init_frame + len(_s.xy) - 1 + num_frames]
        dist_diff_max = 200  # np.linalg.norm(p1_xy[0] - xy_i)
        vel_diff_max = np.array([_s.p1.speed_max * 3, _s.p1.speed_max * 3])  # MIGHT NEED SEPARATE X Y HERE
        speed_diff_at_end_sought = 0.3
        speed_far = _s.p1.speed_max * 1.2
        speed_smoothing = 0.1  # this amount is vxy_new and 1-this is vxy_i

        dist_cond_break = 15

        xy = []
        vxy = []

        for i in range(1, num_frames):  # where will you be next frame? Then I will append myself to that.

            if i == 52:
                adfg = 5

            error_pos = p1_xy[i] - xy_i
            error_vel = p1_vxy[i] - vxy_i

            _pid_pos = _s.pid_pos.update(error_pos)
            _pid_vel = _s.pid_vel.update(error_vel)

            norm_pos = np.linalg.norm(_pid_pos) + 1e-6
            norm_vel = np.linalg.norm(_pid_vel) + 1e-6

            _pid_pos01 = _pid_pos / norm_pos * min(norm_pos / dist_diff_max, 1.0)
            _pid_vel01 = _pid_vel / norm_vel * min(norm_vel / np.linalg.norm(vel_diff_max), 1.0)

            dist01 = np.linalg.norm(error_pos) + 1e-6
            dist01 = np.clip(dist01 / dist_diff_max, 0.0, 1.0)  # Fades from 1 (far) to 0 (close)
            dist01 = max(dist01, 0.999)  # always use at least 50% position

            vxy_new = dist01 * _pid_pos01 + (1 - dist01) * _pid_vel01  # SHOULD BE 0-1

            speed_p1 = np.linalg.norm(p1_vxy[i])
            speed_near = speed_p1 + speed_diff_at_end_sought
            speed_non_smoothed = dist01 * speed_far + (1 - dist01) * speed_near
            vxy_new = vxy_new / (np.linalg.norm(vxy_new) + 1e-6) * speed_non_smoothed

            vxy_i = vxy_i * (1 - speed_smoothing) + vxy_new * speed_smoothing
            speed_debug = np.linalg.norm(vxy_i)
            xy_i += vxy_i

            xy.append(np.copy(xy_i))
            vxy.append(np.copy(vxy_i))

            dist = np.linalg.norm(p1_xy[i] - xy_i)
            if dist < dist_cond_break:
                break

        '''Correction based on difference in speed between rocket and p1 at this stage. 
        Rocket speed will generally not be speed_p1 + speed_diff_at_end here because of speed_smoothing.
        TODO: speed_diff_at end sought does not necessarily have to be constant. If rocket is much faster than p1 then just do two orbits in the landing.  
        '''
        NUM_CORR = 20
        if len(xy) > 20:
            # speed_end = np.linalg.norm(np.array([xy[-1][0] - xy[-2][0], xy[-1][1] - xy[-2][1]]))
            speed_current = np.linalg.norm(vxy[-1])
            speed_sought = np.linalg.norm(p1_vxy[len(xy)]) + speed_diff_at_end_sought
            # speed_multiplier = speed_sought / (speed_current + 1e-6)
            scale_factors = np.linspace(1.0, speed_sought / (speed_current + 1e-6), NUM_CORR)
            vxy_scaled = [v * f for v, f in zip(vxy[-NUM_CORR:], scale_factors)]
            xy_scaled = [xy[-NUM_CORR]]
            for i in range(1, len(vxy_scaled)):
                xy_scaled.append(xy_scaled[-1] + vxy_scaled[i])
            xy = xy[0:len(xy) - NUM_CORR] + xy_scaled

        # ZORDERS ====================================
        zorders = np.full((len(xy),), fill_value=_s.zorders[-1])
        # ============================================

        _s.xy = np.concatenate((_s.xy, np.asarray(xy, dtype=np.float32)))
        _s.alphas = np.concatenate((_s.alphas, np.full((len(xy),), fill_value=0.5)))
        _s.zorders = np.concatenate((_s.zorders, zorders))

        p1_xy_debug = _s.p1.xy[_s.init_frame + len(_s.xy) - 1]  # THIS IS NOW THE CURRENT VALUE
        speed_i_debug = np.linalg.norm(np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]]))
        speed_p1_debug = np.linalg.norm(_s.p1.vxy[_s.init_frame + len(_s.xy) - 1])
        speed_diff_debug = speed_p1_debug + speed_diff_at_end_sought - speed_i_debug
        ads = 5

    def landing(_s):

        """
        num_frames_t: The number of frames that the function lasts. Currently it's just a constant but eventually it may be set dynamically.

        4 components with shape (num_frames_t, 2)
        xy_t_rot: Orbital motion that is as similar as possible to the takeoff function. It is centered on 0 currently so can't be shown in isolation. The three other components can all be used in isolation to enable debugging. Note y_squeeze = 0.00001 is necessary here to ensure the orbit starts at an origin.
        xy_v0: The inital coordinate (xy_i) is used to generate an array xy_v0, which is moved using entry velocity vxy_i (which is also reduced by some number of frames NUM_0 <= num_frames_t)
        p1_shifted: p1's position shifted by -dist_xy. dist_xy is the distance between the rocket and p1 at the beginning of this function.
        p1_actuall: p1's position (the target coordinate).
        """

        # CURRENT FRAME =========================================
        xy_i = np.copy(_s.xy[-1, :])
        vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])
        p1_vxy_i = _s.p1.vxy[_s.init_frame + len(_s.xy) - 1]
        speed_diff_debug = np.linalg.norm(p1_vxy_i) - np.linalg.norm(vxy_i)

        # NEXT FRAME (This is where this function starts working) =========================================
        xy_i += vxy_i  # can now be used as first value
        xy_v0 = [xy_i]
        dist_xy = _s.p1.xy[_s.init_frame + len(_s.xy)] - xy_i

        num_frames_t = int(400)
        num_rot = 1

        y_squeeze = 0.00001

        r = np.linspace(1, 0, num_frames_t)

        xy_t = np.zeros((num_frames_t, 2))
        xy_t[:, 0] = np.sin(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * r
        xy_t[:, 1] = -np.cos(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * r * y_squeeze

        # Apply tilt by rotating the coordinates
        tilt = np.arctan2(vxy_i[1], vxy_i[0])

        cos_theta = np.cos(tilt)
        sin_theta = np.sin(tilt)
        x_rot = cos_theta * xy_t[:, 0] - sin_theta * xy_t[:, 1]
        y_rot = sin_theta * xy_t[:, 0] + cos_theta * xy_t[:, 1]

        xy_t_rot = np.copy(xy_t)
        xy_t_rot[:, 0] = x_rot
        xy_t_rot[:, 1] = y_rot

        progress = np.linspace(0, 1, num_frames_t)

        xy_t_rot *= np.expand_dims((1 - sigmoid_blend(progress)), axis=1)

        NUM_0 = 50
        use_v0 = np.zeros((num_frames_t,))
        use_v0[0:NUM_0] = np.linspace(1, 0, NUM_0)
        for i in range(1, len(use_v0)):
            xy_v0.append(xy_v0[-1] + vxy_i * use_v0[i])

        p1_shifted = -dist_xy + _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_t]
        p1_actuall = _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_t]

        # ===========================

        w_v0 = (1 - sigmoid_blend(progress)) ** 2  # dies off quickly
        w_p1_shifted = 2 * sigmoid_blend(progress) * (1 - sigmoid_blend(progress))  # peak in middle
        w_p1_actual = sigmoid_blend(progress) ** 2  # ramps up smoothly

        # Normalize weights
        sum_weights = w_v0 + w_p1_shifted + w_p1_actual
        w_v0 /= sum_weights
        w_p1_shifted /= sum_weights
        w_p1_actual /= sum_weights

        # Shape (num_frames_t, 2)
        weights = lambda w: np.stack((w, w), axis=1)

        xy = xy_t_rot + (
                weights(w_v0) * xy_v0 +
                weights(w_p1_shifted) * p1_shifted +
                weights(w_p1_actual) * p1_actuall
        )


        # =============================================

        _s.xy = np.concatenate((_s.xy, np.asarray(xy)))
        _s.alphas = np.concatenate((_s.alphas, np.full((len(xy),), fill_value=0.99)))

        # ZORDERS =====================================
        zorders = np.full((len(xy),), dtype=int, fill_value=_s.p1.zorders[_s.init_frame + len(_s.xy)] + 10)
        vxy = np.gradient(xy, axis=1)
        inds_neg = np.where(vxy[:, 1] <= 0)[0]  # when they move up, they move BEHIND
        zorders[inds_neg] -= 20
        _s.zorders = np.concatenate((_s.zorders, zorders))

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
            self.prev_error = np.zeros(2)
            self.integral = np.zeros(2)

        def update(self, error):
            proportional = self.kp * error
            self.integral += error
            integral = self.ki * self.integral
            derivative = self.kd * (error - self.prev_error)
            self.prev_error = error
            return proportional + integral + derivative

    # def landing(_s):
    #
    #     """
    #     num_frames_t: The number of frames that the function lasts. Currently it's just a constant but eventually it may be set dynamically.
    #
    #     4 components with shape (num_frames_t, 2)
    #     xy_t_rot: Orbital motion that is as similar as possible to the takeoff function. It is centered on 0 currently so can't be shown in isolation. The three other components can all be used in isolation to enable debugging. Note y_squeeze = 0.00001 is necessary here to ensure the orbit starts at an origin.
    #     xy_v0: The inital coordinate (xy_i) is used to generate an array xy_v0, which is moved using entry velocity vxy_i (which is also reduced by some number of frames NUM_0 <= num_frames_t)
    #     p1_shifted: p1's position shifted by -dist_xy. dist_xy is the distance between the rocket and p1 at the beginning of this function.
    #     p1_actuall: p1's position (the target coordinate).
    #     """
    #
    #     # CURRENT FRAME =========================================
    #     xy_i = np.copy(_s.xy[-1, :])
    #     vxy_i = np.array([_s.xy[-1, 0] - _s.xy[-2, 0], _s.xy[-1, 1] - _s.xy[-2, 1]])
    #     p1_vxy_i = _s.p1.vxy[_s.init_frame + len(_s.xy) - 1]
    #     speed_diff_debug = np.linalg.norm(p1_vxy_i) - np.linalg.norm(vxy_i)
    #
    #     # NEXT FRAME (This is where this function starts working) =========================================
    #     xy_i += vxy_i  # can now be used as first value
    #     xy_v0 = [xy_i]
    #     dist_xy = _s.p1.xy[_s.init_frame + len(_s.xy)] - xy_i
    #
    #     num_frames_t = int(400)
    #     num_rot = 1
    #
    #     y_squeeze = 0.00001
    #
    #     r = np.linspace(25, 0, num_frames_t)
    #
    #     xy_t = np.zeros((num_frames_t, 2))
    #     xy_t[:, 0] = np.sin(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * r
    #     xy_t[:, 1] = -np.cos(np.linspace(0, num_rot * 2 * np.pi, num_frames_t)) * r * y_squeeze
    #
    #     # Apply tilt by rotating the coordinates
    #     tilt = np.arctan2(vxy_i[1], vxy_i[0])
    #
    #     cos_theta = np.cos(tilt)
    #     sin_theta = np.sin(tilt)
    #     x_rot = cos_theta * xy_t[:, 0] - sin_theta * xy_t[:, 1]
    #     y_rot = sin_theta * xy_t[:, 0] + cos_theta * xy_t[:, 1]
    #
    #     xy_t_rot = np.copy(xy_t)
    #     xy_t_rot[:, 0] = x_rot
    #     xy_t_rot[:, 1] = y_rot
    #
    #     NUM_0 = 200
    #     use_v0 = np.zeros((num_frames_t,))
    #     use_v0[0:NUM_0] = np.linspace(1, 0, NUM_0)
    #     for i in range(1, len(use_v0)):
    #         xy_v0.append(xy_v0[-1] + vxy_i * use_v0[i])
    #     use_v0 = np.stack((use_v0, use_v0), axis=1)
    #
    #     p1_shifted = -dist_xy + _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_t]
    #     p1_actuall = _s.p1.xy[_s.init_frame + len(_s.xy):_s.init_frame + len(_s.xy) + num_frames_t]
    #
    #     use_p1_shifted = np.full((num_frames_t,), fill_value=1.)
    #     use_p1_shifted[0:NUM_0] = np.linspace(0, 1, NUM_0)
    #     use_p1_shifted = np.stack((use_p1_shifted, use_p1_shifted), axis=1)
    #
    #     # ============================
    #
    #     use_v0_and_shifted = np.linspace(1, 0, num=num_frames_t)
    #     use_v0_and_shifted = np.stack((use_v0_and_shifted, use_v0_and_shifted), axis=1)
    #     use_p1_actuall = np.linspace(0, 1, num=num_frames_t)
    #     use_p1_actuall = np.stack((use_p1_actuall, use_p1_actuall), axis=1)
    #
    #     # xy = xy_t_rot + use_v0 * xy_v0 + use_shifted * shifted
    #     xy = xy_t_rot + use_v0_and_shifted * (use_v0 * xy_v0 + use_p1_shifted * p1_shifted) + use_p1_actuall * p1_actuall
    #
    #     zorders = np.full((len(xy),), dtype=int, fill_value=9999)
    #     _s.xy = np.concatenate((_s.xy, np.asarray(xy)))
    #     _s.alphas = np.concatenate((_s.alphas, np.full((len(xy),), fill_value=0.99)))
    #     _s.zorders = np.concatenate((_s.zorders, np.asarray(zorders)))




