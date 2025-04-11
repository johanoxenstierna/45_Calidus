

def R_gi_():
    R = []


    # OGUN =================================
    R.append({
        'od': ['Ogun', 'GSS'],
        # 'speed_min': 1.0,
        # 'speed_max': 2,
        'init_frame_step': 400,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Ogun', 'Jupiter'],
        'speed_min': 1.0,
        'speed_max': 3,
        'init_frame_step': 400,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Ogun', 'Everglade'],
        'speed_min': 1.0,
        'speed_max': 3,
        'init_frame_step': 500,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    # VENUS =================================

    R.append({
        'od': ['Venus', 'GSS'],
        'speed_min': 1.0,
        'speed_max': 2,
        'init_frame_step': 500,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    # # NAUV =================================
    R.append({
        'od': ['Nauvis', 'GSS'],
        'speed_min': 1.0,
        'speed_max': 1.2,
        'init_frame_step': 80,  # 100
        'frames_max': 2000,
        'frames_min': 200,
    })

    # R.append({
    #     'od': ['GSS', 'Nauvis'],
    #     'speed_min': 1.0,
    #     'speed_max': 1.5,
    #     'init_frame_step': 300,  # 100
    #     'frames_max': 2000,
    #     'frames_min': 200,
    # })

    R.append({
        'od': ['GSS', 'Venus'],
        'speed_min': 1.0,
        'speed_max': 2.0,
        'init_frame_step': 500,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['GSS', 'Astro0b'],
        'speed_min': 1.0,
        'speed_max': 2.0,
        'init_frame_step': 400,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Nauvis', 'Petussia'],
        'speed_min': 1.0,
        'speed_max': 2.0,
        'init_frame_step': 600,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    # MARS =================================

    R.append({
        'od': ['Mars', 'GSS'],
        'speed_min': 1.0,
        'speed_max': 2.0,
        'init_frame_step': 400,  # 100
        'frames_max': 2000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Mars', 'Ogun'],
        'speed_min': 1.0,
        'speed_max': 2.0,
        'init_frame_step': 600,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Mars', 'Astro0b'],
        'speed_min': 1.0,
        'speed_max': 2.0,
        'init_frame_step': 700,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    # Astro0b =============================

    R.append({
        'od': ['Astro0b', 'GSS'],
        'speed_min': 1.0,
        'speed_max': 2,
        'init_frame_step': 600,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Astro0b', 'Everglade'],
        'speed_min': 1.0,
        'speed_max': 2,
        'init_frame_step': 600,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    # JUPITER =============================
    R.append({
        'od': ['Petussia', 'GSS'],
        'speed_min': 1.0,
        'speed_max': 2,
        'init_frame_step': 600,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Everglade', 'GSS'],
        'speed_min': 1.0,
        'speed_max': 2,
        'init_frame_step': 600,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    R.append({
        'od': ['Everglade', 'Astro0b'],
        'speed_min': 1.0,
        'speed_max': 2,
        'init_frame_step': 600,  # 100
        'frames_max': 3000,
        'frames_min': 200,
    })

    return R
