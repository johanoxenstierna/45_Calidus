import numpy as np


# Function to generate a planet's motion
def generate_planet_motion(pi_offset, num_rot, tilt_angle, r, center, frames_tot):
    # Generate the elliptical motion for the planet
    xy = np.zeros((frames_tot, 2))
    xy[:, 0] = np.sin(np.linspace(0 + pi_offset, num_rot * 2 * np.pi + pi_offset, frames_tot)) * r
    xy[:, 1] = -np.cos(np.linspace(0 + pi_offset, num_rot * 2 * np.pi + pi_offset, frames_tot)) * 0.1 * r

    # Apply tilt by rotating the coordinates
    cos_theta = np.cos(tilt_angle)
    sin_theta = np.sin(tilt_angle)
    x_rot = cos_theta * xy[:, 0] - sin_theta * xy[:, 1]
    y_rot = sin_theta * xy[:, 0] + cos_theta * xy[:, 1]

    # Shift by center coordinates (rotation around centroid)
    xy_pos = np.column_stack([x_rot + center[0], y_rot + center[1]])

    # Velocity is the time derivative of position
    vxy = np.gradient(xy_pos, axis=0)

    return xy_pos, vxy


# Planet p0 parameters
pi_offset_p0 = -1
num_rot_p0 = 1.4
tilt_angle_p0 = 0.3
r_p0 = 300
center_p0 = np.array([960, 540])  # Centroid coordinates

# Planet p1 parameters
pi_offset_p1 = 0
num_rot_p1 = 1.7
tilt_angle_p1 = 0
r_p1 = 400
center_p1 = np.array([960, 540])  # Centroid coordinates

# Simulation parameters
frames_tot = 1000  # Total frames in the simulation
r_max = 100  # Max velocity of the rocket (adjust as needed)

# Generate planet positions and velocities
xy_pos_p0, vxy_p0 = generate_planet_motion(pi_offset_p0, num_rot_p0, tilt_angle_p0, r_p0, center_p0, frames_tot)
xy_pos_p1, vxy_p1 = generate_planet_motion(pi_offset_p1, num_rot_p1, tilt_angle_p1, r_p1, center_p1, frames_tot)

# Rocket's initial position and velocity
xy_pos_r = xy_pos_p0[0]  # Rocket starts at p0's position
vxy_r = vxy_p0[0]  # Rocket starts with p0's velocity


# Function to calculate rocket velocity
def rocket_velocity(xy_pos_r, xy_pos_p1, vxy_p0, vxy_p1, r_max, frame, frames_tot):
    # Direction from the rocket to p1
    direction_to_p1 = xy_pos_p1[frame] - xy_pos_r
    direction_to_p1 /= np.linalg.norm(direction_to_p1)  # Normalize

    # Calculate the distance between the rocket and p1
    distance_to_p1 = np.linalg.norm(xy_pos_p1[frame] - xy_pos_r)

    # Define a velocity profile (smooth acceleration/deceleration)
    max_velocity = r_max  # Maximum velocity

    # The rocket's velocity will depend on the distance
    if distance_to_p1 < 1:
        # Slowing down as it gets close to p1
        vxy_r = vxy_p1[frame]  # End velocity matches p1's velocity
    else:
        # Acceleration phase
        velocity_factor = min(1, distance_to_p1 / (frames_tot / 2))  # Accelerate until halfway
        vxy_r = vxy_p0[frame] + velocity_factor * (max_velocity - np.linalg.norm(vxy_p0[frame])) * direction_to_p1

    return vxy_r


# Simulate rocket movement
rocket_trajectory = np.zeros((frames_tot, 2))
for frame in range(1, frames_tot):
    # Get the current velocity for the rocket
    vxy_r = rocket_velocity(xy_pos_r, xy_pos_p1, vxy_p0, vxy_p1, r_max, frame, frames_tot)

    # Update the rocket's position based on velocity (simple physics integration)
    xy_pos_r += vxy_r * 1  # Assuming time step of 1 for simplicity

    rocket_trajectory[frame] = xy_pos_r  # Store the rocket's position

