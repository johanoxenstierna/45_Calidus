import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        # Proportional term
        proportional = self.kp * error
        # Integral term
        self.integral += error
        integral = self.ki * self.integral
        # Derivative term
        derivative = self.kd * (error - self.prev_error)
        self.prev_error = error
        return proportional + integral + derivative


# Example: Elliptical motion of p0
num_rot = 5
t = np.linspace(0, num_rot * 2 * np.pi, 400)
a, b = 300, 40  # semi-major and semi-minor axes
p0_xy = np.column_stack((-a * np.cos(t) + 1000, b * np.sin(t) + 500))

# Compute velocity of p0
p0_vxy = np.gradient(p0_xy, axis=0)

# Initial conditions for p1
p1_xy = np.array([1000, 300.])  # starting at a different position
p1_vxy = np.array([-1., 0])  # starting with some initial velocity

# PID controllers for position and velocity
kp = 0.5  # compulsory
ki = 0.1  # goes crazy when ob
kd = 0.05
pid_pos_x = PIDController(kp=kp, ki=ki, kd=kd)
pid_pos_y = PIDController(kp=kp, ki=ki, kd=kd)
pid_vel_x = PIDController(kp=kp, ki=ki, kd=kd)
pid_vel_y = PIDController(kp=kp, ki=ki, kd=kd)

'''
Proportional (Kp): Controls the immediate response to position and velocity errors.
Integral (Ki): Helps eliminate steady-state errors by accounting for accumulated past errors.
Derivative (Kd): Damps the system by reacting to the rate of change of errors.
'''

# Initialize figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Position Matching with PID Control')
ax.grid(True)

# Plot p0 trajectory
ax.plot(p0_xy[:, 0], p0_xy[:, 1], label='p0 trajectory', color='blue', alpha=0.5)

# Plot initial positions
p0_start, = ax.plot([], [], 'bo', label='Starting p0 position')
p1_start, = ax.plot([], [], 'ro', label='Starting p1 position')
p1_pos, = ax.plot([], [], 'r-', label='p1 trajectory')

# PID simulation loop for animation
# dt = 1  # time step
n_steps = len(t)


def update(frame):
    global p1_xy, p1_vxy

    # Calculate errors
    error_pos_x = p0_xy[frame, 0] - p1_xy[0]
    error_pos_y = p0_xy[frame, 1] - p1_xy[1]
    error_vel_x = p0_vxy[frame, 0] - p1_vxy[0]
    error_vel_y = p0_vxy[frame, 1] - p1_vxy[1]

    # Get PID control outputs (desired accelerations)
    acc_x = pid_pos_x.update(error_pos_x) + pid_vel_x.update(error_vel_x)
    acc_y = pid_pos_y.update(error_pos_y) + pid_vel_y.update(error_vel_y)

    # Update p1's velocity and position
    p1_vxy += np.array([acc_x, acc_y])  # velocity update (velocity = acceleration * dt)
    p1_xy += p1_vxy  # position update (position = velocity * dt)

    # Update the plot data
    p0_start.set_data([p0_xy[frame, 0]], [p0_xy[frame, 1]])  # Update p0's position
    p1_start.set_data([p1_xy[0]], [p1_xy[1]])  # Update p1's current position
    p1_pos.set_data(np.append(p1_pos.get_xdata(), p1_xy[0]),
                    np.append(p1_pos.get_ydata(), p1_xy[1]))  # Update p1's trajectory

    return p0_start, p1_start, p1_pos


# Create the animation
ani = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True, repeat=False)

# Show the animation
plt.legend()
plt.show()


# OLD
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
#
# class PIDController:
#     def __init__(self, kp, ki, kd):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.prev_error = 0
#         self.integral = 0
#
#     def update(self, error, dt):
#         # Proportional term
#         proportional = self.kp * error
#         # Integral term
#         self.integral += error * dt
#         integral = self.ki * self.integral
#         # Derivative term
#         derivative = self.kd * (error - self.prev_error) / dt
#         self.prev_error = error
#         return proportional + integral + derivative
#
#
# # Example: Elliptical motion of p0
# num_rot = 5
# t = np.linspace(0, num_rot * 2 * np.pi, 400)
# a, b = 300, 40  # semi-major and semi-minor axes
# p0_xy = np.column_stack((-a * np.cos(t) + 1000, b * np.sin(t) + 500))
#
# # Compute velocity of p0
# p0_vxy = np.gradient(p0_xy / 100, axis=0)
#
# # Initial conditions for p1
# p1_xy = np.array([1000, 300.])  # starting at a different position
# p1_vxy = np.array([-1., 0])  # starting with some initial velocity
#
# # PID controllers for position and velocity
# kp = 0.1  # compulsory
# ki = 0.0000001  # goes crazy when ob
# kd = 0.00000000005
# pid_pos_x = PIDController(kp=kp, ki=ki, kd=kd)
# pid_pos_y = PIDController(kp=kp, ki=ki, kd=kd)
# pid_vel_x = PIDController(kp=kp, ki=ki, kd=kd)
# pid_vel_y = PIDController(kp=kp, ki=ki, kd=kd)
#
# '''
# Proportional (Kp): Controls the immediate response to position and velocity errors.
# Integral (Ki): Helps eliminate steady-state errors by accounting for accumulated past errors.
# Derivative (Kd): Damps the system by reacting to the rate of change of errors.
# '''
#
# # Initialize figure and axis for animation
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_xlim(0, 1920)
# ax.set_ylim(0, 1080)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Position Matching with PID Control')
# ax.grid(True)
#
# # Plot p0 trajectory
# ax.plot(p0_xy[:, 0], p0_xy[:, 1], label='p0 trajectory', color='blue', alpha=0.5)
#
# # Plot initial positions
# p0_start, = ax.plot([], [], 'bo', label='Starting p0 position')
# p1_start, = ax.plot([], [], 'ro', label='Starting p1 position')
# p1_pos, = ax.plot([], [], 'r-', label='p1 trajectory')
#
# # PID simulation loop for animation
# dt = 1  # time step
# n_steps = len(t)
#
#
# def update(frame):
#     global p1_xy, p1_vxy
#
#     # Calculate errors
#     error_pos_x = p0_xy[frame, 0] - p1_xy[0]
#     error_pos_y = p0_xy[frame, 1] - p1_xy[1]
#     error_vel_x = p0_vxy[frame, 0] - p1_vxy[0]
#     error_vel_y = p0_vxy[frame, 1] - p1_vxy[1]
#
#     # Get PID control outputs (desired accelerations)
#     acc_x = pid_pos_x.update(error_pos_x, dt) + pid_vel_x.update(error_vel_x, dt)
#     acc_y = pid_pos_y.update(error_pos_y, dt) + pid_vel_y.update(error_vel_y, dt)
#
#     # Update p1's velocity and position
#     p1_vxy += np.array([acc_x, acc_y]) * dt  # velocity update (velocity = acceleration * dt)
#     p1_xy += p1_vxy * dt  # position update (position = velocity * dt)
#
#     # Update the plot data
#     p0_start.set_data([p0_xy[frame, 0]], [p0_xy[frame, 1]])  # Update p0's position
#     p1_start.set_data([p1_xy[0]], [p1_xy[1]])  # Update p1's current position
#     p1_pos.set_data(np.append(p1_pos.get_xdata(), p1_xy[0]),
#                     np.append(p1_pos.get_ydata(), p1_xy[1]))  # Update p1's trajectory
#
#     return p0_start, p1_start, p1_pos
#
#
# # Create the animation
# ani = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True, repeat=False)
#
# # Show the animation
# plt.legend()
# plt.show()

