import numpy as np
import matplotlib.pyplot as plt


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        # Proportional term
        proportional = self.kp * error
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        return proportional + integral + derivative


# Example: Elliptical motion of p0
t = np.linspace(0, 10, 200)
a, b = 5, 3  # semi-major and semi-minor axes
p0_xy = np.column_stack((a * np.cos(t), b * np.sin(t)))

# Compute velocity of p0
p0_vxy = np.gradient(p0_xy, axis=0)

# Initial conditions for p1
p1_xy = np.array([0, 4.])  # starting at a different position
p1_vxy = np.array([0.5, 0])  # starting with some initial velocity

# PID controllers for position and velocity
# pid_pos_x = PIDController(kp=0.5, ki=0.1, kd=0.05)
# pid_pos_y = PIDController(kp=0.5, ki=0.1, kd=0.05)
# pid_vel_x = PIDController(kp=0.5, ki=0.1, kd=0.05)
# pid_vel_y = PIDController(kp=0.5, ki=0.1, kd=0.05)

pid_pos_x = PIDController(kp=0.5, ki=0.2, kd=0.2)
pid_pos_y = PIDController(kp=0.5, ki=0.2, kd=0.2)
pid_vel_x = PIDController(kp=0.1, ki=0.2, kd=0.1)
pid_vel_y = PIDController(kp=0.1, ki=0.2, kd=0.1)
.2
# Simulation loop
dt = 0.1  # time step
n_steps = len(t)
positions_p1 = [p1_xy.copy()]
for i in range(n_steps):
    # Calculate errors
    error_pos_x = p0_xy[i, 0] - p1_xy[0]
    error_pos_y = p0_xy[i, 1] - p1_xy[1]
    error_vel_x = p0_vxy[i, 0] - p1_vxy[0]
    error_vel_y = p0_vxy[i, 1] - p1_vxy[1]

    # Get PID control outputs (desired accelerations)
    acc_x = pid_pos_x.update(error_pos_x, dt) + pid_vel_x.update(error_vel_x, dt)
    acc_y = pid_pos_y.update(error_pos_y, dt) + pid_vel_y.update(error_vel_y, dt)

    # Update p1's velocity and position
    p1_vxy += np.array([acc_x, acc_y]) * dt  # velocity update (velocity = acceleration * dt)
    p1_xy += p1_vxy * dt  # position update (position = velocity * dt)

    # Store the position for plotting
    positions_p1.append(p1_xy.copy())

# Convert list to array for easier plotting
positions_p1 = np.array(positions_p1)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(p0_xy[:, 0], p0_xy[:, 1], label='p0 trajectory', color='blue')
plt.plot(positions_p1[:, 0], positions_p1[:, 1], label='p1 trajectory', color='red', linestyle='--')
plt.scatter([p1_xy[0]], [p1_xy[1]], color='red', label='Starting p1 position', zorder=5)
plt.scatter([p0_xy[0, 0]], [p0_xy[0, 1]], color='blue', label='Starting p0 position', zorder=5)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position Matching with PID Control')
plt.grid(True)
plt.show()
