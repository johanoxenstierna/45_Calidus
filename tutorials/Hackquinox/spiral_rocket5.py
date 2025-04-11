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
        proportional = self.kp * error
        self.integral += error
        integral = self.ki * self.integral
        derivative = self.kd * (error - self.prev_error)
        self.prev_error = error
        return proportional + integral + derivative


class RocketLandingSimulation:
    def __init__(self, kp, ki, kd, max_turn_rate=np.pi / 12):
        self.p1 = np.array([1000, 500])  # Destination planet within 1920x1080
        self.rocket_pos = np.array([100., 100.])  # Initial position
        self.rocket_vel = np.array([200., 2.])  # Initial velocity
        self.radius_orbit = 200.0
        self.theta = 0.0
        self.prev_orbit_point = self.get_orbit_point()
        self.max_turn_rate = 1  # Maximum turn rate in radians

        # Initialize PID controllers
        self.pid_x = PIDController(kp, ki, kd)
        self.pid_y = PIDController(kp, ki, kd)
        self.pid_vx = PIDController(kp, ki, kd)
        self.pid_vy = PIDController(kp, ki, kd)

        # Plot setup
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_xlim(0, 1920)
        self.ax.set_ylim(1080, 0)
        self.rocket_dot, = self.ax.plot([], [], 'ro')
        self.orbit_dot, = self.ax.plot([], [], 'bo')

    def get_orbit_point(self):
        """Get the orbit target with a shrinking radius."""
        self.radius_orbit = max(10, self.radius_orbit * 0.99)
        self.theta += 0.1
        x = self.p1[0] + self.radius_orbit * np.cos(self.theta)
        y = self.p1[1] + self.radius_orbit * np.sin(self.theta)
        return np.array([x, y])

    def limit_turn_rate(self, desired_direction):
        """Limits the change in direction to prevent sharp turns."""
        current_direction = np.arctan2(self.rocket_vel[1], self.rocket_vel[0])
        desired_angle = np.arctan2(desired_direction[1], desired_direction[0])
        angle_diff = np.arctan2(np.sin(desired_angle - current_direction), np.cos(desired_angle - current_direction))

        # Limit the angle change
        if abs(angle_diff) > self.max_turn_rate:
            desired_angle = current_direction + np.sign(angle_diff) * self.max_turn_rate

        return np.array([np.cos(desired_angle), np.sin(desired_angle)])

    def update(self, frame):
        dummy_orbit = self.get_orbit_point()
        orbit_velocity = (dummy_orbit - self.prev_orbit_point) / 0.05
        self.prev_orbit_point = dummy_orbit

        # Position errors
        error_x = dummy_orbit[0] - self.rocket_pos[0]
        error_y = dummy_orbit[1] - self.rocket_pos[1]

        # Velocity errors
        error_vx = orbit_velocity[0] - self.rocket_vel[0]
        error_vy = orbit_velocity[1] - self.rocket_vel[1]

        # PID Updates
        pid_x = self.pid_x.update(error_x)
        pid_y = self.pid_y.update(error_y)
        pid_vx = self.pid_vx.update(error_vx)
        pid_vy = self.pid_vy.update(error_vy)

        # Desired velocity direction
        desired_velocity = np.array([pid_x + pid_vx, pid_y + pid_vy])
        limited_velocity_direction = self.limit_turn_rate(desired_velocity)

        # Adjust rocket velocity and position
        speed = np.linalg.norm(self.rocket_vel)
        self.rocket_vel = limited_velocity_direction * speed  # Maintain speed while adjusting direction
        self.rocket_pos += self.rocket_vel * 0.05

        # Update plot
        self.rocket_dot.set_data([self.rocket_pos[0]], [self.rocket_pos[1]])
        self.orbit_dot.set_data([dummy_orbit[0]], [dummy_orbit[1]])
        return self.rocket_dot, self.orbit_dot

    def run(self):
        ani = FuncAnimation(self.fig, self.update, frames=500, interval=1, blit=True)
        plt.show()


# Example usage
simulation = RocketLandingSimulation(kp=1.0, ki=0.1, kd=0.05)
simulation.run()
