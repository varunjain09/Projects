import pybullet as p
import time
from simple_pid import PID
from robot_prop_2 import setup_simulation, load_robot, apply_torque, add_target_point
import numpy as np

# Constants
XML_PATH = "two_wheel_robot.xml"  # Path to your robot file
TARGET_ANGLE = 0.0  # Target tilt angle (radians)
if p.getConnectionInfo()["isConnected"]:
    p.disconnect()

p.connect(p.GUI)

# Set the camera to view the target
# p.resetDebugVisualizerCamera(
#     cameraDistance=40,   # Distance of the camera from the origin
#     cameraYaw=0,         # Horizontal angle
#     cameraPitch=-20,     # Vertical angle
#     cameraTargetPosition=[15, 0, 0.5]  # Focus point of the camera
# )

# PID Parameters
pid = PID(50000.0, 1, 1800, setpoint=TARGET_ANGLE)  # Adjust Kp, Ki, Kd as needed
pid.output_limits = (-10, 10)  # Torque limits (adjust as required)

def set_small_tilt(robot_id, tilt_angle):
    """
    Directly sets a small tilt angle for the robot's orientation.

    :param robot_id: ID of the robot in PyBullet
    :param tilt_angle: Desired tilt angle in radians
    """
    # Get current position and orientation of the robot
    position, orientation = p.getBasePositionAndOrientation(robot_id)

    # Convert tilt angle into a quaternion (assuming pitch tilt about Y-axis)
    quaternion = p.getQuaternionFromEuler([tilt_angle, 0, 0])  # Only tilt about Y-axis

    # Reset the base orientation with the tilt
    p.resetBasePositionAndOrientation(robot_id, position, quaternion)

# Initialize Simulation
plane_id = setup_simulation()
p.setTimeStep(0.001)
p.setRealTimeSimulation(0)
robot_id = load_robot(XML_PATH)
target_id=add_target_point(position=[5,0,0.5], color=(1,0,0,1), radius=0.5)

initial_tilt_angle = 5 * np.pi / 180  # 5 degrees tilt
set_small_tilt(robot_id, initial_tilt_angle)
print(f"Applied an initial tilt of {initial_tilt_angle * 180 / np.pi:.2f} degrees")






# Simulation Loop
for step in range(50000):
    # Step simulation
    p.stepSimulation()

    # Get the robot's orientation
    _, orientation = p.getBasePositionAndOrientation(robot_id)
    # print(f"Step {step}: Raw Orientation (Quaternion) : {orientation}")
    tilt_angle, _ , _ = p.getEulerFromQuaternion(orientation)  # Extract tilt angle (pitch)
    # print(f"Step {step}: Calculated Tilt Angle (Radians): {tilt_angle: .4f}")

    # # Compute the torque using PID controller
    torque = -1 * pid(tilt_angle)

    # # Apply torque to wheels
    apply_torque(robot_id, 0, 1, torque)
    # print(f"Step: {step}, Torque: {torque:.4f}, Left Vel: {left_wheel_velocity:.4f}, Right Vel: {right_wheel_velocity:.4f}")
    if step % 100 == 0:  # Log every 100 steps
        print(f"Step: {step}, Tilt Angle: {tilt_angle:.4f}, Torque: {torque:.4f}")


    # # Sleep for timestep
    time.sleep(0.001)

