import pybullet as p
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from robot_prop_2 import setup_simulation, load_robot, apply_torque, add_target_point
import time

# Neural Network for DQN Policy
class DQNPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.output_layer(x)
        return q_values

# Hyperparameters
learning_rate = 1e-3
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
T_horizon = 1000  # Increased from 200 to 1000 to allow longer training episodes

# Environment and Robot Setup
XML_PATH = "two_wheel_robot.xml"
p.connect(p.GUI)
plane_id = setup_simulation()
robot_id = load_robot(XML_PATH)

# Function to change target position to face the robot
def change_target_position():
    return [0, -5, 0.5]  # Set target directly in front of the robot

target_position = change_target_position()
target_id = add_target_point(position=target_position, color=(1, 0, 0, 1), radius=0.5)

# DQN Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = DQNPolicy(state_dim=4, action_dim=3).to(device)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# DQN Training Loop
while True:
    # Reset robot to initial position and orientation
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.3], p.getQuaternionFromEuler([0, 0, 0]))
    target_position = change_target_position()
    p.resetBasePositionAndOrientation(target_id, target_position, [0, 0, 0, 1])
    distance_to_target = np.linalg.norm(np.array(target_position[:2]) - np.array([0, 0]))  # Initialize distance to target
    state = np.array([0, 0, distance_to_target, 0], dtype=np.float32)  # Initialize state as [x, y, distance_to_target, angle_to_target]
    total_reward = 0

    for t in range(T_horizon):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Epsilon-Greedy Action Selection
        if np.random.rand() < epsilon:
            action = np.random.choice(3)
        else:
            with torch.no_grad():
                q_values = policy(state_tensor)
                action = torch.argmax(q_values).item()

        # Convert action to torque
        if action == 0:
            torque = -10
        elif action == 1:
            torque = 0
        else:
            torque = 10

        # Apply torque to robot
        apply_torque(robot_id, 0, 1, torque)
        p.stepSimulation()

        # Update state based on robot position and target
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        target_position, _ = p.getBasePositionAndOrientation(target_id)
        distance_to_target = np.linalg.norm(np.array(target_position[:2]) - np.array(position[:2]))
        robot_direction = np.arctan2(target_position[1] - position[1], target_position[0] - position[0])
        tilt_angle, _, _ = p.getEulerFromQuaternion(orientation)
        angle_to_target = robot_direction - tilt_angle
        next_state = np.array([position[0], position[1], distance_to_target, angle_to_target], dtype=np.float32)

        # Calculate reward
        if abs(angle_to_target) < np.pi / 4:  # High reward for moving towards the target
            reward = 50 - distance_to_target
        else:  # Heavy penalty for moving away from the target
            reward = -100 - distance_to_target
        total_reward += reward

        # Convert next_state to tensor
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        # Compute target Q-value
        with torch.no_grad():
            target_q = reward + gamma * torch.max(policy(next_state_tensor)).item()

        # Compute current Q-value
        q_values = policy(state_tensor)
        current_q = q_values[0, action]

        # Compute loss
        loss = criterion(current_q, torch.FloatTensor([target_q]).to(device))

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        state = next_state

        # Monitor the robot's position in each step
        print(f"Step {t}, Position: {position}, Distance to Target: {distance_to_target}, Angle to Target: {angle_to_target}")

        # Terminate if robot reaches the target
        if distance_to_target < 0.1:
            print("Target reached!")
            break

        # Restart training loop if the robot falls
        if position[2] < 0.1:  # Assuming z < 0.1 means the robot has fallen
            print("Robot has fallen, restarting episode...")
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode finished, Total Reward: {total_reward}")

p.disconnect()
