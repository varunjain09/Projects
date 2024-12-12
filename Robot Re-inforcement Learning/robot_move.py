import gym
import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robot_prop import setup_simulation, load_robot, apply_torque
import warnings

# Suppress Gymnasium compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Gymnasium environments.*")

class TwoWheelRobotEnv(gym.Env):
    def __init__(self):
        super(TwoWheelRobotEnv, self).__init__()
        
        # PyBullet setup
        setup_simulation()
        p.setTimeStep(0.01)
        p.setRealTimeSimulation(0)
        self.robot_id = load_robot("two_wheel_robot.xml")

        # Observation space: x position, tilt angle, and angular velocity
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf, -np.pi, -np.inf]), high=np.array([np.inf, np.pi, np.inf]), dtype=np.float32)
        # Action space: torque applied to wheels
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

    def reset(self):
        # Reset the robot's position and tilt
        position = [0, 0, 0.3]
        quaternion = p.getQuaternionFromEuler([0, 0, 0])  # Reset tilt angle to zero
        p.resetBasePositionAndOrientation(self.robot_id, position, quaternion)
        
        # Return initial observation
        return np.array([0.0, 0.0, 0.0])

    def step(self, action):
        # Apply action torque to wheels
        torque = action[0]
        apply_torque(self.robot_id, 0, 1, torque)
        
        # Step simulation
        p.stepSimulation()
        
        # Observe new state
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        x_position = position[0]
        tilt_angle, _, _ = p.getEulerFromQuaternion(orientation)
        tilt_rate = p.getBaseVelocity(self.robot_id)[1][1]
        
        # Reward function: Move forward and keep upright
        reward = -abs(tilt_angle) + 0.1 * x_position  # Reward for moving forward and staying balanced
        
        # Determine if episode is done
        done = abs(tilt_angle) > np.pi / 4 or x_position >= 30  # Episode ends if tilt exceeds 45 degrees or robot reaches 30 meters
        
        # Construct observation
        observation = np.array([x_position, tilt_angle, tilt_rate])
        
        return observation, reward, done, {}

    def render(self, mode='human'):
        # No special rendering needed
        pass

    def close(self):
        p.disconnect()

# Create and wrap the environment
env = DummyVecEnv([lambda: TwoWheelRobotEnv()])

# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)

# Test the agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
