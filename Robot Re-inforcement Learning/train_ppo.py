import gym
from stable_baselines3 import PPO
from robot_env import RobotEnv

# Create the environment
env = RobotEnv(xml_path="two_wheel_robot.xml", target_position=(30, 0))

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_robot_model")
env.close()