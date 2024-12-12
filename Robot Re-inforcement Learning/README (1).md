**README \- Two-Wheel Robot Training with DQN**  
**Overview**  
This project demonstrates a two-wheel robot learning to navigate towards a target using Deep Q-Network (DQN) approach. The robot is trained in a PyBullet simulation environment, where it learns to turn and move towards a target point. The simulation aims to teach the robot to optimize its path and face the target direction effectively.  
**Project Components**  
**1\. Simulation Setup**  
The project uses PyBullet for the physics simulation and a custom XML-based robot model (two\_wheel\_robot.xml). The main components of the setup are:

* **Simulation Environment**: Initialized with a plane, robot, and target point.  
* **Robot Model**: The two-wheel robot model is loaded from an XML file.  
* **Target Point**: The target is represented as a red sphere placed in front of the robot.

**2\. Deep Q-Network (DQN) Implementation**  
The robot's policy is defined using a DQN. The main features include:

* **Neural Network**: The DQN has two hidden layers with 128 units each, using ReLU activation functions.  
* **Training Loop**: The robot continuously updates its policy to minimize the distance to the target and align itself towards it.

The neural network structure is:

* Input layer: State (4 features: x, y, distance to target, angle to target).  
* Hidden layers: 2 layers of 128 units each.  
* Output layer: 3 actions (turn left, move forward, turn right).

**3\. Reward Function**  
The reward function is designed to encourage the robot to face the target and minimize the distance to it:

* **Distance Penalty**: The reward penalizes the distance between the robot and the target.  
* **Angle Penalty**: There is an additional penalty for not aligning directly with the target direction.

The reward function is defined as:  
reward \= \-distance\_to\_target \- 0.1 \* abs(angle\_to\_target)  
This ensures that the robot prioritizes reducing its distance while also turning towards the target.  
**4\. Target Position Update**  
The target is always positioned directly in front of the robot to help it learn how to face and move in the correct direction. This is controlled by:

* The change\_target\_position() function, which places the target at \[0, \-5, 0.5\] relative to the robot.

**5\. Hyperparameters**

* **Learning Rate**: 1e-3  
* **Discount Factor (Gamma)**: 0.99  
* **Exploration Rate (Epsilon)**: Starts at 1.0 and decays to 0.01 with each episode to balance exploration and exploitation.  
* **Episode Length**: Increased to 1000 steps per episode for extended training.

**How to Run the Simulation**  
**Prerequisites**

* Python 3.x  
* All required packages are listed in requirements.txt

**Instructions**

1. Install the required Python packages:

pip install \-r requirements.txt

2. Connect to PyBullet GUI by running the script:

python robot\_train.py

3. The training will begin with the robot attempting to reach the target point, continuously updating its policy based on the reward function.  
4. The training loop runs indefinitely, with each episode resetting if the robot reaches the target or falls over.

**Increasing Training Speed**  
To increase the training speed, the time.sleep() call was removed to reduce delays between each simulation step.  
**File Descriptions**

* **robot\_train.py**: The main script containing the setup for the robot, target point, DQN policy, and training loop.  
* **main-2.py**: An additional script to manage the environment setup and interactions.  
* **two\_wheel\_robot.xml**: The XML file that defines the two-wheel robot's physical structure.  
* **robot\_prop\_2.py**: Helper functions used for setting up the simulation, applying torques, and controlling the robot's actions.  
* **requirements.txt**: Contains a list of all necessary packages and dependencies for the project.

**Improvements and Considerations**

* **Training Speed**: The time.sleep() function was removed to speed up the training.  
* **Target Placement**: The target point is always directly in front of the robot, but a more complex scenario could involve random target placements for better learning.  
* **Reward Function**: Future improvements can include adding penalties for collisions or rewarding efficient paths.

**Known Issues**

* Occasionally, the robot might fall over due to excessive torque or improper alignment. When this happens, the episode restarts.  
* The model could benefit from more advanced exploration methods or additional sensors for stability.

**Future Work**

* Explore more sophisticated reward functions to handle complex scenarios with dynamic targets.

   
