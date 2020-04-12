# Controlling a Robot Arm

In this subproject, an agent is trained to control a robot arm by using 
__Twin Delayed Deep Deterministic Policy Gradient (TD3)__.  
<br>
![Trained Agent]()
<br>

## The Environment

In this environment, the agent controls a double jointed arm. A reward of +0.1 is provided for each time step during 
which the agent's hand is in the goal location (the sphere). Thus, the goal of the agent is to maintain its position at 
the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities 
of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry 
in the action vector need to be a number between -1 and 1.

The task is episodic and in order to consider the environment solved the agent must get an average score of +30 
over 100 consecutive episodes.

## Setup: Download the Unity Environment
For this subproject, you will not need to install Unity - the environment is pre-built and can be downloaded from one of 
the links below. You also need to perform the steps described in section "Getting Started" of the main README.md.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Create the directory ./robot_arm/environment, place the unzipped content inside this folder and set the 
variable REACHER_PATH (defined in training.ipynb and evaluation.ipynb) as path to Reacher.exe inside the environment 
folder.