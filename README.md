# Reinforcement Learning with Unity and Pytorch
This repository contains several intuitive, high quality implementations 
of state-of-the-art reinforcement learning algorithms solving Unity-based environments. 

## Table of Contents
- __collecting_bananas__  
  This directory contains everything you need to train and evaluate an Agent that learns to collect 
  bananas in a large environment.
- __robot_arm__  
  This directory contains everything you need to train and evaluate an Agent that learns to control a robotic arm.
- __playing_tennis__  
  This directory contains everything you need to train and evaluate two Agents that learn to play tennis in a 
  Multi-Agent-Scenario.
  
All environments were built in Unity. For more information on any of the subproject, please refer to the README.md 
inside the subproject's directory.

## Getting started
To set up the necessary dependencies, follow the steps described here.  

_(__Disclaimer__: This steps assume that you are using Anaconda. If you don't, I highly recommend you to do so.  
You also might want to use Jupyter Notebooks for a more visual experience during training / evaluation.)_

1. Create and activate a new environment:  
   ```
   Linux or Mac:
   conda create --name rlup python=3.6
   source activate rlup

   Windows:
   conda create --name rlup python=3.6 
   activate rlup
   ```
2. Clone the repository (if you haven't already!), and cd into the setup folder. Then, install several dependencies.
   ```
   git clone https://github.com/HFBN/Reinforcement-Learning-with-Unity-and-PyTorch.git
   cd deep-reinforcement-learning/python
   pip install .
   ```
3. There have been some issues with Windows and pytorch as well as TensorFlow. Therefore, I recommend to install them
manually.
   ```
   conda install -c conda-forge tensorflow==1.14.0
   conda install -c pytorch pytorch==1.4.0
   ```
4. Create an IPython kernel for the rlup environment.
   ```
   python -m ipykernel install --user --name rlup --display-name "rlup"
   ```
   Before running code in a notebook, change the kernel to match the rlup environment by using the drop-down Kernel menu.
   
## Author
- Jonas J. Mühlbauer, Artifical Intelligence Consultant