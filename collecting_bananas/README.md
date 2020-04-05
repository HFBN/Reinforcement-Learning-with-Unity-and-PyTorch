# Collecting Bananas

Inside this directory, an agent is trained to navigate (and collect bananas!) in a large, square world.  
<br>
![Trained Agent](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)
<br>

## The Environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around 
the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete
 actions are available, corresponding to:

- **`0`** - move forward.  
- **`1`** - move backward.  
- **`2`** - turn left.  
- **`3`** - turn right.  

The task is episodic, and in order to consider the environment solved, we want the agent to get an average score of 
+13 over 100 consecutive episodes.

## Download the Unity Environment
For this project, you will not need to install Unity - the environment is pre-built and you can download it from one of 
the links below.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, place the unzipped content in collecting_bananas/environment/banana_windows.