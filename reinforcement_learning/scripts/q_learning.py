#!/bin/usr/env python3
"""
RL Q-Learning Example 

Finding the best way in the node based map for an autonomous robot

----------------------------------------------------
|                                                   | 
|     R1               R2                    R3     | 
|                                                   |
|-------------                          |           |
|            |                          |           |
|     R4     |         R5               |    R6     |
|            |                          |           |
|            |                           -----------|
|                                                   |
|     R7               R8                    R9     |
|                                                   |
----------------------------------------------------


"""

import numpy as np

# define the Q-Learning params

# discount factor
gamma = 0.75

# learning rate
alpha = 0.9

# states

states = {
    'R1' : 0,
    'R2' : 1,
    'R3' : 2,
    'R4' : 3,
    'R5' : 4,
    'R6' : 5,
    'R7' : 6,
    'R8' : 7,
    'R9' : 8
}

# actions

actions = [0,1,2,3,4,5,6,7,8]

# rewards

rewards = np.array([[0,1,0,0,0,0,0,0,0],
                    [1,0,1,0,0,0,0,0,0],
                    [0,1,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,1,0,0],
                    [0,1,0,0,0,0,0,1,0],
                    [0,0,1,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,1,0],
                    [0,0,0,0,1,0,1,0,1],
                    [0,0,0,0,0,0,0,1,0]])

# Maps indices to locations

state_to_location = dict((state,location) for location,state in states.items())

# Define the actions
actions = [0,1,2,3,4,5,6,7,8]

# the function

def get_route(start_node,end_node):
    # copy rewards to set the priority according to user sended end node
    new_rewards = np.copy(rewards)
    end_state = states[end_node]
    new_rewards[end_state,end_state] = 99

    # Q-Learning
    Q = np.array(np.zeros([9,9]))

    for i in range(1000):
        current_state = np.random.randint(0,9)
        traversable_actions = []
        for j in range(9):
            if(new_rewards[current_state,j] > 0):
                traversable_actions.append(j)
        next_state = np.random.choice(traversable_actions)
        # calculate the temporal difference
        td = new_rewards[current_state,next_state] + gamma * Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
        # update Q via Belmann Eqn.
        Q[current_state,next_state] += alpha * td
    
    route = [start_node]
    next_node = start_node
    
    while(next_node != end_node):
        start_state = states[start_node]
        next_state = np.argmax(Q[start_state,])
        next_node = state_to_location[next_state]
        route.append(next_node)
        start_node = next_node
    
    return route

print(get_route('R9', 'R1'))