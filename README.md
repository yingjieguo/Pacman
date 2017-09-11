# Pacman

	Designed Pacman agents which can rationally eat all dots and avoid ghosts at the same time. 
  
	Search down the road: used depth-first, breadth-first, uniform cost, and A* search algorithms to 
                        solve navigation and traveling problem.
                        
	Implemented Multi-agent: reflex agent, minimax agent, alpha-beta pruning agent, expectimax agent 
                           with different assumptions about how ghosts will act, agent act based on
                           current state and successor state, evaluated states based on the distance
                           to the ghosts and number of remaining food.
                           
	Implemented Q-learning agent: train the agent to learn about value of positions and actions from
                                interactions with the environment and find optimal policy. 
                                
	Designed agents that can locate and eat invisible ghosts: Update the agent's belief distribution over the ghost's position when the ghost moves. 
                          Implemented particle filtering algorithm in hidden Markov model tracks the movement of hidden ghosts. 
