# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    food = newFood.asList()
    result = float("inf")

    if(len(food) == 0):
        return result
    else:
        closestFood = min([manhattanDistance(newPos, i ) for i in food])
        ghostPos = successorGameState.getGhostPositions()
        closestGhost = min([manhattanDistance(newPos, i ) for i in ghostPos])
        resultPart1 = closestGhost if closestGhost < 4 else 10

        distAllGhost = 0
        for i in ghostPos:
            distAllGhost = distAllGhost + manhattanDistance(i, newPos)
        resultPart2 = distAllGhost if distAllGhost < 5 else 10
      
        result = (resultPart1/10) * (resultPart2/10) * (100000 - 30*len(food) - 3*closestFood/random.randint(1,6))
        return result





def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def max_value(agent,state,depth):
        legalActions = state.getLegalActions(0) 
        if depth > self.depth or len(legalActions) == 0: 
            return self.evaluationFunction(state)
        else: 
            score = -(float("inf"))
            for action in legalActions:
                score = max(score, min_value(0,state.generateSuccessor(0,action),depth))              
        return  score
    

    def min_value(agent,state,depth):
        depth += 1 #return 3
        if depth > self.depth: 
            return self.evaluationFunction(state)
        else: 
            score = float("inf")
            for gitem in range(1,state.getNumAgents()): #ghost1 2 3 
                legalActions = state.getLegalActions(gitem)
                if len(legalActions) == 0: 
                    return  self.evaluationFunction(state)
                else:
                    for action in legalActions:
                       score = min(score, max_value(gitem,state.generateSuccessor(gitem,action),depth))    
        return  score 

    legalActions = gameState.getLegalActions()
    numGhosts = gameState.getNumAgents() - 1
    bestAction = Directions.STOP
    score = -(float("inf"))
    for action in legalActions:
        nextState = gameState.generateSuccessor(0, action)
        prevScore = score
        score = max(score, min_value(0,nextState,1))
        if score > prevScore:
            bestAction = action
    return bestAction
    



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def min_value(agent,state,depth,alpha,belta):
        depth += 1
        if depth > self.depth: 
            return self.evaluationFunction(state)
        else:
            score=float("inf") 
            for gitem in range(1,state.getNumAgents()):
                legalActions=state.getLegalActions(gitem)
                if len(legalActions)==0: 
                    return  self.evaluationFunction(state)
                else:
                    for action in legalActions:
                        score=min(score, max_value(gitem,state.generateSuccessor(gitem,action),depth,alpha,belta)) 
                        if score <= alpha:
                            return  score
                        belta = min(belta,score)
        return  score 
    

    def max_value(agent,state,depth,alpha,belta):
        score = -(float("inf"))
        legalActions = state.getLegalActions(0) 
        if depth > self.depth and len(LegalActions)==0: 
            return self.evaluationFunction(state) 
        else: 
            for action in legalActions:
                score = max(score, min_value(0,state.generateSuccessor(0,action),depth,alpha,belta)) 
                if score >= belta:
                    return  score
                alpha = max(alpha,score)
        return score

    legalActions = gameState.getLegalActions(0)
    bestAction = Directions.STOP
    score = -(float("inf"))
    alpha = -(float("inf"))
    beta = float("inf")
    for action in legalActions:
        nextState = gameState.generateSuccessor(0, action)
        prevScore = score
        score = max(score, min_value(0, gameState.generateSuccessor(0, action), 1, alpha, beta))
        if score > prevScore:
            bestAction = action
        if score >= beta:
            return bestAction
        alpha = max(alpha, score)
    return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    return self.expect_time(gameState, self.depth)[1]

  def expect_time(self, gameState, depth, agentIndex=0):
      if gameState.isWin() or gameState.isLose() or depth == 0:
          return ( self.evaluationFunction(gameState), )

      numAgents = gameState.getNumAgents()
      newDepth = depth if agentIndex != numAgents - 1 else depth - 1
      newAgentIndex = (agentIndex + 1) % numAgents
      legalActions = gameState.getLegalActions(agentIndex)

      try:
          legalActions.remove(DIRECTIONS.STOP)
      except: pass

      actionList = [ \
        (self.expect_time(gameState.generateSuccessor(agentIndex, a), \
         newDepth, newAgentIndex)[0], a) for a in gameState.getLegalActions(agentIndex)]

      if(agentIndex == 0):
          return max(actionList)
      else:
          return ( reduce(lambda s, a: s + a[0], actionList, 0)/len(legalActions), )

    
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  ghostDistances = []
  for gs in newGhostStates:
     ghostDistances += [manhattanDistance(gs.getPosition(),newPos)]

  foodList = newFood.asList()
  wallList = currentGameState.getWalls().asList()
  emptyFoodNeighbors = 0
  foodDistances = []

  def foodNeighbors(foodPos):
     foodNeighbors = [(foodPos[0]-1,foodPos[1])\
     , (foodPos[0],foodPos[1]-1), (foodPos[0],foodPos[1]+1), (foodPos[0]+1,foodPos[1])]
     return foodNeighbors

  for f in foodList:
     neighbors = foodNeighbors(f)
     for fn in neighbors:
         if fn not in wallList and fn not in foodList:
             emptyFoodNeighbors += 1
     foodDistances += [manhattanDistance(newPos,f)]

  inverseFoodDist = 0
  if len(foodDistances) > 0:
     inverseFoodDist = 1.0/(min(foodDistances))


  score = reduce(lambda x, y: x + y, newScaredTimes)\
   + (min(ghostDistances)*((inverseFoodDist**4))) + currentGameState.getScore()-(float(emptyFoodNeighbors)*4.5)
  return score

  

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

