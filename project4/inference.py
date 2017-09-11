# inference.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import random
import busters
import game

class InferenceModule:
  """
  An inference module tracks a belief distribution over a ghost's location.
  This is an abstract class, which you should not modify.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  
  def __init__(self, ghostAgent):
    "Sets the ghost agent for later access"
    self.ghostAgent = ghostAgent
    self.index = ghostAgent.index

  def getJailPosition(self):
     return (2 * self.ghostAgent.index - 1, 1)
    
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.
    
    You must first place the ghost in the gameState, using setGhostPosition below.
    """
    ghostPosition = gameState.getGhostPosition(self.index) # The position you set
    actionDist = self.ghostAgent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist
  
  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index] = game.AgentState(conf, False)
    return gameState
  
  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getNoisyGhostDistances()
    if len(distances) >= self.index: # Check for missing observations
      obs = distances[self.index - 1]
      self.observe(obs, gameState)
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]   
    self.initializeUniformly(gameState)
    
  ######################################
  # Methods that need to be overridden #
  ######################################
  
  def initializeUniformly(self, gameState):
    "Sets the belief state to a uniform prior belief over all positions."
    pass
  
  def observe(self, observation, gameState):
    "Updates beliefs based on the given distance observation and gameState."
    pass
  
  def elapseTime(self, gameState):
    "Updates beliefs for a time step elapsing from a gameState."
    pass
    
  def getBeliefDistribution(self):
    """
    Returns the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence so far.
    """
    pass

class ExactInference(InferenceModule):
  """
  The exact dynamic inference module should use forward-algorithm
  updates to compute the exact belief function at each time step.
  """
  
  def initializeUniformly(self, gameState):
    "Begin with a uniform distribution over ghost positions."
    self.beliefs = util.Counter()
    for p in self.legalPositions: self.beliefs[p] = 1.0
    self.beliefs.normalize()
  
  def observe(self, observation, gameState):
    """
    Updates beliefs based on the distance observation and Pacman's position.
    
    The noisyDistance is the estimated manhattan distance to the ghost you are tracking.
    
    The emissionModel below stores the probability of the noisyDistance for any true 
    distance you supply.  That is, it stores P(noisyDistance | TrueDistance).

    self.legalPositions is a list of the possible ghost positions (you
    should only consider positions that are in self.legalPositions).

    A correct implementation will handle the following special case:
      *  When a ghost is captured by Pacman, all beliefs should be updated so
         that the ghost appears in its prison cell, position self.getJailPosition()

         You can check if a ghost has been captured by Pacman by
         checking if it has a noisyDistance of None (a noisy distance
         of None will be returned if, and only if, the ghost is
         captured).
         
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getPacmanPosition()
    
    "*** YOUR CODE HERE ***"
    # Replace this code with a correct observation update
    # Be sure to handle the jail.
    """
    allPossible = util.Counter()
    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(p, pacmanPosition)
      if emissionModel[trueDistance] > 0: allPossible[p] = 1.0
    allPossible.normalize()
    """
    allPossible = util.Counter()
    """
    for p in self.legalPositions:
      if noisyDistance == None:
        if p == self.getJailPosition():
          allPossible[p] = 1.0
        else:
          allPossible[p] = 0.0
      else:
        trueDistance = util.manhattanDistance(p, pacmanPosition)
        allPossible[p] = emissionModel[trueDistance] * self.beliefs[p]
    """
    if noisyDistance == None:
      allPossible[self.getJailPosition()] = 1
      self.beliefs = allPossible
    else:
      for p in self.legalPositions:
        trueDistance = util.manhattanDistance(p, pacmanPosition)
        if emissionModel[trueDistance] > 0:
          allPossible[p] = self.beliefs[p] * emissionModel[trueDistance]
    allPossible.normalize()
    "*** YOUR CODE HERE ***"
    self.beliefs = allPossible
    
  def elapseTime(self, gameState):
    """
    Update self.beliefs in response to a time step passing from the current state.
    
    The transition model is not entirely stationary: it may depend on Pacman's
    current position (e.g., for DirectionalGhost).  However, this is not a problem,
    as Pacman's current position is known.

    In order to obtain the distribution over new positions for the
    ghost, given its previous position (oldPos) as well as Pacman's
    current position, use this line of code:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    Note that you may need to replace "oldPos" with the correct name
    of the variable that you have used to refer to the previous ghost
    position for which you are computing this distribution.

    newPosDist is a util.Counter object, where for each position p in self.legalPositions,
    
    newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

    (and also given Pacman's current position).  You may also find it useful to loop over key, value pairs
    in newPosDist, like:

      for newPos, prob in newPosDist.items():
        ...

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper methods provided in InferenceModule above:

      1) self.setGhostPosition(gameState, ghostPosition)
          This method alters the gameState by placing the ghost we're tracking
          in a particular position.  This altered gameState can be used to query
          what the ghost would do in this position.
      
      2) self.getPositionDistribution(gameState)
          This method uses the ghost agent to determine what positions the ghost
          will move to from the provided gameState.  The ghost must be placed
          in the gameState with a call to self.setGhostPosition above.
    """
    
    "*** YOUR CODE HERE ***"
    allPossible = util.Counter()

    for p in self.legalPositions:
      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, p))
      for newPos, prob in newPosDist.items():
        allPossible[newPos] += prob * self.beliefs[p]
    self.beliefs = allPossible

  def getBeliefDistribution(self):
    return self.beliefs

class ParticleFilter(InferenceModule):
  """
  A particle filter for approximately tracking a single ghost.
  
  Useful helper functions will include random.choice, which chooses
  an element from a list uniformly at random, and util.sample, which
  samples a key from a Counter by treating its values as probabilities.
  """

  
  def __init__(self, ghostAgent, numParticles=300):
     InferenceModule.__init__(self, ghostAgent);
     self.setNumParticles(numParticles)
  
  def setNumParticles(self, numParticles):
    self.numParticles = numParticles

  
  def initializeUniformly(self, gameState):
    "Initializes a list of particles. Use self.numParticles for the number of particles"
    "*** YOUR CODE HERE ***"
    legalPos = self.legalPositions    
    self.particles = util.Counter()
    for i in xrange(0,self.numParticles):
      self.particles[random.choice(legalPos)] += 1
    self.particles.normalize()
  
  def observe(self, observation, gameState):
    """
    Update beliefs based on the given distance observation. Make
    sure to handle the special case where all particles have weight
    0 after reweighting based on observation. If this happens,
    resample particles uniformly at random from the set of legal
    positions (self.legalPositions).

    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated so
         that the ghost appears in its prison cell, self.getJailPosition()

         You can check if a ghost has been captured by Pacman by
         checking if it has a noisyDistance of None (a noisy distance
         of None will be returned if, and only if, the ghost is
         captured).
         
      2) When all particles receive 0 weight, they should be recreated from the
          prior distribution by calling initializeUniformly. Remember to
          change particles to jail if called for.
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getPacmanPosition()
    "*** YOUR CODE HERE ***"
    """
    beliefs = util.Counter()
    if noisyDistance == None:
      self.particles = [self.getJailPosition() for i in range(self.numParticles)]
    else:
      for p in self.particles:
        beliefs[p] += emissionModel[util.manhattanDistance(p, pacmanPosition)]
      if beliefs.totalCount() != 0:
        self.particles = [util.sample(beliefs) for i in range(self.numParticles)]
      else:
        self.initializeUniformly(gameState)
    util.raiseNotDefined()
    """
    """weights = util.Counter()
    for particle in self.particles:
      trueDistance = util.manhattanDistance(particle, pacmanPosition)
      weights[particle] = emissionModel[trueDistance]*self.particles[particle]
    newParticles = util.Counter()
    if sum(weights.values()) is not 0:
      for i in range(self.numParticles): newParticles[util.sample(weights)] += 1
    else:
      for i in range(self.numParticles): newParticles[random.choice(self.legalPositions)] += 1
    self.particles = newParticles"""

    if self.particles.totalCount() == 0:
        self.initializeUniformly(self.numParticles)
    allPossible = util.Counter()
    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(p, pacmanPosition)
      if emissionModel[trueDistance] > 0:
          allPossible[p] = emissionModel[trueDistance]*self.particles[p]
    allPossible.normalize()
    if noisyDistance == None:
        for p in self.particles:
            if p == self.getJailPosition():
                allPossible[p] = 1.0
            else:
                allPossible[p] = 0.0
    allPossible.normalize()            
    self.particles = util.Counter()
    for i in xrange(0,self.numParticles):
        self.particles[util.sampleFromCounter(allPossible)] += 1
    self.particles.normalize()

  def elapseTime(self, gameState):
    """
    Update beliefs for a time step elapsing.

    As in the elapseTime method of ExactInference, you should use:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    to obtain the distribution over new positions for the ghost, given
    its previous position (oldPos) as well as Pacman's current
    position.
    """
    "*** YOUR CODE HERE ***"
    """self.particles = [util.sample(self.getPositionDistribution(self.setGhostPosition(gameState, particle))) for particle in self.particles]
    util.raiseNotDefined()"""
    """
    tranModel = {}
    for p in self.particles:
      futureGameState = self.setGhostPosition(gameState, p)
      tranModel[p] = self.getPositionDistribution(futureGameState)
    newParticles = util.Counter()
    for p in self.particles:
      for i in range(self.particles[p]):
        newParticles[util.sample(tranModel[p])] += 1
    self.particles = newParticles"""
    allPossible = util.Counter()
    for p in self.legalPositions:
        oldPos = p
        newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
        for newPos, prob in newPosDist.items():
         allPossible[newPos] += prob*self.particles[oldPos]
    allPossible.normalize()
    self.particles = allPossible

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    "*** YOUR CODE HERE ***"
    """
    dist = util.Counter()
    for p in self.particles:
      dist[p] += 1
    dist.normalize()
    return dist
    util.raiseNotDefined()"""
    return self.particles

class MarginalInference(InferenceModule):
  "A wrapper around the JointInference module that returns marginal beliefs about ghosts."

  def initializeUniformly(self, gameState):
    "Set the belief state to an initial, prior value."
    if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
    jointInference.addGhostAgent(self.ghostAgent)
    
  def observeState(self, gameState):
    "Update beliefs based on the given distance observation and gameState."
    if self.index == 1: jointInference.observeState(gameState)
    
  def elapseTime(self, gameState):
    "Update beliefs for a time step elapsing from a gameState."
    if self.index == 1: jointInference.elapseTime(gameState)
    
  def getBeliefDistribution(self):
    "Returns the marginal belief over a particular ghost by summing out the others."
    jointDistribution = jointInference.getBeliefDistribution()
    dist = util.Counter()
    for t, prob in jointDistribution.items():
      dist[t[self.index - 1]] += prob
    return dist
  
class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."

  def __init__(self, numParticles=600):
     self.setNumParticles(numParticles)
  
  def setNumParticles(self, numParticles):
    self.numParticles = numParticles
  
  def initialize(self, gameState, legalPositions):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.ghostAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()
    
  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of ghost positions. Use self.numParticles for the number of particles"
    "*** YOUR CODE HERE ***"
    
    self.particles = []
    for i in range(self.numParticles):
      self.particles.append(tuple([random.choice(self.legalPositions) for j in range(self.numGhosts)]))
    "util.raiseNotDefined()"
    """legalpos = self.legalPositions
    self.particles = []
    for i in xrange(0,self.numParticles):
      ghostTuple = []
      for i in range(self.numGhosts):
          ghostTuple.append(random.choice(legalpos))
      ghostTuple = tuple(ghostTuple)
      self.particles.append(ghostTuple)"""

  def addGhostAgent(self, agent):
    "Each ghost agent is registered separately and stored (in case they are different)."
    self.ghostAgents.append(agent)
    
  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.

    To loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    Then, assuming that "i" refers to the index of the
    ghost, to obtain the distributions over new positions for that
    single ghost, given the list (prevGhostPositions) of previous
    positions of ALL of the ghosts, use this line of code:

      newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, prevGhostPositions),
                                                   i, self.ghostAgents[i])

    Note that you may need to replace "prevGhostPositions" with the
    correct name of the variable that you have used to refer to the
    list of the previous positions of all of the ghosts, and you may
    need to replace "i" with the variable you have used to refer to
    the index of the ghost for which you are computing the new
    position distribution.

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper functions defined below in this file:

      1) setGhostPositions(gameState, ghostPositions)
          This method alters the gameState by placing the ghosts in the supplied positions.
      
      2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
          This method uses the supplied ghost agent to determine what positions 
          a ghost (ghostIndex) controlled by a particular agent (ghostAgent) 
          will move to in the supplied gameState.  All ghosts
          must first be placed in the gameState using setGhostPositions above.
          
          The ghost agent you are meant to supply is self.ghostAgents[ghostIndex-1],
          but in this project all ghost agents are always the same.
    """
    newParticles = []
    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of ghost positions
      "*** YOUR CODE HERE ***"
      "newParticles.append(tuple(newParticle))"
      """newGameState = setGhostPositions(gameState, newParticles)
      for ghost in self.ghostAgents:
        tranModel = getPositionDistributionForGhost(newGameState, ghost.index, ghost)
        newParticle[ghost.index - 1] = util.sample(tranModel)"""
      prevGhostPositions = list(oldParticle)
      for i in range(self.numGhosts):
        newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, prevGhostPositions), i, self.ghostAgents[i])
        newParticle[i] = util.sampleFromCounter(newPosDist)
      """for ghostIndex in range(1,len(gameState.getLivingGhosts())):
        if (gameState.getLivingGhosts()[ghostIndex] == True):
          tmpState = setGhostPositions(gameState, newParticle)
          updatedParticle = util.sample(getPositionDistributionForGhost(tmpState, ghostIndex, self.ghostAgents[ghostIndex - 1]))
          newParticle.pop(ghostIndex - 1)
          newParticle.insert(ghostIndex - 1, updatedParticle)"""

      newParticles.append(tuple(newParticle))
    self.particles = newParticles

  def getJailPosition(self, i):
    return (2 * i + 1, 1);
  
  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy observations.

    As in elapseTime, to loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated so
         that the ghost appears in its prison cell, position self.getJailPosition(i)
         where "i" is the index of the ghost.

         You can check if a ghost has been captured by Pacman by
         checking if it has a noisyDistance of None (a noisy distance
         of None will be returned if, and only if, the ghost is
         captured).

      2) When all particles receive 0 weight, they should be recreated from the
          prior distribution by calling initializeParticles. Remember to
          change ghosts' positions to jail if called for.
    """ 
    pacmanPosition = gameState.getPacmanPosition()
    noisyDistances = gameState.getNoisyGhostDistances()
    if len(noisyDistances) < self.numGhosts: return
    emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

    "*** YOUR CODE HERE ***"
    """weights = [util.Counter() for i in range(self.numGhosts)]

    for p in self.particles:
      for pos in range(self.numGhosts):
        trueDist = util.manhattanDistance(p[pos], pacmanPosition)
        weights[pos][p[pos]] += emissionModels[pos][trueDist]
    newParticles = []
    for i in range(self.numParticles):
      newParticle = []
      for ghost in range(self.numGhosts):
        if noisyDistances[ghost] != 999:
          newParticle.append(util.sample(weights[ghost]))
        else: 
          newParticle.append(tuple([getJailPosition(ghost),1]))
      newParticles.append(tuple(newParticle))
    self.particles = newParticles"""

    """allPossible = util.Counter()
    for p in xrange(self.numParticles):
      weight = 1
      for i in xrange(self.numGhosts):
          if noisyDistances[i]:
              trueDistance = util.manhattanDistance(self.particles[p][i], pacmanPosition)
              emissionModel = emissionModels[i]
              weight *= emissionModel[trueDistance]
      allPossible[self.particles[p]] += weight
    allPossible.normalize()

    self.particles = []
    if allPossible.totalCount() == 0:
        self.initializeUniformly(self.numParticles)
    else:
        for i in xrange(self.numParticles):
            self.particles.append(util.sampleFromCounter(allPossible))
    
    self.newParticles = []
    for p in self.particles:
        p = list(p)
        for i in range(self.numGhosts):
            if noisyDistances[i] == None:
                p[i] = self.getJailPosition(i)
        self.newParticles.append(tuple(p))
    self.particles = self.newParticles  """

    """particleWeightsCounter = util.Counter()
    if (len(emissionModels) != 0):
      for particleArray in self.particles:
          particleArrayList = list(particleArray)
          particle_weights = util.Counter()
          for ghostIndex in range(self.numGhosts):
            if (noisyDistances[ghostIndex] == 999):
              particleArrayList[ghostIndex] = (2 * (ghostIndex+1) - 1, 1)
              particle_weights[ghostIndex] = 1.0
            else:
              trueDist = util.manhattanDistance(pacmanPosition, particleArrayList[ghostIndex])
              weight = (float) (emissionModels[ghostIndex][trueDist])
              particle_weights[ghostIndex] += weight
          weightProduct = 1
          for particle in particle_weights:
            weightProduct *= particle_weights[particle]
          particleArrayList = tuple(particleArrayList)
          particleWeightsCounter[particleArrayList] = weightProduct + particleWeightsCounter[particleArrayList]
      
      if (sum(particleWeightsCounter.values()) == 0):
        self.initializeParticles()
        return

      newParticles = []
      for n in range(len(self.particles)):
        resample = util.sampleFromCounter(particleWeightsCounter)
        newParticles.append(resample)
      self.particles = newParticles"""  

    beliefs = util.Counter()
    for ghost in range(self.numGhosts):
      if not noisyDistances[ghost]:
        for particle in range(self.numParticles):
          listParticle = list(self.particles[particle])
          listParticle[ghost] = self.getJailPosition(ghost)
          self.particles[particle] = tuple(listParticle)
    for particle in self.particles:
      beliefs[particle] += reduce(lambda x, y: x * y, [emissionModels[ghost][util.manhattanDistance(particle[ghost], pacmanPosition)] for ghost in range(self.numGhosts) if noisyDistances[ghost] != None])
    if beliefs.totalCount() != 0:
      self.particles = [util.sample(beliefs) for k in range(self.numParticles)]
    else:
      self.initializeParticles()  
  
  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

# One JointInference module is shared globally across instances of MarginalInference 
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
  """
  Returns the distribution over positions for a ghost, using the supplied gameState.
  """

  # index 0 is pacman, but the students think that index 0 is the first ghost.
  ghostPosition = gameState.getGhostPosition(ghostIndex+1) 
  actionDist = agent.getDistribution(gameState)
  dist = util.Counter()
  for action, prob in actionDist.items():
    successorPosition = game.Actions.getSuccessor(ghostPosition, action)
    dist[successorPosition] = prob
  return dist
  
def setGhostPositions(gameState, ghostPositions):
  "Sets the position of all ghosts to the values in ghostPositionTuple."
  for index, pos in enumerate(ghostPositions):
    conf = game.Configuration(pos, game.Directions.STOP)
    gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
  return gameState  

