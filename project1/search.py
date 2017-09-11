# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from game import Directions

    if(problem.isGoalState(problem.getStartState())):
        return [Directions.STOP]

    succStack = util.Stack()
    
    visNodeArray = []
    visNodeArray.append(problem.getStartState())
    prevNodeDict = {}
    dirDict = {}

    goalState = []
    finalNodeArray = []
    finalDircArray = []
    #util.raiseNotDefined()

    for i in problem.getSuccessors(problem.getStartState()):
        prevNodeDict[i[0]] = problem.getStartState()
        dirDict[i[0]] = i[1]
        succStack.push(i)

    while True:
        if (succStack.isEmpty()):
            break
        tempSuccessor = succStack.pop()
        visNodeArray.append(tempSuccessor[0])
        if(problem.isGoalState(tempSuccessor[0])):
            goalState = tempSuccessor[0]
            break
        for j in problem.getSuccessors(tempSuccessor[0]):
            if j[0] not in visNodeArray and j[0] not in prevNodeDict.keys():
                succStack.push(j)
                prevNodeDict[j[0]] = tempSuccessor[0]
                dirDict[j[0]] = j[1]

    while True:
        finalNodeArray.insert(0,goalState)
        if prevNodeDict[goalState] != problem.getStartState():
            goalState = prevNodeDict[goalState]
        else:
            break

    for k in finalNodeArray:
        finalDircArray.append(dirDict[k])

    return finalDircArray

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    from util import Queue
    from game import Directions

    if (problem.isGoalState(problem.getStartState())):
        return [Directions.STOP]

    succQueue = Queue()
    visNodeArray = []
    visNodeArray.append(problem.getStartState())
    prevNodeDict = {}
    dirDict = {}

    goalState = []
    finalNodeArray = []
    finalDircArray = []

    for i in problem.getSuccessors(problem.getStartState()):
        prevNodeDict[i[0]] = problem.getStartState()
        dirDict[i[0]] = i[1]
        succQueue.push(i)

    while True:
        if succQueue.isEmpty():
            break

        nodeTemp = succQueue.pop()
        visNodeArray.append(nodeTemp[0])

        if(problem.isGoalState(nodeTemp[0])):
            goalState = nodeTemp[0]
            break

        for action in problem.getSuccessors(nodeTemp[0]):
            child = action[0]
            if child not in visNodeArray and child not in prevNodeDict.keys():
                succQueue.push(action)
                prevNodeDict[child] = nodeTemp[0]
                dirDict[child] = action[1]

    while True:
        finalNodeArray.insert(0,goalState)
        if prevNodeDict[goalState] != problem.getStartState():
            goalState = prevNodeDict[goalState]
        else:
            break

    for k in finalNodeArray:
        finalDircArray.append(dirDict[k])

    return finalDircArray



def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from game import Directions
    
    actions = []
    
    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), 0)
    visited = []
    
    while (frontier.isEmpty() == False):
        (currentS, currentP, currentC) = frontier.pop()
        if (problem.isGoalState(currentS) == True):
            actions = currentP
            break
        if (visited.count(currentS) == 0):
            visited.append(currentS)
            successors = problem.getSuccessors(currentS)
            for i in range(0,len(successors)):
                (neighbor, direction, cost) = successors[i]
                if (visited.count(neighbor) == 0):
                    frontier.push((neighbor, (currentP +[direction]), (currentC + cost)), (currentC + cost))
                
    return actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def gen_search(problem, frontier, heuristics=None):
    """Node are stored in the frontier and take the following form:
    (state, actions, cost)
    state is what we get from successors
    actions is the set of action to go to this state from the start
    cost is an integer
    """
    def getNodePathCost(search_node):
        """Returns the cost of the search node(dude popped out of getSuccessors funtions)"""
        return search_node[2]
    
    def getNodeState(search_node):
        return search_node[0]
    
    def getNodePrnt(search_node):
        return search_node[3]
    
    def getNodeAction(search_node):
        """Returns the actions needed to reach a given node from its parent i.e. Direction like North, east ,West"""
        return search_node[1]
    
    def getNodeDirs(search_node):
        """returns the directions of an associated node"""
        startState=problem.getStartState()
        directions=[]
        current_node=search_node
        while(getNodeState(current_node) != startState):
            directions.append(getNodeAction(current_node))
            current_node=getNodePrnt(current_node)
            
        directions.reverse()
        return directions
    
    def makeNewNode(successor,parent):
        """Takes in a successor triple and a search node from frontier, and returns a new frontier type node
        a successor triple is: (state, action, stepCost)
          where action is the step to get there from parent,
          stepCost is the incremental cost"""
        state=successor[0]
        par=parent
        dirt=successor[1]
        cost=successor[2]+getNodePathCost(parent)
          
        return (state,dirt,cost,par)
    def isNewNode(state,cost):
        return state not in explored and (state not in front_dict or front_dict[state]>cost)   
    
    startState=problem.getStartState();
    startNode=(startState,None,0,None)
    frontier.push(startNode)
    explored={}
    front_dict={}
    front_dict[startState]=0
    
    while(True):
        if frontier.isEmpty():
            print "no solution Found"
            return None
        currentNode=frontier.pop()
        if(problem.isGoalState(getNodeState(currentNode))):
            break;
        explored[getNodeState(currentNode)]=0
        for successor in problem.getSuccessors(getNodeState(currentNode)):
            newNode = makeNewNode(successor, currentNode)
            
            if isNewNode(getNodeState(newNode), getNodePathCost(newNode)):
                frontier.push(newNode)
                front_dict[getNodeState(newNode)]=getNodePathCost(newNode)
    return getNodeDirs(currentNode) 
    
def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    def getPriority(searchNode):
        cost=searchNode[2]+heuristic(searchNode[0],problem)
        return cost
    frontier=util.PriorityQueueWithFunction(getPriority)
    return gen_search(problem,frontier,heuristic)




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
