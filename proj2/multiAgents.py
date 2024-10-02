# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    # This is a Q fuction Q(s,a) -> A number
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
        successorGameState = currentGameState.generatePacmanSuccessor(action) # Next state after taking "action" in "currentGameState" 
        newPos = successorGameState.getPacmanPosition() 
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        stateValue = 0

        # Reward being closer to food
        if newFood.asList():
            closestFood = min(
                manhattanDistance(newPos, futurePos) 
                for futurePos in newFood.asList()
            ) 
            stateValue += 5 / closestFood if closestFood != 0 else 0

        # Get ghost posistions so Pacman can be rewarded or punished
        ghostPosistions = [ghostState.getPosition() for ghostState in newGhostStates]

        # Punish for being closer to ghosts
        if ghostPosistions :
            for curGhostPos in ghostPosistions :
                curGhostDis = manhattanDistance(newPos, curGhostPos)
                stateValue -= 3 / curGhostDis if curGhostDis != 0 else 0

        # Punish Pacman if his next posistion is a ghost posistion
        if newPos in ghostPosistions :
            stateValue -= 50

        # What score do we have in the next state
        stateValue += successorGameState.getScore()

        # Punish pacman for staying still
        if currentGameState.getPacmanPosition() == newPos :
            stateValue -= 1

        # Reward for increasing scared times
        for scaredTime in newScaredTimes :
            stateValue += scaredTime / 2
        # return successorGameState.getScore()
        return stateValue

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth

        # To start we want to find the Pacmans max value
        maxAction = None
        value = float('-inf')
        for action in gameState.getLegalActions(0) :
            nextState = gameState.generateSuccessor(0, action)
            nextStateValue = self.minValue(nextState, 1, depth, 0)
            if nextStateValue > value :
                value = nextStateValue
                maxAction = action

        return maxAction

    def maxValue(self, gameState, agentIndex, maxDepth, curDepth) :
        # Make sure we haven't gone too deep
        if curDepth == maxDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Initialize the value to be negative infinity because this is a max 
        value = float('-inf')

        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions :
            nextState = gameState.generateSuccessor(agentIndex, action)
            # Don't increment depth here. Next index should always be 1
            value = max(value, self.minValue(nextState, agentIndex + 1, maxDepth, curDepth))

        return value

    def minValue(self, gameState, agentIndex, maxDepth, curDepth) :
        # Make sure we haven't gone too deep
        if curDepth == maxDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Initialize the value to be infinity because this is a min
        value = float('inf')

        # Initialize numAgents here because it is used multiple times
        numAgents = gameState.getNumAgents()

        # Find all legal actions and calculate their value
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions :
            nextState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex < numAgents - 1:
                value = min(value, self.minValue(nextState, agentIndex + 1, maxDepth, curDepth))
            else :
                value = min(value, self.maxValue(nextState, 0, maxDepth, curDepth + 1))

            # value = min(value, self.minValue(nextState, agentIndex + 1, maxDepth, curDepth) if agentIndex < gameState.getNumAgents() - 1 
                            # else self.maxValue(nextState, 0, maxDepth, curDepth + 1))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth

        # To start we want to find the Pacmans max value
        maxAction = None
        value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0) :
            nextState = gameState.generateSuccessor(0, action)
            nextStateValue = self.minValue(nextState, 1, depth, 0, alpha, beta)
            if nextStateValue > value :
                value = nextStateValue
                maxAction = action
            # This step is critical unless you don't want your alpha from your left side to propegate
            alpha = max(alpha, nextStateValue)

        return maxAction

    def maxValue(self, gameState, agentIndex, maxDepth, curDepth, alpha, beta) :
        # Make sure we haven't gone too deep
        if curDepth == maxDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Initialize the value to be negative infinity because this is a max 
        value = float('-inf')

        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions :
            nextState = gameState.generateSuccessor(agentIndex, action)
            # Don't increment depth here. Next index should always be 1
            value = max(value, self.minValue(nextState, agentIndex + 1, maxDepth, curDepth, alpha, beta))

            if value > beta :
                return value
            alpha = max(alpha, value)

        return value

    def minValue(self, gameState, agentIndex, maxDepth, curDepth, alpha, beta) :
        # Make sure we haven't gone too deep
        if curDepth == maxDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Initialize the value to be infinity because this is a min
        value = float('inf')

        # Initialize numAgents here because it is used multiple times
        numAgents = gameState.getNumAgents()

        # Find all legal actions and calculate their value
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions :
            nextState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex < numAgents - 1:
                value = min(value, self.minValue(nextState, agentIndex + 1, maxDepth, curDepth, alpha, beta))
            else :
                value = min(value, self.maxValue(nextState, 0, maxDepth, curDepth + 1, alpha, beta))

            if value < alpha :
                return value
            beta = min(beta, value)


            # value = min(value, self.minValue(nextState, agentIndex + 1, maxDepth, curDepth) if agentIndex < gameState.getNumAgents() - 1 
                            # else self.maxValue(nextState, 0, maxDepth, curDepth + 1))
        return value

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
        # Initialize needed variables
        maxAction = None
        value = float('-inf')
        # For loop that loops through all of Pacmans moves
        for action in gameState.getLegalActions(0) :
            nextState = gameState.generateSuccessor(0, action)
            nextStateEV = self.expectimax(nextState, 1, self.depth, 0)
            if nextStateEV > value :
                maxAction = action
                value = nextStateEV

        return maxAction

    def expectimax(self, gameState, agentIndex, maxDepth, curDepth) :
        # Make sure we haven't gone too deep
        if curDepth == maxDepth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Initialize variables we will need later
        legalActions = gameState.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()

        # If we are moving as Pacman
        if agentIndex == 0 :
            # For loop that loops through all of Pacmans moves
            value = float('-inf')
            for action in legalActions :
                nextState = gameState.generateSuccessor(0, action)
                nextStateValue = self.expectimax(nextState, 1, maxDepth, curDepth)
                value = max(value, nextStateValue)
            return value

        else :
            expectedSV = 0
            for action in legalActions : 
               # calculate expected value for each succesive state
               nextState = gameState.generateSuccessor(agentIndex, action)

               # if this is the last ghost increment depth and go to Pacman
               if agentIndex == numAgents - 1 :
                   expectedSV += self.expectimax(nextState, 0, maxDepth, curDepth + 1)
               else :
                   expectedSV += self.expectimax(nextState, agentIndex + 1, maxDepth, curDepth)
            # Because everyting is uniformally distributed we can do this at the end
            return expectedSV / len(legalActions) 

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
    stateValue = 0

    # Reward being closer to food
    if newFood.asList():
        closestFood = min(
            manhattanDistance(newPos, futurePos) 
            for futurePos in newFood.asList()
        ) 
        stateValue += 5 / closestFood if closestFood != 0 else 0

    # Get ghost posistions so Pacman can be rewarded or punished
    ghostPosistions = [ghostState.getPosition() for ghostState in newGhostStates]

    # Punish for being closer to ghosts
    if ghostPosistions :
        for curGhostPos in ghostPosistions :
            curGhostDis = manhattanDistance(newPos, curGhostPos)
            stateValue -= 3 / curGhostDis if curGhostDis != 0 else 0

    # Punish Pacman if his next posistion is a ghost posistion
    if newPos in ghostPosistions :
        stateValue -= 50

    # What score do we have in the next state
    stateValue += currentGameState.getScore()

    # Punish pacman for staying still
    if currentGameState.getPacmanPosition() == newPos :
        stateValue -= 1

    # Reward for increasing scared times
    for scaredTime in newScaredTimes :
        stateValue += scaredTime / 2

    # return successorGameState.getScore()
    return stateValue 

# Abbreviation
better = betterEvaluationFunction
