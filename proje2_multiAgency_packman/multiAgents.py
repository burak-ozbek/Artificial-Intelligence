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


        newFoodList = newFood.asList()
        newGhostList = [g.getPosition() for g in newGhostStates]

        foodDistances = [util.manhattanDistance(newPos, f) for f in newFoodList]
        ghostDistances = [util.manhattanDistance(newPos, g) for g in newGhostList]

        closestFoodDistance = min(foodDistances) if len(foodDistances) > 0 else 0
        closestGhostDistance = min(ghostDistances) if len(ghostDistances) > 0 else 0
        newFoodCount = len(newFoodList)

        return 1/(closestFoodDistance + 0.0001) - 3*newFoodCount - 2/(closestGhostDistance + 0.0001)

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
        #util.raiseNotDefined()

        def search_depth(state, depth, agent):
            current_agent = state.getNumAgents()
            current_depth = self.depth

            if agent == current_agent:
                if depth == current_depth:
                    return self.evaluationFunction(state)
                else:
                    return search_depth(state, depth + 1, 0)
            else:
                actions = state.getLegalActions(agent)

                if len(actions) == 0:
                    return self.evaluationFunction(state)

                next_node = (search_depth(state.generateSuccessor(agent, action),depth, agent + 1) for action in actions)

                return (max(next_node) if agent == 0 else min(next_node))

        return max(gameState.getLegalActions(0), key = lambda x: search_depth(gameState.generateSuccessor(0, x), 1, 1))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        def maximum_value(agent, depth, game_state, alpha, beta):
            val = float("-inf")
            for next_node in game_state.getLegalActions(agent):
                val = max(val, alpha_beta_prune(1, depth, game_state.generateSuccessor(agent, next_node), alpha, beta))
                if val > beta:
                    return val
                alpha = max(alpha, val)
            return val

        def minimum_value(agent, depth, game_state, alpha, beta):
            val = float("inf")
            next_agent = agent + 1
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1
            next_nodes = game_state.getLegalActions(agent)
            for next_node in next_nodes:
                val = min(val, alpha_beta_prune(next_agent, depth, game_state.generateSuccessor(agent, next_node), alpha, beta))
                if val < alpha:
                    return val
                beta = min(beta, val)
            return val

        def alpha_beta_prune(agent, depth, game_state, alpha, beta):
            current_depth = self.depth
            if game_state.isLose() or game_state.isWin() or depth == current_depth:
                return self.evaluationFunction(game_state)

            if agent == 0:
                return maximum_value(agent, depth, game_state, alpha, beta)
            else:
                return minimum_value(agent, depth, game_state, alpha, beta)

        utility = float("-inf")
        action = Directions.WEST
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            ghost_Value = alpha_beta_prune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if ghost_Value > utility:
                utility = ghost_Value
                action = agentState
            if utility > beta:
                return utility
            alpha = max(alpha, utility)


        return action

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
        #util.raiseNotDefined()
        def search_depth(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                next_states = (search_depth(1, depth, gameState.generateSuccessor(agent, newState))
                               for newState in gameState.getLegalActions(agent))
                return max(next_states)
            else:
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                next_states = (search_depth(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
                cnt_child = len(gameState.getLegalActions(agent))
                stateProbability = sum(next_states) / float(cnt_child)

                return stateProbability

        maximum_value = float("-inf")
        action = Directions.WEST
        for agent_Current_State in gameState.getLegalActions(0):
            utility = search_depth(1, 0, gameState.generateSuccessor(0, agent_Current_State))
            if utility > maximum_value or maximum_value == float("-inf"):
                maximum_value = utility
                action = agent_Current_State

        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    min_food_distance = -1

    for food in newFoodList:
        distance = util.manhattanDistance(newPos, food)
        if min_food_distance >= distance or min_food_distance == -1:
            min_food_distance = distance

    distances_to_ghosts = 1
    proximity_to_ghosts = 0

    for ghost_state in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost_state)
        distances_to_ghosts += distance
        if distance <= 1:
            proximity_to_ghosts += 1

    newCapsule = currentGameState.getCapsules()
    numberOfCapsules = len(newCapsule)
    newScore = currentGameState.getScore() + (1 / float(min_food_distance)) - (1 / float(distances_to_ghosts)) - proximity_to_ghosts - numberOfCapsules

    return newScore

# Abbreviation
better = betterEvaluationFunction
