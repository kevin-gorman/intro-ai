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
        #return successorGameState.getScore()


        if successorGameState.isWin():
          return 100000

        if len(currentGameState.getGhostStates()) > len(newGhostStates):
          return 10000

        allPellets = newFood.asList()		
        closetsPelletDis = min([util.manhattanDistance(newPos, pellet) for pellet in allPellets])	
        ghostDis = min([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
      
      
        # add points to score of sucessor state if successor state is Pacman eating food.
        if ( currentGameState.getNumFood() > successorGameState.getNumFood() ):
          score = 5 * ghostDis if (ghostDis < 4) else 100 
        else:
          score = 5 * ghostDis - 3 * closetsPelletDis if (ghostDis < 4) else 100 - 3 * closetsPelletDis
        
        if len(currentGameState.getCapsules()):
          capsuleDis = min([util.manhattanDistance(newPos, capsule) for capsule in currentGameState.getCapsules()])
          score += 50 - 3 * capsuleDis if (capsuleDis < 4) else 0
        

        # Always moving
        if action == Directions.STOP:
          score -= 10000

        if ghostDis <= 5 & newScaredTimes[0] == 0:	# Avoid very close ghost
          score -= 100000

        elif ghostDis <= 20 & newScaredTimes[0] != 0:	# Go for a ghost
          score += 150 - 5 * ghostDis
          if ghostDis < 5:
            score += 1000 - 10 * ghostDis

        return score


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
        """
        "*** YOUR CODE HERE ***"
        
        def minScore(state, depth, agentNum):

          lastGhost = gameState.getNumAgents() - 1
          legalActions = state.getLegalActions(agentNum)

          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

          if agentNum == lastGhost:
            return min([maxScore(state.generateSuccessor(agentNum, action), depth - 1)] for action in legalActions)[0]
          else:
            return min([minScore(state.generateSuccessor(agentNum, action), depth, agentNum + 1)] for action in legalActions)[0]

        def maxScore(state, depth):

          legalActions = state.getLegalActions(0)

          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

          return max([minScore(state.generateSuccessor(0, action), depth, 1)] for action in legalActions)[0]

        score = -float('inf')
        for possibleAction in gameState.getLegalActions():

          prevScore = score
          successor = gameState.generateSuccessor(0, possibleAction)
          
          score = max(score, minScore(successor, self.depth, 1))
          if score > prevScore:
            action = possibleAction

        return action
        



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"  
        def minScore(gameState, alpha, beta, agentNum, depth):

          lastGhost = gameState.getNumAgents() - 1

          if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          score = float("inf")
          legalActions = gameState.getLegalActions(agentNum)

          for action in legalActions:
            successor = gameState.generateSuccessor(agentNum, action)

            if agentNum == lastGhost:

              score = min(score, maxScore(successor, alpha, beta, depth - 1))
              if score < alpha:
                return score
              beta = min(beta, score)

            else:
              
              score = min(score, minScore(successor, alpha, beta, agentNum + 1, depth))
              if score < alpha:
                return score
              beta = min(beta, score)
              
          return score


        def maxScore(state, alpha, beta, depth):

          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

          score = -float("inf")
          legalActions = state.getLegalActions(0)

          for action in legalActions:
            successor = state.generateSuccessor(0, action)
            score = max(score, minScore(successor, alpha, beta, 1, depth))

            if score > beta:
              return score

            alpha = max(alpha, score)

          return score
        
        action = Directions.STOP

        score 	= -(float("inf"))
        alpha 	= -(float("inf"))
        beta 	= float("inf")

        for possibleAction in gameState.getLegalActions(0):

          prevscore = score
          nextState = gameState.generateSuccessor(0, possibleAction)
          score = max(score, minScore(nextState, alpha, beta, 1, self.depth))

          if score > prevscore:
            action = possibleAction

          if score > beta:
            return action
          
          alpha = max(alpha, score)

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


        # To be called at end of expimax when choosing max action #
        def expNext(x):
            return expectimax(gameState.generateSuccessor(0, x), 1, 1)
       

        # Expectimax function #
        def expectimax(state, agent, depth):

           
            # If ghost turn (min layer) #
            if agent == state.getNumAgents():


                # If we're not at the bottom depth, we increase depth #
                # and call the function again with it being pacman turn #                 
                if depth != self.depth:
                    return expectimax(state, 0, depth + 1)

                # If we're at max depth just evaluate the actions #
                else:
                    return self.evaluationFunction(state)

                                      
            else:
                # get possible actions #
                actions = state.getLegalActions(agent)

                # if we can't do anything, return evaluating state #                
                if len(actions) == 0:
                    return self.evaluationFunction(state)

                # Here we can do something, so let's find what it is #
                # by calling expectimax on all possible actions #                
                maxAct = (expectimax(state.generateSuccessor(agent, action), agent + 1, depth) for action in actions)


                # if ghost turn (min layer) #
                if agent != 0:
                    # Here get expected random move by ghost #
                    # Choosing uniformly at random #
                    nextEl = list(maxAct)
                    return sum(nextEl) / len(nextEl)   

                # If pacman turn (max layer)#               
                else:
                    # pick maximim of possible next actions #
                    return max(maxAct)
         
               

        # return best option #
        return max(gameState.getLegalActions(0), key=expNext)

        # Call expectiMax with initial depth = 0 and get an action  #
        # Pacman plays first -> agent == 0 or self.index            #
        # We can will more likely than minimax. Ghosts may not play #
        # optimal in some cases                                     #

        return expectiMax(gameState,self.index,0)[1]

        "util.raiseNotDefined()"

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: We want to weigh the food, game score and ghosts into our decision.
      So we need to take in our current position, the ghost states, food location
      and the ability to eat ghosts (so they're properly weighted). Each element
      (food left, distance to food, distance to ghosts, ability to eat ghosts and
      game score) is being taken into account in return. 
      I got my weights by just kind of playing around with them seeing what got higher scores
      Also, I score well on all the tests but the 6th one I get something in the 600s. cant figure out why
    """
    "*** YOUR CODE HERE ***"

    # Get the current position, ghost states, time to eat ghosts, and food states #
    nextPosition = currentGameState.getPacmanPosition()
    nextGhost = currentGameState.getGhostStates()
    eatGhostTime =  [ghostState.scaredTimer for ghostState in nextGhost]
    nextFood = [food for food in currentGameState.getFood().asList() if food]

    
    # get the closest ghost #
    closeGhost = min( manhattanDistance(nextPosition, ghost.configuration.pos) for ghost in nextGhost)

    # get the closest food if food remains #
    if nextFood:
        closeFood = min( manhattanDistance(nextPosition, food) for food in nextFood)
    else:
        closeFood = 0

    # get the time remaining to eat ghosts #
    eatTime = min(eatGhostTime)



    # weigh ghost distance accordingly to if they can be eaten #
    if eatTime == 0:
        # +1 in denominator so no divide by 0 #
        ghostWeight = -2 / (closeGhost + 1)
    else:
        ghostWeight = 0.5 / (closeGhost + 1)

    # slihtly weigh ability to eat ghosts #
    eatWeight = eatTime * 0.5

    # count food left to get #
    foodLeft = -len(nextFood)

    # weigh distance to food #
    distWeight = 0.5 / (closeFood + 1)

    # weigh the current game score #
    gameScore = currentGameState.getScore() * 0.7

    # We want to return the sum of these #
    sum = ghostWeight + eatWeight + foodLeft + distWeight + gameScore
    return sum

# Abbreviation
better = betterEvaluationFunction

