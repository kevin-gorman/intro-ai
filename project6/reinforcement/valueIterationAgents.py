# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # for each iteration
        for its in range(self.iterations):
          # make counter to track highest values
          count = util.Counter()
          # get states
          states = self.mdp.getStates()
          # for lets find the max q values over the states
          for state in states:
            # this way max will update the first time
            max = float("-inf")
            # iterate through the possible actions
            for action in self.mdp.getPossibleActions(state):
              # check if q values is new max, if so update
              if self.computeQValueFromValues(state, action) > max:
                max = self.computeQValueFromValues(state, action)
              # update count
              count[state] = max
          # update values
          self.values = count


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # set Q value to 0, we will update in loop
        Q = 0
        # for each transition state update Q
        for next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            Q += prob * ((self.discount * self.values[next]) + self.mdp.getReward(state, action, next))
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # check if we're at terminal state
        if  self.mdp.isTerminal(state):
            return  None

        # get the list of actions
        acts =  self.mdp.getPossibleActions(state)
        # set max val and actions to the first vale, we will update in loop
        maxAct = acts[0]
        maxVal =  self.getQValue(state, acts[0])

        # iterate over actions to update which one is max an dassociated val
        for act in acts:
           # if max update act and val
           if maxVal <= self.getQValue (state, act):
                maxAct = act
                maxVal = self.getQValue (state, act)

        # return our max act
        return maxAct

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # get states        
        states = self.mdp.getStates()
        num_states = len(states)

        # for each iteration
        for its in range(self.iterations):
          # set working state
          state = states[its % len(states)]
          # check that working state isnt terminal
          if not self.mdp.isTerminal(state):
            # array for values
            vals = []
            # iteration over actions
            for action in self.mdp.getPossibleActions(state):
              # get Q value
              Q = self.computeQValueFromValues(state, action)
              # add to list
              vals.append(Q)
            # add max val to values
            self.values[state] = max(vals)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # make queue to store states in getStates order
        queue = util.PriorityQueue()
        # iterate over states 
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            # vals array to add Q values to
            vals = []
            # for each action get Q values and append to vals
            for action in self.mdp.getPossibleActions(state):
              Q = self.computeQValueFromValues(state, action)
              vals.append(Q)
            # add Q values to the priority queue
            queue.update(state, - abs(max(vals) - self.values[state]))

        # now get predecessor states
        pred = {}
        for state in self.mdp.getStates():
          # check that state is not terminal
          if not self.mdp.isTerminal(state):
            # for each action, for each state
            for action in self.mdp.getPossibleActions(state):
              for next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                # if next is a pred, then add the states
                if next in pred:
                  pred[next].add(state)
                # else set the next
                else:
                  pred[next] = {state}

        # for iterations
        for its in range(self.iterations):
          if queue.isEmpty():
            break
          state = queue.pop()
          if not self.mdp.isTerminal(state):
            values = []
            # for each action get Q value
            for action in self.mdp.getPossibleActions(state):
              Q = self.computeQValueFromValues(state, action)
              values.append(Q)
            # update Q values
            self.values[state] = max(values)

          # for each predecessor
          for p in pred[state]:
            if not self.mdp.isTerminal(p):
              vals = []
              # for each action get Q value
              for action in self.mdp.getPossibleActions(p):
                Q = self.computeQValueFromValues(p, action)
                vals.append(Q)
              # if difference is smaller than theta, update queue
              if abs(max(vals) - self.values[p]) > self.theta:
                queue.update(p, -abs(max(vals) - self.values[p]))