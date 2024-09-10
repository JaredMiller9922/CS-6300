# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# Note: DFS isn't guaranteed to find the shortest path. It simply checks if a path
#   exists from the root to the target
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    # Create explored set, frontier, and a move_list
    explored = set()
    stack = util.Stack()

    # Ensure the start state isn't the goal
    if (problem.isGoalState(problem.getStartState())):
        return []

    # Add the start state to the frontier
    stack.push((problem.getStartState(), []))
    
    # While stack isn't empty
    while not stack.isEmpty() :
        # Pop the stack and explore popped state 
        cur_state, cur_move_list = stack.pop()

        # Is the state the goal
        if problem.isGoalState(cur_state) : 
            return cur_move_list 

        # Explore state if not previously explored
        if cur_state not in explored:
            for successor in problem.getSuccessors(cur_state) : 
                # Push state : succesor[0] and action : succesor[1]
                stack.push((successor[0], cur_move_list + [successor[1]]))
        # Mark state as explored
        explored.add(cur_state)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Create explored set, frontier, and a move_list
    explored = set()
    queue = util.Queue()

    # Ensure the start state isn't the goal
    if (problem.isGoalState(problem.getStartState())):
        return []

    # Add the start state to the frontier
    queue.push((problem.getStartState(), []))
    
    # While queue isn't empty
    while not queue.isEmpty() :
        # Pop the queue and explore popped state 
        cur_state, cur_move_list = queue.pop()

        # Is the state the goal
        if problem.isGoalState(cur_state) : 
            return cur_move_list 

        # Explore state if not previously explored
        if cur_state not in explored:
            for successor in problem.getSuccessors(cur_state) : 
                # Push state : succesor[0] and action : succesor[1]
                queue.push((successor[0], cur_move_list + [successor[1]]))
        # Mark state as explored
        explored.add(cur_state)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Create explored set, frontier, and a move_list
    explored = set()
    p_queue = util.PriorityQueue()

    # Ensure the start state isn't the goal
    if (problem.isGoalState(problem.getStartState())):
        return []

    # Add the start state to the frontier
    p_queue.update((problem.getStartState(), [], 0), 0)
    
    # While p_queue isn't empty
    while not p_queue.isEmpty() :
        # Pop the queue and explore popped state 
        cur_state, cur_move_list, cur_cost_to_come = p_queue.pop()

        # Is the state the goal
        if problem.isGoalState(cur_state) : 
            return cur_move_list 

        # Explore state if not previously explored
        if cur_state not in explored:
            for successor in problem.getSuccessors(cur_state) : 
                # Push state : succesor[0] and action : succesor[1]
                p_queue.update((successor[0], cur_move_list + [successor[1]], cur_cost_to_come + successor[2]), 
                                cur_cost_to_come + successor[2])
        # Mark state as explored
        explored.add(cur_state)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Create explored set, frontier, and a move_list
    explored = set()
    p_queue = util.PriorityQueue()

    # Ensure the start state isn't the goal
    if (problem.isGoalState(problem.getStartState())):
        return []

    # Add the start state to the frontier
    p_queue.update((problem.getStartState(), [], 0), 0)
    
    # While p_queue isn't empty
    while not p_queue.isEmpty() :
        # Pop the queue and explore popped state 
        cur_state, cur_move_list, cur_cost_to_come = p_queue.pop()

        # Is the state the goal
        if problem.isGoalState(cur_state) : 
            return cur_move_list 

        # Explore state if not previously explored
        if cur_state not in explored:
            for successor in problem.getSuccessors(cur_state) : 
                # Push state : succesor[0] and action : succesor[1]
                p_queue.update((successor[0], cur_move_list + [successor[1]], cur_cost_to_come + successor[2]),
                                cur_cost_to_come + successor[2] + heuristic(successor[0], problem))
        # Mark state as explored
        explored.add(cur_state)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
