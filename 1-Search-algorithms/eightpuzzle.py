# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:39:52 2017

@author: juhan-peep.ernits
"""

"""
Good to know :
    
    x = column number
    y = row number
    state.index = ranking of the number
    (x,y) = {0;1;2}²


"""

"""
Python language :
    
    list('abc') -> ['a','b','c'] 
    range(1,4) -> [1,2,3,4]
"""
import search

class EightPuzzle(search.Problem):
    
    initial = ()
    
    goal = (1,2,3,4,5,6,7,8,0)
    
    def __init__(self, initial):
        self.initial = initial
    
    def actions(self, state):
        blank = state.index(0)
        y, x = divmod(blank,3)
        action=[]
        if x == 0:
            action.append("right")
        elif x == 1:
            action.append("right")
            action.append("left")
        else:
            action.append("left")
        if y == 0:
            action.append("down")
        elif y == 1:
            action.append("down")
            action.append("up")
        else:
            action.append("up")
        return action
    
    def print_puzzle(self):
        for i in range(0,8):
            print(state.index(i))

    def result(self, state, action):
        blank = state.index(0)
        newState=list(state)
        if action == "right":
            newState[blank]=state[blank+1]
            newState[blank+1]=0
        if action == "left":
            newState[blank]=state[blank-1]
            newState[blank-1]=0
        if action == "down":
            newState[blank]=state[blank+3]
            newState[blank+3]=0
        if action == "up":
            newState[blank]=state[blank-3]
            newState[blank-3]=0
        return tuple(newState)

    def goal_test(self, state):
        return state == self.goal
 
#give the number-1 of nodes in the wrong position     
    def h1(self,node):
        h=0
        for i in range(9):
            if node.state[i]!=self.goal[i]:
                h=h+1
        if h>0:
            h=h-1
        #print(node.state)
        #print("h1: " + str(h))
        return h

#give the number of moves to do to change the position of a node
    def h2(self,node):
        h=0
        for i in range(1,9):
            goal_index = self.goal.index(i)
            current_index = node.state.index(i)
            y1, x1 = divmod(goal_index,3)
            y2, x2 = divmod(current_index,3)
            h=h+abs(y1-y2)+abs(x1-x2)
        #print(node.state)
        #print("h2: " + str(h))
        return h
    



puzzle = EightPuzzle((1,2,3,4,5,6,7,0,8))

puzzle.actions(puzzle.initial)
puzzle.result((1,2,3,4,5,6,7,8,0),"up")





#search.tree_search(puzzle,[])
#search.graph_search(puzzle,[])
puzzle = EightPuzzle((1,2,3,4,5,6,7,0,8))
search.breadth_first_search(puzzle)

puzzle = EightPuzzle((1,2,3,4,5,6,0,7,8))
search.breadth_first_search(puzzle)

instrumentedPuzzle = search.InstrumentedProblem(puzzle)
search.breadth_first_search(instrumentedPuzzle)


def h0(x) :
    return 0

def astar_with_h0 (puzzle):
    search.astar_search(puzzle, h0)

def astar_with_h1(puzzle):
    search.astar_search(puzzle, puzzle.h1)
    
def astar_with_h2(puzzle):
    search.astar_search(puzzle, puzzle.h2)
    
#search.astar_search(puzzle, puzzle.h1)
#search.astar_search(puzzle, puzzle.h2)


# 8 6 7
# 2 5 4
# 3 . 1

hardPuzzle = EightPuzzle((8,6,7,2,5,4,3,0,1))

search.astar_search(hardPuzzle, hardPuzzle.h2)

search.compare_searchers([puzzle],"EightPuzzle 2",
                         searchers=[search.breadth_first_search,
                                    search.iterative_deepening_search,
                                    astar_with_h0,
                                    astar_with_h1,
                                    astar_with_h2])



#def __main__(args):
