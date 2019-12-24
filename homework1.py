import search

#We here define the relative position of the neighbours of a piece in the list

up    = -6
down  = 6
left  = -1
right = 1

#Limits of the puzzle
limit_line_up = 0
limit_line_down = 4
limit_column_left = 0
limit_column_right = 5

previous_actions=[]

blank = {-1, -2}
bsquare = 1
square = {2, 3, 4, 5, 6, 7}
rectangle = {8, 9, 10}
corner = {11, 12, 13, 14}
anything = {-1, -2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

#5 rows, 6 columns. Each number represents a kind of piece. Negative are the blanks

state_hard = (-1,11,11,12,13,13,
             -2,11,12,12,13,8,
             2,3,1,1,9,8,
             4,5,1,1,9,14,
             6,7,10,10,14,14)

state_task = (1,1,2,3,8,8,
             1,1,4,5,9,9,
             -1,-2,6,7,10,10,
             11,11,12,13,13,14,
             11,12,12,13,14,14)

state_test = (11,11,12,2,8,8,
         11,12,12,3,-1,9,
         -2,7,6,10,10,9,
         13,13,14,1,1,5,
         13,14,14,1,1,4)

state3 = (11,11,12,2,8,8,
         11,12,12,3,9,9,
         13,13,10,10,5,-1,
         13,14,6,1,1,4,
         14,14,7,1,1,-2)


state2 = (11,11,12,2,8,8,
         11,12,12,3,9,9,
         13,13,10,10,5,-1,
         13,14,6,1,1,4,
         14,14,7,1,1,-2)

state1 = (11,11,12,2,8,8,
         11,12,12,3,9,9,
         13,13,10,10,4,5,
         13,14,6,1,1,-1,
         14,14,7,1,1,-2)

state0 = (11,11,12,2,8,8,
         11,12,12,3,9,9,
         13,13,10,10,4,5,
         13,14,6,-1,1,1,
         14,14,7,-2,1,1)

#tuple([tuple (v) for v in state])

#Displays the list as a puzzle of 6x5 pieces
def displayPuzzle (state):
    for i in range (1,31):
        print (state[i-1], end=' ')
        if ((i%6 == 0) & (i!=1)):
            print("\n")
    print("\n")


class HW1Puzzle(search.Problem):

    initial = ()

    goal = (anything,anything,anything,anything,anything,anything,
            anything,anything,anything,anything,anything,anything,
            anything,anything,anything,anything,anything,anything,
            anything,anything,anything,anything,bsquare,bsquare,
            anything,anything,anything,anything,bsquare,bsquare)

    #Checks when the goal is achieved.
    #We consider that having the big square in the last case is a sufficient condition

    def goal_test(self, state):
        if state[22] == self.goal[22] and state[23] == self.goal[23] and state[28] == self.goal[28] and state[29] == self.goal[29]:
            return True
    
    def __init__(self, initial):
        self.initial = initial

    def actions(self, state):
        blank1 = state.index(-1)
        blank2 = state.index(-2)
        line1, column1 = divmod(blank1,6)  #peut-être inversés
        line2, column2 = divmod(blank2,6)
        action=[]
        global previous_actions
        #print("Actions précédentes:", previous_actions)

        ''' Définition des collisions entre blocs'''

        ''' STEP 0 : One-blank situation '''

        if column1 < limit_column_right:
            if state[blank1+right] in square:
                if "left_square1" not in previous_actions: action.append("right_square1")
        if column2 < limit_column_right:
            if state[blank2+right] in square:
                if "left_square2" not in previous_actions: action.append("right_square2")
        if column1 < limit_column_right-1:
            if state[blank1+right] in rectangle and state[blank1+2*right] == state[blank1+right]:
                if "left_hrectangle1" not in previous_actions: action.append("right_hrectangle1")
        if column2 < limit_column_right-1:
            if state[blank2+right] in rectangle and state[blank2+2*right] == state[blank2+right]:
                if "left_hrectangle2" not in previous_actions: action.append("right_hrectangle2")


        if column1 > limit_column_left:
            if state[blank1+left] in square:
                if "right_square1" not in previous_actions: action.append("left_square1")
        if column2 > limit_column_left:
            if state[blank2+left] in square:
                if "right_square2" not in previous_actions: action.append("left_square2")
        if column1 > limit_column_left+1:
            if state[blank1+left] in rectangle and state[blank1+2*left] == state[blank1+left]:
                if "right_hrectangle1" not in previous_actions: action.append("left_hrectangle1")
        if column2 > limit_column_left+1:
            if state[blank2+left] in rectangle and state[blank2+2*left] == state[blank2+left]:
                if "right_hrectangle2" not in previous_actions: action.append("left_hrectangle2")


        if line1 < limit_line_down:
            if state[blank1+down] in square:
                if "up_square1" not in previous_actions: action.append("down_square1")
        if line2 < limit_line_down:
            if state[blank2+down] in square:
                if "up_square2" not in previous_actions: action.append("down_square2")
        if line1 < limit_line_down-1:
            if state[blank1+down] in rectangle and state[blank1+2*down] == state[blank1+down]:
                if "up_vrectangle1" not in previous_actions: action.append("down_vrectangle1")
        if line2 < limit_line_down-1:
            if state[blank2+down] in rectangle and state[blank2+2*down] == state[blank2+down]:
                if "up_vrectangle2" not in previous_actions: action.append("down_vrectangle2")


        if line1 > limit_line_up:
            if state[blank1+up] in square:
                if "down_square1" not in previous_actions: action.append("up_square1")
        if line2 > limit_line_up:
            if state[blank2+up] in square:
                if "down_square2" not in previous_actions: action.append("up_square2")
        if line1 > limit_line_up+1:
            if state[blank1+up] in rectangle and state[blank1+2*up] == state[blank1+up]:
                if "down_vrectangle1" not in previous_actions: action.append("up_vrectangle1")
        if line2 > limit_line_up+1:
            if state[blank2+up] in rectangle and state[blank2+2*up] == state[blank2+up]:
                if "down_vrectangle2" not in previous_actions: action.append("up_vrectangle2")


        ''' Two-blanks situation '''

        ''' STEP 1 : blank1 and blank2 are horizontally side by side'''

        if column1 < limit_column_right:
            if state[blank1+right] == state[blank2]:
                if line1 > limit_line_up:
                    if state[blank1+up] == bsquare and state[blank2+up] == bsquare:
                        if "down_bsquare" not in previous_actions: action.append("up_bsquare")
                    if state[blank1+up] in corner and state[blank2+up] == state[blank1+up]:
                        if "down_corner" not in previous_actions: action.append("up_corner")
                    if state[blank1+up] in rectangle and state[blank2+up] == state[blank1+up]:
                        if "down_hrectangle" not in previous_actions: action.append("up_hrectangle")

                if line1 < limit_line_down:
                    if state[blank1+down] == bsquare and state[blank2+down] == bsquare:
                        if "up_bsquare" not in previous_actions: action.append("down_bsquare")
                    if state[blank1+down] in corner and state[blank2+down] == state[blank1+down]:
                        if "up_corner" not in previous_actions: action.append("down_corner")
                    if state[blank1+down] in rectangle and state[blank2+down] == state[blank1+down]:
                        if "up_hrectangle" not in previous_actions: action.append("down_hrectangle")

        if column1 > limit_column_left:
            if state[blank1+left] == state[blank2]:
                if line1 > limit_line_up:
                    if state[blank1+up] == bsquare and state[blank2+up] == bsquare:
                        if "down_bsquare" not in previous_actions: action.append("up_bsquare")
                    if state[blank1+up] in corner and state[blank2+up] == state[blank1+up]:
                        if "down_corner" not in previous_actions: action.append("up_corner")
                    if state[blank1+up] in rectangle and state[blank2+up] == state[blank1+up]:
                        if "down_hrectangle" not in previous_actions: action.append("up_hrectangle")

                if line1 < limit_line_down:
                    if state[blank1+down] == bsquare and state[blank2+down] == bsquare:
                        if "up_bsquare" not in previous_actions: action.append("down_bsquare")
                    if state[blank1+down] in corner and state[blank2+down] == state[blank1+down]:
                        if "up_corner" not in previous_actions: action.append("down_corner")
                    if state[blank1+down] in rectangle and state[blank2+down] == state[blank1+down]:
                        if "up_hrectangle" not in previous_actions: action.append("down_hrectangle")

        ''' STEP 2 : blank1 and blank2 are vertically side by side'''

        if line1 > limit_line_up:
            if (state[blank1+up] == state[blank2]):
                if column1 < limit_column_right:
                    if state[blank1+right] == bsquare and state[blank2+right] == bsquare:
                        if "left_bsquare" not in previous_actions: action.append("right_bsquare")
                    if (state[blank1+right] in corner) and (state[blank2+right] == state[blank1+right]):
                        if "left_corner" not in previous_actions: action.append("right_corner")
                    if ((state[blank1+right] in rectangle) and (state[blank2+right] == state[blank1+right])):
                        if "left_vrectangle" not in previous_actions: action.append("right_vrectangle")

                if column1 > limit_column_left:
                    if state[blank1+left] == bsquare and state[blank2+left] == bsquare:
                        if "right_bsquare" not in previous_actions: action.append("left_bsquare")
                    if (state[blank1+left] in corner) and (state[blank2+left] == state[blank1+left]):
                        if "right_corner" not in previous_actions: action.append("left_corner")
                    if ((state[blank1+left] in rectangle) and (state[blank2+left] == state[blank1+left])):
                        if "right_vrectangle" not in previous_actions: action.append("left_vrectangle")

                    #pas sûr pour les deux derniers cas
        if line1 < limit_line_down:
            if (state[blank1+down] == state[blank2]):
                if column1 < limit_column_right:
                    if state[blank1+right] == bsquare and state[blank2+right] == bsquare:
                        if "left_bsquare" not in previous_actions: action.append("right_bsquare")
                    if (state[blank1+right] in corner) and (state[blank2+right] == state[blank1+right]):
                        if "left_corner" not in previous_actions: action.append("right_corner")
                    if ((state[blank1+right] in rectangle) and (state[blank2+right] == state[blank1+right])):
                        if "left_vrectangle" not in previous_actions: action.append("right_vrectangle")

                if column1 > limit_column_left:
                    if state[blank1+left] == bsquare and state[blank2+left] == bsquare:
                        if "right_bsquare" not in previous_actions: action.append("left_bsquare")
                    if (state[blank1+left] in corner) and (state[blank2+left] == state[blank1+left]):
                        if "right_corner" not in previous_actions: action.append("left_corner")
                    if ((state[blank1+left] in rectangle) and (state[blank2+left] == state[blank1+left])):
                        if "right_vrectangle" not in previous_actions: action.append("left_vrectangle")

        ''' STEP 3 : blank1 and blank2 are diagonally side by side'''

        if column1 > limit_column_left and line1 < limit_line_down:
            if (state[blank1+down+left] == state[blank2]):
                if (line1 > limit_line_up):
                    if((state[blank1+up] in corner) and (state[blank2+up] == state[blank1+up])):
                        if "down_corner_diag" not in previous_actions: action.append("up_corner_diag")

                if (line2 < limit_line_down):
                    if((state[blank1+down] in corner) and (state[blank2+down] == state[blank1+down])):
                        if "up_corner_diag" not in previous_actions: action.append("down_corner_diag")

                if (column1 < limit_column_right):
                    if((state[blank1+right] in corner) and (state[blank2+right] == state[blank1+right])):
                        if "left_corner_diag" not in previous_actions: action.append("right_corner_diag")

                if (column2 > limit_column_left):
                    if((state[blank1+left] in corner) and (state[blank2+left] == state[blank1+left])):
                        if "right_corner_diag" not in previous_actions: action.append("left_corner_diag")

        if column1 < limit_column_right and line1 < limit_line_down:
            if (state[blank1+down+right] == state[blank2]):
                if (line1 > limit_line_up):
                    if((state[blank1+up] in corner) and (state[blank2+up] == state[blank1+up])):
                        if "down_corner_diag" not in previous_actions: action.append("up_corner_diag")

                if (line2 < limit_line_down):
                    if((state[blank1+down] in corner) and (state[blank2+down] == state[blank1+down])):
                        if "up_corner_diag" not in previous_actions: action.append("down_corner_diag")

                if (column2 < limit_column_right):
                    if((state[blank1+right] in corner) and (state[blank2+right] == state[blank1+right])):
                        if "left_corner_diag" not in previous_actions: action.append("right_corner_diag")

                if (column1 > limit_column_left):
                    if((state[blank1+left] in corner) and (state[blank2+left] == state[blank1+left])):
                        if "right_corner_diag" not in previous_actions: action.append("left_corner_diag")

        ##Same case with blank1 and blank2 reversed
        if column2 > limit_column_left and line2 < limit_line_down:
            if (state[blank2+down+left] == state[blank1]):
                if (line2 > limit_line_up):
                    if((state[blank1+up] in corner) and (state[blank2+up] == state[blank1+up])):
                        if "down_corner" not in previous_actions: action.append("up_corner_diag")

                if (line1 < limit_line_down):
                    if((state[blank1+down] in corner) and (state[blank2+down] == state[blank1+down])):
                        if "up_corner" not in previous_actions: action.append("down_corner_diag")

                if (column2 < limit_column_right):
                    if((state[blank1+right] in corner) and (state[blank2+right] == state[blank1+right])):
                        if "left_corner" not in previous_actions: action.append("right_corner_diag")

                if (column1 > limit_column_left):
                    if((state[blank1+left] in corner) and (state[blank2+left] == state[blank1+left])):
                        if "right_corner" not in previous_actions: action.append("left_corner_diag")

        if column2 < limit_column_right and line2 < limit_line_down:
            if (state[blank2+down+right] == state[blank1]):
                if (line2 > limit_line_up):
                    if((state[blank1+up] in corner) and (state[blank2+up] == state[blank1+up])):
                        if "down_corner" not in previous_actions: action.append("up_corner_diag")

                if (line1 < limit_line_down):
                    if((state[blank1+down] in corner) and (state[blank2+down] == state[blank1+down])):
                        if "up_corner" not in previous_actions: action.append("down_corner_diag")

                if (column1 < limit_column_right):
                    if((state[blank1+right] in corner) and (state[blank2+right] == state[blank1+right])):
                        if "left_corner" not in previous_actions: action.append("right_corner_diag")

                if (column2 > limit_column_left):
                    if((state[blank1+left] in corner) and (state[blank2+left] == state[blank1+left])):
                        if "right_corner" not in previous_actions: action.append("left_corner_diag")

######################################################################
        #print("Authorized actions:", action)
        #displayPuzzle(state)
        #print("\n\n")
        previous_actions = action

        return action

    def result(self, state, action):
        blank1 = state.index(-1)
        blank2= state.index(-2)
        line_1, column_1 = divmod(blank1,6)
        line_2, column_2 = divmod(blank2,6)
        newState=list(state)

        ''' One-blank situations '''

        #Actions for a small square
        if action == "up_square1":
            newState[blank1] = newState[blank1+up]
            newState[blank1+up] = -1

        if action == "up_square2":
            newState[blank2] = newState[blank2+up]
            newState[blank2+up] = -2

        if action == "down_square1":
            newState[blank1] = newState[blank1+down]
            newState[blank1+down] = -1

        if action == "down_square2":
            newState[blank2] = newState[blank2+down]
            newState[blank2+down] = -2

        if action == "left_square1":
            newState[blank1] = newState[blank1+left]
            newState[blank1+left] = -1

        if action == "left_square2":
            newState[blank2] = newState[blank2+left]
            newState[blank2+left] = -2

        if action == "right_square1":
            newState[blank1] = newState[blank1+right]
            newState[blank1+right] = -1

        if action == "right_square2":
            newState[blank2] = newState[blank2+right]
            newState[blank2+right] = -2


        #Actions for a rectangle
        if action == "up_vrectangle1":
            newState[blank1] = newState[blank1+up]
            newState[blank1+2*up] = -1

        if action == "up_vrectangle2":
            newState[blank2] = newState[blank2+up]
            newState[blank2+2*up] = -2

        if action == "down_vrectangle1":
            newState[blank1] = newState[blank1+down]
            newState[blank1+2*down] = -1

        if action == "down_vrectangle2":
            newState[blank2] = newState[blank2+down]
            newState[blank2+2*down] = -2

        if action == "left_hrectangle1":
            newState[blank1] = newState[blank1+left]
            newState[blank1+2*left] = -1

        if action == "left_hrectangle2":
            newState[blank2] = newState[blank2+left]
            newState[blank2+2*left] = -2

        if action == "right_hrectangle1":
            newState[blank1] = newState[blank1+right]
            newState[blank1+2*right] = -1

        if action == "right_hrectangle2":
            newState[blank2] = newState[blank2+right]
            newState[blank2+2*right] = -2


        ''' Two-blanks situation '''

        #Actions for a big square
        if action == "up_bsquare":
            newState[blank1] = 1
            newState[blank1+2*up] = -1

            newState[blank2] = 1
            newState[blank2+2*up] = -2

        if action == "down_bsquare":
            newState[blank1] = 1
            newState[blank1+2*down] = -1

            newState[blank2] = 1
            newState[blank2+2*down] = -2

        if action == "right_bsquare":
            newState[blank1] = 1
            newState[blank1+2*right] = -1

            newState[blank2] = 1
            newState[blank2+2*right] = -2

        if action == "left_bsquare":
            newState[blank1] = 1
            newState[blank1+2*left] = -1

            newState[blank2] = 1
            newState[blank2+2*left] = -2

        #Actions for corners
            #When blanks are side by side
        if action == "up_corner":
            newState[blank1] = newState[blank1+up]
            newState[blank2] = newState[blank2+up]

            if (newState[blank1+2*up] == newState[blank1+up]):
                newState[blank1+2*up] = -1 
                newState[blank2+up] = -2 
            else:
                newState[blank2+2*up] = -2 
                newState[blank1+up] = -1 

        if action == "down_corner":
            newState[blank1] = newState[blank1+down]
            newState[blank2] = newState[blank2+down]
            
            if (newState[blank1+2*down] == newState[blank1+down]):
                    newState[blank1+2*down] = -1 
                    newState[blank2+down] = -2 
            else:
                newState[blank2+2*down] = -2 
                newState[blank1+down] = -1 

        #Actions for vertical corners
        if action == "left_corner":
            newState[blank1] = newState[blank1+left]
            newState[blank2] = newState[blank2+left]

            if (newState[blank1+2*left] == newState[blank1+left]):
                newState[blank1+2*left] = -1 
                newState[blank2+left] = -2 
            else:
                newState[blank2+2*left] = -2 
                newState[blank1+left] = -1 

        if action == "right_corner":
            newState[blank1] = newState[blank1+right]
            newState[blank2] = newState[blank2+right]

            if (newState[blank1+2*right] == newState[blank1+right]):
                newState[blank1+2*right] = -1 
                newState[blank2+right] = -2 
            else:
                newState[blank2+2*right] = -2 
                newState[blank1+right] = -1 
            
        #When blanks are diagonally side by side
        
        if action == "down_corner_diag":
            newState[blank1] = newState[blank1+down]
            newState[blank2] = newState[blank2+down]
            
            if (line_1 < limit_line_down-1):
                if (newState[blank1+2*down] == newState[blank1+down]):
                    newState[blank1+2*down] = -1
                    newState[blank2+down] = -2
                else:
                    newState[blank1+down]= -1
                    newState[blank2+2*down] = -2
            else:
                newState[blank2+2*down] = -2 
                newState[blank1+down] = -1 
                     
        if action == "up_corner_diag":
            newState[blank1] = newState[blank1+up]
            newState[blank2] = newState[blank2+up]
            
            if (line_1 > limit_line_up+1):
                if (newState[blank1+2*up] == newState[blank1+up]):
                    newState[blank1+2*up] = -1
                    newState[blank2+up] = -2
                else:
                    newState[blank1+up]= -1
                    newState[blank2+2*up] = -2
            else:
                newState[blank2+2*up] = -2 
                newState[blank1+up] = -1 
                
        if action == "left_corner_diag":
            newState[blank1] = newState[blank1+left]
            newState[blank2] = newState[blank2+left]
            
            if (column_1 > limit_column_left+1):
                if (newState[blank1+2*left] == newState[blank1+left]):
                    newState[blank1+2*left] = -1
                    newState[blank2+left] = -2
                else:
                    newState[blank1+left]= -1
                    newState[blank2+2*left] = -2
            else:
                newState[blank2+2*left] = -2 
                newState[blank1+left] = -1 
                     
        if action == "right_corner_diag":
            newState[blank1] = newState[blank1+right]
            newState[blank2] = newState[blank2+right]
            
            if (column_1 < limit_column_right-1):
                if (newState[blank1+2*right] == newState[blank1+right]):
                    newState[blank1+2*right] = -1
                    newState[blank2+right] = -2
                else:
                    newState[blank1+right]= -1
                    newState[blank2+2*right] = -2
            else: 
                newState[blank2+2*right] = -2 
                newState[blank1+right] = -1 

    #Actions for double rectangles
        #Actions for horizontal rectangles
        
        if action == "up_hrectangle":
            newState[blank1] = newState[blank1+up]
            newState[blank1+up] = -1

            newState[blank2] = newState[blank2+up]
            newState[blank2+up] = -2

        if action == "down_hrectangle":
            newState[blank1] = newState[blank1+down]
            newState[blank1+down] = -1

            newState[blank2] = newState[blank2+down]
            newState[blank2+down] = -2

        if action == "left_vrectangle":
            newState[blank1] = newState[blank1+left]
            newState[blank1+left] = -1

            newState[blank2] = newState[blank2+left]
            newState[blank2+left] = -2

        if action == "right_vrectangle":
            newState[blank1] = newState[blank1+right]
            newState[blank1+right] = -1

            newState[blank2] = newState[blank2+right]
            newState[blank2+right] = -2

        #print("Result:\n")
        #displayPuzzle(newState)

        return tuple(newState)

    def h1(self,node) :
        lin,col = divmod(node.state.index(1),6)
        #print("return ",9-lin-col)
        #print("line : ",lin)
        #print("column : ",col)
        return 9-lin-col


    def h2(self,node):
        modul = 0
        index1 = node.state.index(-1)
        index2 = node.state.index(-2)
        lin,col = divmod(node.state.index(1),6)
        lin1,col1 = divmod(node.state.index(-1),6)
        lin2,col2 = divmod(node.state.index(-2),6)
        h1 = 9-lin-col
        #displayPuzzle(node.state)
        if col1 == limit_column_left:
            modul = modul + 1
        elif node.state[index1+left] != bsquare:
                modul = modul + 1
        if col1 == limit_column_right:
            modul = modul + 1
        elif node.state[index1+right] != bsquare:
                modul = modul + 1
        if col2 == limit_column_left:
            modul = modul + 1
        elif node.state[index2+left] != bsquare:
                modul = modul + 1
        if col2 == limit_column_right:
            modul = modul + 1
        elif node.state[index2+right] != bsquare:
                modul = modul + 1
        if lin1 == limit_line_up:
            modul = modul + 1
        elif node.state[index1+up] != bsquare:
                modul = modul + 1
        if lin1 == limit_line_down:
            modul = modul + 1
        elif node.state[index1+down] != bsquare:
                modul = modul + 1
        if lin2 == limit_line_up:
            modul = modul + 1
        elif node.state[index2+up] != bsquare:
                modul = modul + 1
        if lin2 == limit_line_down:
            modul = modul + 1
        elif node.state[index2+down] != bsquare:
                modul = modul + 1
        modul = modul - 6
#        print("h1 : ", self.h1(node))
#        print("module : ", modul)
        return 4*h1 + modul

    
def astar_with_h1 (puzzle):
        search.astar_search(puzzle, puzzle.h1)

def astar_with_h2 (puzzle):
        search.astar_search(puzzle, puzzle.h2)

#displayPuzzle(state_hard) 
    
puzzle0 = HW1Puzzle(state0)
puzzle1 = HW1Puzzle(state1)
puzzle2 = HW1Puzzle(state2)
puzzle3 = HW1Puzzle(state3)
puzzle = HW1Puzzle(state_task)

#search.compare_searchers([puzzle],"HW2Puzzle", searchers=[search.breadth_first_search,
#                                                          search.depth_limited_search,
#                                                          astar_with_h1,
#                                                          astar_with_h2,
#                                                          search.uniform_cost_search,
#                                                          search.iterative_deepening_search])

#search.compare_searchers([puzzle],"HW2Puzzle", searchers=[astar_with_h2])    
    
#print("\nPuzzle 0:")
#search.compare_searchers([puzzle0],"HW2Puzzle", 
#                         searchers=[search.breadth_first_search,
#                                    astar_with_h1,astar_with_h2])    
#
#print("\nPuzzle 1:")
#search.compare_searchers([puzzle1],"HW2Puzzle", 
#                         searchers=[search.breadth_first_search,
#                                    astar_with_h1,astar_with_h2])    
#    
#print("\nPuzzle 2:")
#search.compare_searchers([puzzle2],"HW2Puzzle", 
#                         searchers=[search.breadth_first_search,
#                                    astar_with_h1,astar_with_h2])    
#
#print("\nPuzzle:")
#search.compare_searchers([puzzle3],"HW2Puzzle", 
#                         searchers=[search.breadth_first_search,
#                                    astar_with_h1,astar_with_h2])    
 
#search.depth_limited_search(puzzle)
#print("\n\n")

#print("astar1")
#astar_with_h1(puzzle)
print("astar2")
astar_with_h2(puzzle)
#print("breadth")
#search.breadth_first_search(puzzle)
