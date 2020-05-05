#from __future__ import print_function

import stanford_simplified
#import question_analysis
import answer_analysis

#This function modify the "sentence" list in order to  
        
#user_input = input("Some input please: ")
#print(user_input)
            
#---------------------------

list_traited_text_answers = stanford_simplified.run1()

#-------------------------- DATA COLLECTION ---------------------------

answer_analysis.run(list_traited_text_answers)

#----------------------------- QUESTION ------------------------------------

#question = stanford_simplified.run2()
#answer = question_analysis.run(question, triplet_data)

#------------------------------ ANSWER ---------------------------------
#print("The answer is: ",answer)


#Triplets appending
