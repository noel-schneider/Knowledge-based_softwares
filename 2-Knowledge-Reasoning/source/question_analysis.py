class word():  
    def __init__(self, word, typ, role):
        self.word = word
        self.type = typ
        self.role = role
    
    def display(self):
        print("word: ",self.word)
        print("type: ",self.type)
        print("role: ",self.role)
        
class triplet():
    def __init__(self, subject, linker, complement):
        self.subject = subject
        self.linker = linker
        self.complement = complement

    def display(self):
        print(self.subject, self.linker, self.complement)


#-------------------------------------------------------------

VP = {'VBD','VBG','VBN','VBP','VBZ'}
NN = {'NNS','NNP','NNPS','NN'}
PP = {'TO','FOR','IN'}
JJ = {'JJ','JJS','JJR'}
WW = {'WP','WRB','WDT'}

OTHERS = {'CD','JJ','PRP','CC','RB'}

ALLOWEDTYPES = NN | VP | PP | OTHERS 
ALLOWEDTYPES_QUESTIONS = ALLOWEDTYPES | WW

EXCEPTIONS = {'of'}
ROLES_QUESTIONS = {'SBARQ','SQ','FRAG'}

location = {"goes", "go", "went", "was"}
verb = {"VBZ"}
VerbDefinition = {"is"}
NounDirection = {"east", "west", "north", "south", "situated", "placed"}

nxt = 1
prv = -1

question1 = ['ROOT', 'SBARQ', 'WHNP', 'WP', 'What', 'SQ', 'VBZ', 'is', 'NP', 'NP', 'DT', 'the', 'NN', 'capital', 'PP', 'IN', 'of', 'NP', 'NNP', 'Estonia', '.', '?']
question2 = ['ROOT', 'SBARQ', 'WHADVP', 'WRB', 'Where', 'SQ', 'VBZ', 'is', 'NP', 'NNP', 'Tallinn', '.', '?']
question3 = [['(ROOT', '(SBARQ', '(WHNP', '(WDT', 'Which)', '(NN', 'city))', '(SQ', '(VBZ', 'is)', '(NP', '(NP', '(NN', 'west))', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Tallinn)))))', '(.', '?)))']]

filename = 'input.txt'
with open(filename) as f:
    data = f.readlines()
    
text_answers = data[0]

#--------------------------------------------------------------------
def getInfobox(sujet):
    return 'city'

def cleanText(txt):
    for index, item in enumerate(txt):
        for ch in ['(', ')']:
            if (ch in txt[index]):
                txt[index] = txt[index].replace(ch,"")
        
def printSentence(Sentence):
    for index, item in enumerate(Sentence):
        Sentence[index].display()
        print("")

def stanfordToQuestion(text):
    index = 0
    sentence = []
    
    CurrentRole = ''     
    for lettre in text:
        #Changement for each new met role
        if(lettre in ROLES_QUESTIONS):
            CurrentRole = lettre
        #Implementation in the sentance list if the word has reason to be in    
        if(lettre in ALLOWEDTYPES_QUESTIONS and lettre not in EXCEPTIONS):
            sentence.append(word(text[index + nxt], lettre, CurrentRole))
        
        index = index + 1
    #print("Sentence:", sentence)
    return sentence


def preparationToTripletQuestion(sent):
    index = 0
          
    for struct in sent:
        #If two words have the same type and role, we join them
        if(index != len(sent)-1):
            if(struct.type == sent[index + nxt].type and struct.role == sent[index + nxt].role):
                struct.word = struct.word + sent[index + nxt].word
                del sent[index + nxt]
            # management of "is north of ..."
            if(index != 0):
                if(struct.word in NounDirection and sent[index + prv].word in VerbDefinition and sent[index + nxt].type == 'IN'):
                    struct.word = sent[index + prv].word + struct.word + sent[index + nxt].word
                    struct.type = 'DIRECTION'
                    del sent[index + nxt]
                    del sent[index + prv]
                if(struct.word == 'capital' and sent[index + prv].word in VerbDefinition and sent[index + nxt].type == 'IN'):
                    struct.type = 'CAPITAL'#laaa
                    struct.word = 'capital' #attention: modifié rdf:capital
                    del sent[index + nxt]
                    del sent[index + prv]
                if(sent[index].type in WW and sent[index + nxt].word in VerbDefinition ):
                    struct.word = sent[index].word
                    struct.role = 'QUESTION'
                    del sent[index + nxt]
                
        
        index = index + 1
    return sent

def makeTripletQuestion(sentence):
    
    listOfTriplets = [triplet("aa", "aa", "aa")]
    tripletIndex = 0 
    if len(sentence) == 3:
        listOfTriplets[tripletIndex].subject = sentence[0].word
        listOfTriplets[tripletIndex].linker = sentence[1].word
        listOfTriplets[tripletIndex].complement = sentence[2].word
    else:
        for item in sentence:
            if item.role == "SBARQ":
                listOfTriplets[tripletIndex].subject = item.word
                
            elif (item.role == "FRAG"):
                listOfTriplets[tripletIndex].subject = item.word
            
            elif (item.type == "DIRECTION"):
                listOfTriplets[tripletIndex].linker = item.word
                
            elif (item.type == "CAPITAL"):#laaa
                listOfTriplets[tripletIndex].linker = item.word
                
            elif (item.type == "inside"):
                listOfTriplets[tripletIndex].linker = 'in'#laaa
                
            elif item.word in location: 
                listOfTriplets[tripletIndex].linker = "in"#laaa
    
            elif (item.type in verb) & (item.word not in location):
                listOfTriplets[tripletIndex].linker = item.word
    
            elif (item.role == "SQ") & (item.type == "NNP"):
                listOfTriplets[tripletIndex].complement = item.word
            
            elif (item.role == "ADJP") & (item.type == "CD"):
                listOfTriplets[tripletIndex].complement = item.word
            
# DISPLAYING
    index=0
    for item in listOfTriplets:
         print(index)
         print (item.subject, item.linker, item.complement)   
         index = index+1
#         print(item.subject, "is :", getWikiInfobox(item.subject))
#         print(item.complement, "is :", getWikiInfobox(item.complement))
         
    #print (listOfTriplets[0].subject, listOfTriplets[0].linker, listOfTriplets[0].complement)

    return listOfTriplets

def getAnswerDataBase(questionTriplet, answerTriplet):
    
    for tripl in answerTriplet:
        for t in tripl:
            if(questionTriplet.subject == 'What'):
                if(questionTriplet.linker == t.linker and questionTriplet.complement == t.complement):
                    return t.subject
            elif(questionTriplet.subject == 'Where'):
                if(questionTriplet.complement == t.subject and t.linker == 'in'):
                    return t.complement
            elif(questionTriplet.subject == 'What'):
                if(questionTriplet.linker == t.linker and questionTriplet.complement == t.complement):
                    return t.subject
            #elif(questionTriplet.subject == 'city'):
            else: #gère les cas "Which city" et "Which country"
                if(getInfobox(t.subject) == questionTriplet.subject):
                    if(questionTriplet.linker == t.linker and questionTriplet.complement == t.complement):
                        return t.subject
                
    return 'no answer'

def getAnswerInternet(questionTriplet):
    
    if(questionTriplet.subject == 'What' and questionTriplet.linker == 'is'):
        return getInfobox(questionTriplet.complement)
    return 'no answer'
    
#function turning questions into triplets
def questionAnalyse(question_list, triplet_list):
    print('Question:',question_list)
    transformed_question = stanfordToQuestion(question_list)  
    printSentence(transformed_question)
    preparationToTripletQuestion(transformed_question)
    print('preparation to Triplet Question:')
    printSentence(transformed_question)
    tripletQuestion = makeTripletQuestion(transformed_question)
    return tripletQuestion

def run(question_raw, tripletList):
    for item in question_raw:    
       cleanText(item)
    print("\ncleanText :\n", question_raw)
    Tquestion = questionAnalyse(question_raw[0], tripletList)
    #print("tripletList", tripletList)
    #print("tripletList[0]", tripletList[0])
    #print("tripletList[0].linker", tripletList[0].linker)
    answer = getAnswerDataBase(Tquestion[0], tripletList)
    if answer == 'no answer':
        answer = getAnswerInternet(Tquestion[0])
        

    return answer
