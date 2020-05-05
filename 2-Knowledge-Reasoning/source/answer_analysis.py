# -*- coding: utf-8 -*-

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

#--------------------------------------------------------------------------

VP = {'VBD','VBG','VBN','VBP','VBZ'}
NN = {'NNS','NNP','NNPS','NN'}
PP = {'TO','FOR','IN'}

OTHERS = {'CD','JJ','PRP','CC','RB'}

ALLOWEDTYPES = NN | VP | PP | OTHERS 
ROLES = {'S','VP','PP','ADJP','determinant'}

EXCEPTIONS = {'of'} #to manage the case "is north of"

nxt = 1
prv = -1

location = {"goes", "go", "went", "was"}
verb = {"VBZ"}
VerbDefinition = {"is"}
NounDirection = {"east", "west", "north", "south", "situated", "placed"}

#----------------------------------------------------------------------------

#textA = ['(ROOT', '(S', '(NP', '(NNP', 'Barack)', '(NNP', 'Obama))', '(VP', '(VBZ', 'eats)', '(NP', '(NP', '(DT', 'the)', '(NN', 'dog))', '(CC', 'and)', '(NP', '(DT', 'the)', '(NN', 'cat))))', '(.', '.)))']
#textB = ['(ROOT', '(S', '(NP', '(NNP', 'Robert))', '(VP', '(VBD', 'bought)', '(NP', '(NP', '(NN', 'cheese))', '(CC', 'and)', '(NP', '(NNP', 'Joe)', '(NNS', 'drinks)', '(NN', 'water))))', '(.', '.)))']
#
#text_1 = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(ADVP', '(RB', 'north)', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Paris)))))', '(.', '.)))']
#text_2 = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(ADVP', '(RB', 'east)', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Paris)))))', '(.', '.)))']
# 
text21 = [['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(ADVP', '(RB', 'north)', '(PP', '(IN', 'of)', '(NP', '(NNP', 'France)))))', '(.', '.)))'], ['(ROOT', '(S', '(NP', '(PRP', 'It))', '(VP', '(VBZ', 'likes)', '(NP', '(NN', 'fruit)))', '(.', '.)))']]
#
#text1 = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(NP', '(NP', '(NN', 'capital))', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Estonia)))))', '(.', '.)))'], ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(PP', '(IN', 'in)', '(NP', '(NNP', 'Estonia))))', '(.', '.)))']
#text = text1

#----------------------------------------------------------------------------

def cleanText(txt):
    for index, item in enumerate(txt):
        for ch in ['(', ')']:
            if (ch in txt[index]):
                txt[index] = txt[index].replace(ch,"")
                
def printSentence(Sentence):
    for index, item in enumerate(Sentence):
        Sentence[index].display()
        print("")
        
def makeTriplet(sentence):
    
    listOfTriplets = [triplet("aa", "aa", "aa")]
    tripletIndex = 0 
    
    for item in sentence:
        if item.role == "S":
            if(item.type == 'PRP' and tripletIndex != 0):
                listOfTriplets[tripletIndex].subject = listOfTriplets[tripletIndex-1].subject
            else:
                listOfTriplets[tripletIndex].subject = item.word
            
        elif (item.type == "CC"):
            tripletIndex = tripletIndex+1
            listOfTriplets.append(triplet("bb","bb","bb"))
        
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

        elif (item.role == "PP") & (item.type == "NNP"):
            listOfTriplets[tripletIndex].complement = item.word
            
        elif ((item.role == "NP") & (item.type != "DT")):
            listOfTriplets[tripletIndex].complement = item.word
            if( (tripletIndex>0) & (listOfTriplets[tripletIndex].subject=="bb")):
                listOfTriplets[tripletIndex].subject = listOfTriplets[tripletIndex-1].subject
                listOfTriplets[tripletIndex].linker = listOfTriplets[tripletIndex-1].linker
        
        elif (item.role == "ADJP") & (item.type == "CD"):
            listOfTriplets[tripletIndex].complement = item.word
            
        
    index=0
    for item in listOfTriplets:
         print(index)
         print (item.subject, item.linker, item.complement)   
         index = index+1
#         print(item.subject, "is :", getWikiInfobox(item.subject))
#         print(item.complement, "is :", getWikiInfobox(item.complement))
         
    #print (listOfTriplets[0].subject, listOfTriplets[0].linker, listOfTriplets[0].complement)

    return listOfTriplets


def makeTriplet_wikipedia(sentence):
    
    listOfTriplets = [triplet("aa", "aa", "aa")]
    tripletIndex = 0 
    
    for item in sentence:
        if item.role == "S":
            if(item.type == 'PRP' and tripletIndex != 0):
                listOfTriplets[tripletIndex].subject = listOfTriplets[tripletIndex-1].subject
            else:
                listOfTriplets[tripletIndex].subject = 'https://en.wikipedia.org/wiki/'+ item.word
            
        elif (item.type == "CC"):
            tripletIndex = tripletIndex+1
            listOfTriplets.append(triplet("bb","bb","bb"))
        
        elif (item.type == "DIRECTION"):
            listOfTriplets[tripletIndex].linker = item.word
            
        elif (item.type == "CAPITAL"):#laaa
            listOfTriplets[tripletIndex].linker = item.word
            
        elif (item.type == "inside"):
            listOfTriplets[tripletIndex].linker = 'rdf:in'#laaa
            
        elif item.word in location: 
            listOfTriplets[tripletIndex].linker = "rdf:in"#laaa

        elif (item.type in verb) & (item.word not in location):
            listOfTriplets[tripletIndex].linker = item.word

        elif (item.role == "PP") & (item.type == "NNP"):
            listOfTriplets[tripletIndex].complement = item.word
            
        elif ((item.role == "NP") & (item.type != "DT")):
            listOfTriplets[tripletIndex].complement = item.word
            if( (tripletIndex>0) & (listOfTriplets[tripletIndex].subject=="bb")):
                listOfTriplets[tripletIndex].subject = listOfTriplets[tripletIndex-1].subject
                listOfTriplets[tripletIndex].linker = listOfTriplets[tripletIndex-1].linker
        
        elif (item.role == "ADJP") & (item.type == "CD"):
            listOfTriplets[tripletIndex].complement = item.word
    
    index=0
    for item in listOfTriplets:
         print(index)
         print (item.subject, item.linker, item.complement)   
         index = index+1

def makeTriplet_logic(sentence, file):
    
    listOfTriplets = [triplet("aa", "aa", "aa")]
    tripletIndex = 0 
    
    for item in sentence:
        if item.role == "S":
            if(item.type == 'PRP' and tripletIndex != 0):
                listOfTriplets[tripletIndex].subject = listOfTriplets[tripletIndex-1].subject
            else:
                listOfTriplets[tripletIndex].subject = 'ex:' + item.word
            
        elif (item.type == "CC"):
            tripletIndex = tripletIndex+1
            listOfTriplets.append(triplet("bb","bb","bb"))
        
        elif (item.type == "DIRECTION"):
            listOfTriplets[tripletIndex].linker = 'id:' + item.word
            
        elif (item.type == "CAPITAL"):#laaa
            listOfTriplets[tripletIndex].linker = 'id:' + item.word
            
        elif (item.type == "inside"):
            listOfTriplets[tripletIndex].linker = 'id:in'#laaa
            
        elif item.word in location: 
            listOfTriplets[tripletIndex].linker = "id:in"#laaa

        elif (item.type in verb) & (item.word not in location):
            listOfTriplets[tripletIndex].linker = item.word

        elif (item.role == "PP") & (item.type == "NNP"):
            listOfTriplets[tripletIndex].complement = 'ex:' + item.word
            
        elif ((item.role == "NP") & (item.type != "DT")):
            listOfTriplets[tripletIndex].complement = item.word
            if( (tripletIndex>0) & (listOfTriplets[tripletIndex].subject=="bb")):
                listOfTriplets[tripletIndex].subject = listOfTriplets[tripletIndex-1].subject
                listOfTriplets[tripletIndex].linker = listOfTriplets[tripletIndex-1].linker
        
        elif (item.role == "ADJP") & (item.type == "CD"):
            listOfTriplets[tripletIndex].complement = item.word
    
    index=0
    for item in listOfTriplets:
         print(index)
         print (item.subject, item.linker, item.complement)
         chaine = '\nrdf("' + item.subject + '", "' + item.linker + '", "' + item.complement + '").\n'
         file.write(chaine)
         index = index+1
         
    
    
                        

def preparationToTriplet(sent):
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
                if(sent[index].word in VerbDefinition and sent[index + nxt].type == 'IN'):
                    struct.word = sent[index].word + sent[index + nxt].word
                    struct.type = 'inside'
                    del sent[index + nxt]
                
        
        index = index + 1
    return sent

def stanfordToSentence(text):
    
    index = 0
    sentence = []

    CurrentRole = ''            
    for lettre in text:
        #Changement for each new met role
        if(lettre in ROLES):
            CurrentRole = lettre
        #Implementation in the sentance list if the word has reason to be in    
        if(lettre in ALLOWEDTYPES and lettre not in EXCEPTIONS):
            sentence.append(word(text[index + nxt], lettre, CurrentRole))
        
        index = index + 1
    return sentence

def writeToFile(ListofSentences):
    fichier = open("otter_alpha.in", "w")
    fichier.write("% clear automatic strategy selection\nclear(auto).\n% use capital letters (A,X,T,...) as vars")
    fichier.write("set(prolog_style_variables).\n% select the search strategy\nset(hyper_res). % an alternative is to use set(binary_res).")
    fichier.write("set(factor).\n% select sensible amount of output\nclear(print_given). % uncomment to see input and process")
    fichier.write("set(print_kept).  % this is important: prints all generated and kept clauses\nassign(stats_level, 0).\n% just make it stop after N secs")
    fichier.write("assign(max_seconds, 10).\nlist(sos).\n% -----------------------------------------------------------------------------------------------------\n")
    
    makeTriplet_logic(ListofSentences[0], fichier)
    makeTriplet_logic(ListofSentences[1], fichier)
    
    #A MODIFIER EN FONCTION DU NOMBRE DE TRIPLETS
    
    fichier.write('\n% -----------------------------------------------------------------------------------------------------\n\n')

    fichier.write('% -------- ORIENTATION ----------\n    	% ------- TRANSITIVITY ------\n\n    	  -rdf(X,"id:westof", Y) |\n')
    fichier.write('    	  -rdf(Y,"id:westof", Z) |\n    	  rdf(X,"id:westof",Z).\n\n    	  -rdf(X,"id:eastof", Y) |\n')
    fichier.write('          -rdf(Y,"id:eastof", Z) |\n    	  rdf(X,"id:eastof",Z).\n\n    	  -rdf(X,"id:northof", Y) |\n    	  -rdf(Y,"id:northof", Z) |\n')
    fichier.write('	  rdf(X,"id:northof",Z).\n\n    	  -rdf(X,"id:southof", Y) |\n    	  -rdf(Y,"id:southof", Z) |\n    	  rdf(X,"id:southof",Z).\n\n')
    fichier.write('    % -------- LOCATION -------\n\n    	% ----- CAPITAL ------\n\n    		  -rdf(X,"id:capital", Y) |\n    		  rdf(X,"http://dbpedia.org/ontology/type", "http://conceptnet5.media.mit.edu/web/c/en/city").\n')
    fichier.write('\n    		  -rdf(X,"id:capital", Y) |\n    		  rdf(X,"http://dbpedia.org/ontology/IN", Y).\n\n    		  -rdf(X,"id:capital", Y) |\n    		  rdf(Y,"http://dbpedia.org/ontology/type", "http://conceptnet5.media.mit.edu/web/c/en/country").')
    fichier.write('\n\n    	% ----- EUROPE ------\n\n    		  -rdf(X,"http://dbpedia.org/ontology/type", "http://conceptnet5.media.mit.edu/web/c/en/city") |\n    		  -rdf(X,"http://dbpedia.org/ontology/IN", Y) |\n')
    fichier.write('                 -rdf(Y, "http://dbpedia.org/ontology/type", "http://conceptnet5.media.mit.edu/web/c/en/country") |\n    		  -rdf(Y, "http://dbpedia.org/ontology/IN", "ex:Europe").\n\n    end_of_list.')
    
    fichier.close()
    
def run(text):
    for item in text:    
        cleanText(item)
        print("\ncleanText :\n", text)    
        
    mySentences = []
    for item in text:
        mySentences.append(stanfordToSentence(item))
    print("\nstanfordToSentence:\n")
    for index, item in enumerate(mySentences):
        #print("Sentence", index)        
        printSentence(mySentences[index])
        
    print("\npreparationToTriple\n")
    for index, item in enumerate(mySentences):
        mySentences[index] = preparationToTriplet(mySentences[index])    
    for index, item in enumerate(mySentences):
        #print("Sentence", index)
        printSentence(mySentences[index])
    writeToFile(mySentences)