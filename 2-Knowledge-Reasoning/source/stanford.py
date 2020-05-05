# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:27:47 2017

@author: noel
"""

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

punctuation = {".", "!", "?"}

question1 = ['ROOT', 'SBARQ', 'WHNP', 'WP', 'What', 'SQ', 'VBZ', 'is', 'NP', 'NP', 'DT', 'the', 'NN', 'capital', 'PP', 'IN', 'of', 'NP', 'NNP', 'Estonia', '.', '?']
question2 = ['ROOT', 'SBARQ', 'WHADVP', 'WRB', 'Where', 'SQ', 'VBZ', 'is', 'NP', 'NNP', 'Tallinn', '.', '?']
question3 = ['ROOT', 'FRAG', 'WHNP', 'WDT', 'Which', 'NN', 'city', 'PP', 'NP', 'NN', 'ist', 'NN', 'west', 'PP', 'IN', 'of', 'NP', 'NNP', 'Tallinn', '.', '?']

question1_raw = "What is the capital of Estonia?"
question2 = "Where is Tallinn?"
question3 = "Which city is west of Tallinn?"

text = "Tallinn is capital of Estonia. Tallinn is in Estonia."
#text = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(NP', '(NP', '(NN', 'capital))', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Estonia)))))', '(.', '.)))'], ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(PP', '(IN', 'in)', '(NP', '(NNP', 'Estonia))))', '(.', '.)))']


def numberOfSentences(inputText):
    count = 0;
    for lettre in inputText:
        if (lettre in punctuation):
            count = count+1
    return count

def makeStanfordList(inputText):
    outputList = []
    index = 0
    output = nlp.annotate(inputText, properties={
                        'annotators': 'tokenize,ssplit,pos,depparse,parse',
                        'outputFormat': 'json'})
    while(index < numberOfSentences(inputText)): #nSentences is globally defined as the number of sentences in the text
        outputList.append(' '.join(output['sentences'][index]['parse'].split()).split(' '))
        #### Prints result
        print(output['sentences'][index]['parse'])
        ####
        index = index+1
    return outputList

filename = 'input.txt'
with open(filename) as f:
    data = f.readlines()

def run1():
    print("makeStanfordList text", makeStanfordList(text))
    return makeStanfordList(text)
    #return text

def run2():
    print("makeStanfordList question1", makeStanfordList(question1_raw))
    print("question 1:", question1)
    return makeStanfordList(question1_raw)
    #return question1
