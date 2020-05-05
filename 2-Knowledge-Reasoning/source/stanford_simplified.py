# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:27:47 2017

@author: noel
"""

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

punctuation = {".", "!", "?"}

question1 = [['ROOT', 'SBARQ', 'WHNP', 'WP', 'What', 'SQ', 'VBZ', 'is', 'NP', 'NP', 'DT', 'the', 'NN', 'capital', 'PP', 'IN', 'of', 'NP', 'NNP', 'Estonia', '.', '?']]
question2 = ['ROOT', 'SBARQ', 'WHADVP', 'WRB', 'Where', 'SQ', 'VBZ', 'is', 'NP', 'NNP', 'Tallinn', '.', '?']
question3 = [['(ROOT', '(SBARQ', '(WHNP', '(WDT', 'Which)', '(NN', 'city))', '(SQ', '(VBZ', 'is)', '(NP', '(NP', '(NN', 'west))', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Tallinn)))))', '(.', '?)))']]

#text = "Tallinn is capital of Estonia. Tallinn is in Estonia."

textA = ['(ROOT', '(S', '(NP', '(NNP', 'Barack)', '(NNP', 'Obama))', '(VP', '(VBZ', 'eats)', '(NP', '(NP', '(DT', 'the)', '(NN', 'dog))', '(CC', 'and)', '(NP', '(DT', 'the)', '(NN', 'cat))))', '(.', '.)))']
textB = ['(ROOT', '(S', '(NP', '(NNP', 'Robert))', '(VP', '(VBD', 'bought)', '(NP', '(NP', '(NN', 'cheese))', '(CC', 'and)', '(NP', '(NNP', 'Joe)', '(NNS', 'drinks)', '(NN', 'water))))', '(.', '.)))']

text_1 = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(ADVP', '(RB', 'north)', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Paris)))))', '(.', '.)))']
text_2 = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(ADVP', '(RB', 'east)', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Paris)))))', '(.', '.)))']
 
text21 = [['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(ADVP', '(RB', 'north)', '(PP', '(IN', 'of)', '(NP', '(NNP', 'France)))))', '(.', '.)))'], ['(ROOT', '(S', '(NP', '(PRP', 'It))', '(VP', '(VBZ', 'likes)', '(NP', '(NN', 'fruit)))', '(.', '.)))']]

text1 = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(NP', '(NP', '(NN', 'capital))', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Estonia)))))', '(.', '.)))'], ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(PP', '(IN', 'in)', '(NP', '(NNP', 'Estonia))))', '(.', '.)))']
#text = text1

#text21 = [['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(ADVP', '(RB', 'west)', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Tallinn)))))', '(.', '.)))'], ['(ROOT', '(S', '(NP', '(PRP', 'It))', '(VP', '(VBZ', 'likes)', '(NP', '(NN', 'fruit)))', '(.', '.)))']]

#text = ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(NP', '(NP', '(NN', 'capital))', '(PP', '(IN', 'of)', '(NP', '(NNP', 'Estonia)))))', '(.', '.)))'], ['(ROOT', '(S', '(NP', '(NNP', 'Tallinn))', '(VP', '(VBZ', 'is)', '(PP', '(IN', 'in)', '(NP', '(NNP', 'Estonia))))', '(.', '.)))']

question28 = [['(ROOT', '(SBARQ', '(WHNP', '(WP', 'What))', '(SQ', '(VBZ', 'is)', '(NP', '(NN', 'banana)))', '(.', '?)))'], ['(ROOT', '(SBARQ', '(WHNP', '(WP', 'What))', '(SQ', '(VBZ', 'is)', '(NP', '(NN', 'guitar)))', '(.', '?)))']]


def run1():
    return text1

def run2():
    return question1
