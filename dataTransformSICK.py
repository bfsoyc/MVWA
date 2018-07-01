'''
Copyright @2018 PKU Calis.
All rights reserved.

This python script extract the sentences pairs and their relatedness score from the SICK dataset and hash each word in to an interger based on a given vocabulary dictinoary
for the convenience of resuing.

Usage:
just run this script in root directory after you getting the data prepared.
'''

import os
import utils
import numpy as np

print '===================data transform======================='
inFilePath = './data/SICK/SICK.txt'
inFile = open(inFilePath, "r")

outDir = './data/SICK'
if not os.path.exists(outDir):
	os.makedirs(outDir)

d = 300
# load the vocabulary, of which each row is began with a word followed by n-dimensional embedding vector
vocbPath = './data/glove.6B/glove.6B.300d.txt'
vocbFile = open(vocbPath, "r")
vocb = {}
vocbList = []
cntWord = 0 # count of words
emb = []
for line in vocbFile:
	items = line.split(' ')
	word = items[0]
	l = len(word)
	vocb[word] = cntWord
	vocbList.append(word)	# assume that there is no duplicated word in vocabulary dictionary
	cntWord = cntWord + 1

	vec = line[l+1:].split(' ')
	emb.append(map(float, vec)) 

emb.append([.0] * d) 	# the end symbol of sequences assign to the last word in the dictinoary
np.array(emb).astype('float32').tofile(outDir + '/embMat' + str(d) + '.bin')
vocbFile.close()
# save the vocabulary in the order they occurs in the embedding matrix
vocPath = outDir + '/vocb.txt'
vocFile = open(vocPath, "w")
for word in vocbList:
	vocFile.writelines(word + '\t' +  str(vocb[word]) + '\n')
vocFile.close()
print 'corups vocabulary word saved at %s' % vocPath

# the colume index of the raw data
sentAIdx = 1
sentBIdx = 2
relatedScore = 4

outFilePath = outDir + '/token.txt'
outFile = open(outFilePath, "w")
oov = set()	# store all out of vocabulary words that occurs in SICK dataset

cnt = 0	# count of lines
for line in inFile:
	cnt = cnt + 1;
	if (cnt == 1): 
		continue	# discard the first line, which is the header of the tables
	items = line.split('\t')
	sentA = utils.sentenceNorm(items[sentAIdx])
	sentB = utils.sentenceNorm(items[sentBIdx])
	score = items[relatedScore]

	tokenA = []
	for word in sentA.split(' '):
		if word == "":
			continue
		if word in vocb:
			tokenA.append(str(vocb[word]))
		else:
			print 'oov: ' + word + ' in sentence: \n' + items[sentAIdx]
			oov.add(word)
	tokenB = []
	for word in sentB.split(' '):
		if word == "":
			continue
		if word in vocb:
			tokenB.append(str(vocb[word]))	
		else:
			print 'oov: ' + word + ' in sentence: \n' + items[sentBIdx]
			oov.add(word)	

	outFile.writelines(" ".join( tokenA ) + '\t' + " ".join(tokenB) + '\t' + score + '\n')
inFile.close()
outFile.close()
print 'sentence token saved at %s' % outFilePath

# save the oov
oovPath = outDir + '/oov.txt'
oovFile = open(oovPath, "w")
for key in oov:
	oovFile.writelines(str(key)+'\n')
oovFile.close()
print 'out of vocabulary word saved at %s' % oovPath

print 'SICK data preprocess done.'
print '========================================================'
