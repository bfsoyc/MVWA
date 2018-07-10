'''
Copyright @2018 PKU Calis.
All rights reserved.

This python script extract the sentences pairs and their label score from the WikiQA dataset and hash each word in to an interger based on a given vocabulary dictinoary
for the convenience of resuing.

Usage:
just run this script in root directory after you getting the data prepared.
'''

import os
import numpy as np
import utils

print '===================data transform======================='
inDir = './data/WikiQACorpus'
outDir = './data/WikiQACorpus'
if not os.path.exists(outDir):
	os.makedirs(outDir)

d = 300
vocbPath = './data/glove.6B/glove.6B.300d.txt'
vocbFile = open(vocbPath, "r")
vocb = {}
vocbList = []
cntWord = 0
emb = []
for line in vocbFile:
	items = line.split(' ')
	word = items[0]
	l = len(word)
	vocb[word] = cntWord
	vocbList.append(word)
	cntWord = cntWord + 1

	vec = line[l + 1:].split(' ')
	emb.append(map(float, vec)) 

emb.append([.0] * d)
np.array(emb).astype('float32').tofile(outDir + '/embMat' + str(d) + '.bin')
vocbFile.close()
# save the vocabulary 
vocPath = outDir + '/vocb.txt'
vocFile = open(vocPath, "w")
for word in vocbList:
	vocFile.writelines(word + '\t' +  str(vocb[word]) + '\n')
vocFile.close()
print 'corups vocabulary word saved at %s' % vocPath

sentAIdx = 0
sentBIdx = 1
relatedScoreIdx = 2
dataset = ['train','dev','test']
# loop over all subset
for subset in dataset:
	inFilePath = inDir + '/WikiQA-' + subset + '.txt'	
	inFile = open(inFilePath, "r")
	outFilePath = outDir + '/token-' + subset + '.txt'
	outFile = open(outFilePath, "w")
	oov = set()

	cnt = 0	# count of lines
	previous_question = ''
	q_id = 0
	write_buffer = []
	write_flag = False  # Some questions have no relevant answers. Use this flag to ignore thoes questions.
	for line in inFile:
		cnt = cnt + 1;
		items = line.strip().split('\t')
		sentA = utils.sentenceNorm(items[sentAIdx])
		sentB = utils.sentenceNorm(items[sentBIdx])
		score = items[relatedScoreIdx]

		if (previous_question != sentA):
			q_id = q_id + 1
			if (write_flag):
				for l in write_buffer:
					outFile.writelines(l)
			write_flag = False
			write_buffer[:] = []	# clear the buffer
		previous_question = sentA
		if (int(score) == 1):
			write_flag = True

		tokenA = []
		for word in sentA.split(' '):
			if word == "":
				continue
			if word in vocb:
				tokenA.append(str(vocb[word]))
			else:
				#print 'oov in line %d sentence A:\t%s\n' % (cnt, word) + sentA 
				oov.add(word)
		tokenB = []
		for word in sentB.split(' '):
			if word == "":
				continue
			if word in vocb:
				tokenB.append(str(vocb[word]))	
			else:
				#print 'oov in line %d sentence B:\t%s\n' % (cnt, word) + sentB 
				oov.add(word)

		if (len(tokenA) == 0 or len(tokenB) == 0):
			continue
		write_buffer.append(" ".join(tokenA) + '\t' + " ".join(tokenB) + '\t' + score + '\t' + str(q_id) + '\t' + str(write_flag) + '\n')
	
	if (write_flag):
		for l in write_buffer:
			outFile.writelines(l)
	
	inFile.close()
	outFile.close()
	print 'sentence token saved at %s' % outFilePath

	# save the oov
	oovPath = outDir + '/oov-' + subset + '.txt'
	oovFile = open(oovPath, "w")
	for key in oov:
		oovFile.writelines(str(key)+'\n')
	oovFile.close()	
	print 'out of vocabulary word saved at %s' % oovPath

	print 'WikiQA data subset %s preprocess done.' % subset

print '========================================================'
