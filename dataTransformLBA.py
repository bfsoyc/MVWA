'''
Copyright @2018 PKU Calis.
All rights reserved.

This python script extract the abstract of books and their category label from liabrary dataset.
The samples with empty abstracta are discarded.
Note that the LBA dataset isn't an open dataset, authorized use only.
'''

import os
import csv
import numpy as np
import utils
import jieba
import io
import random

print '===================data transform======================='
inDir = './data/LibBA'
outDir = './data/LibBA'

if not os.path.exists(outDir):
	os.makedirs(outDir)

d = 300
vocbPath = './data/FastText/wiki.zh.vec'  # need an chinese embbedding vocabulary
vocbFile = open(vocbPath, "r")
fin = io.open(vocbPath, 'r', encoding = 'utf-8', newline = '\n', errors = 'ignore')
n, d = map(int, fin.readline().split())
vocb = {}
vocbList = []
cntWord = 0
emb = []
for line in fin:
	items = line.rstrip().split(' ')
	word = items[0]
	vocb[word] = cntWord
	vocbList.append(word)
	cntWord += 1
	emb.append(map(float, items[1:]))

emb.append([.0] * d) 	# the end symbol of sequences assign to the last word in the dictinoar
np.array(emb).astype('float32').tofile(outDir + '/embMat' + str(d) + '.bin')
vocbFile.close()
# save the vocabulary 
vocPath = outDir + '/vocb.txt'
vocFile = open(vocPath, "w")
for word in vocbList:
	vocFile.writelines(word.encode('utf-8') + '\t' +  str(vocb[word]) + '\n')
vocFile.close()
print 'corups vocabulary word saved at %s' % vocPath

inFilePath = inDir + '/calis_barcode_removedup_v2over.csv'
inFile = open(inFilePath, 'r')
outFilePath = outDir + '/token.txt'
outFile = open(outFilePath, 'w')

inCVSFile = csv.reader(inFile)
outLabelFile = open(outDir + '/label.txt', 'w')
outABSFile = open(outDir + '/abstract.txt', 'w')  # save the abstract string
outJiebaFile = open(outDir + '/jieba_seg.txt', 'w')  # save the segmentation of abstract string

debug_flag = False
CLCIdx = 15
AbsIdx = 24
minLen = 8
oov = set()
for r, line in enumerate(inCVSFile):
	if (r==0):
		continue
	CLC = line[CLCIdx]
	ABS = line[AbsIdx]
	if (len(ABS) > 8):  # filter short sentence
		jieba_seg = ' '.join(jieba.cut(ABS))
		outLabelFile.writelines(CLC + '\n')
		outABSFile.writelines(ABS + '\n')
		outJiebaFile.writelines(jieba_seg.encode('utf-8') + '\n')

		token = []		
		for word in jieba_seg.split(' '):
			if word == '':
				continue
			if word in vocb:
				token.append(str(vocb[word]))
			else:
				if (debug_flag):
					print 'oov: ' + word.encode('utf-8') + ' in sentence: \n' + jieba_seg.encode('utf-8')		 
				oov.add(word)
	
		if (len(token) >= minLen):	#filter short text
			outFile.writelines(" ".join(token) + '\t' + CLC + '\n')
			
	if(r % 500 == 0):
		print '.',
	if(r % 10000 == 0):
		print 
print ' '	
inFile.close()
outFile.close()
outLabelFile.close()
outABSFile.close()
outJiebaFile.close()

# save the oov
oovPath = outDir + '/oov.txt'
oovFile = open(oovPath, "w")
for key in oov:
	oovFile.writelines(key.encode('utf-8') +'\n')
oovFile.close()
print 'out of vocabulary word saved at %s' % oovPath

print 'LibBA data preprocess done.'
print '========================================================'

