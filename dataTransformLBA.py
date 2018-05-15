'''
This python script extract the abstract of books and their category label from liabrary dataset.
only save the samples with non-empty abstract
'''
import os
import csv
import numpy as np
import utils
import jieba
import io
import random

debug_flag = False

CLCIdx = 15
AbsIdx = 24
minLen = 8
inFilePath = '../data/LibBook/calis_barcode_removedup_v2over.csv'
outFileDir = '../data/expLBA'
outFilePath = outFileDir + '/cleanData%d.txt' % minLen

if not os.path.exists( outFileDir ):
	os.makedirs( outFileDir )

d = 300
# load the vocabulary, of which row is began with a word followed by n-dimensional embedding vector
vocbPath = '../data/FastText/wiki.zh.vec'
vocbFile = open( vocbPath, "r" )
fin = io.open(vocbPath, 'r', encoding = 'utf-8', newline = '\n', errors = 'ignore')
n, d = map(int, fin.readline().split())
vocb = {}
cntWord = 0
emb = []
vocbList = []
for line in fin:
	items = line.rstrip().split(' ')
	word = items[0]
	vocb[word] = cntWord
	vocbList.append(word)
	cntWord += 1
	emb.append( map(float, items[1:]))

emb.append( [.0] * d ) 	# the end symbol of sequences assign to the last word in the dictinoar
np.array( emb ).astype('float32').tofile( outFileDir + '/embMat' + str(d) + '.bin' )
vocbFile.close()
print 'total word in vocabulary: %d' % cntWord
print 'randomly pick some words to show:'
for i in range(10):
	iid = random.randint(0,cntWord)
	print vocbList[iid]


inFile = open(inFilePath, 'r')
outFile = open(outFilePath, 'w')
inCVSFile = csv.reader(inFile)
outLabelFile = open(outFileDir + '/' + 'label.txt', 'w')
outABSFile = open(outFileDir + '/' + 'abstract.txt', 'w')
outJiebaFile = open(outFileDir + '/' + 'jieba_seg.txt', 'w')

oov = set()
for r, line in enumerate(inCVSFile):
	if (r==0):
		continue
	CLC = line[CLCIdx]
	ABS = line[AbsIdx]
	if (len(ABS) > 8):
		jieba_seg = ' '.join(jieba.cut(ABS))
		outLabelFile.writelines(CLC + '\n')
		outABSFile.writelines(ABS + '\n')
		outJiebaFile.writelines(jieba_seg.encode('utf-8') + '\n')

		token = []		
		for word in jieba_seg.split(' '):
			if word == '':
				continue
			if word in vocb:
				token.append( str(vocb[word]) )
			else:
				if (debug_flag):
					print 'oov: ' + word.encode('utf-8') + ' in sentence: \n' + jieba_seg.encode('utf-8')		 
				oov.add(word)
	
		if (len(token) >= minLen):	#filter short text
			outFile.writelines( " ".join(token) + '\t' + CLC + '\n')
			
	if( r % 500 == 0 ):
		print '.',
	if( r % 10000 == 0 ):
		print 
	
inFile.close()
outFile.close()
outLabelFile.close()
outABSFile.close()
outJiebaFile.close()

# save the oov
oovPath = outFileDir + '/oov.txt'
oovFile = open( oovPath, "w" )
for key in oov:
	oovFile.writelines( key.encode('utf-8') +'\n' )
oovFile.close()

# save the vocabulary 
vocPath = outFileDir + '/vocb.txt'
vocFile = open( vocPath, "w" )
for word in vocbList:
	vocFile.writelines( word.encode('utf-8') + '\t' +  str(vocb[word]) + '\n' )
vocFile.close()
	
