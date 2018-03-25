'''
This python script extract the sentences pairs and their relatedness score from the SICK dataset and hash each word in to an interger based on a given vocabulary dictinoary
for the convenience of resuing.
'''
import os
import numpy as np

inFilePath = '../data/SICK/SICK.txt'
inFile = open( inFilePath, "r" )

outDir = '../data/expCorpus'
if not os.path.exists( outDir ):
	os.makedirs( outDir )

d = 300
# load the vocabulary, of which row is began with a word followed by n-dimensional embedding vector
vocbPath = '../data/glove.6B/glove.6B.300d.txt'
vocbFile = open( vocbPath, "r" )
# vocbSave = open( outDir + '/embMat' + str(d) , "w" ) # store in ascii
vocb = {}
vocbList = []
cntWord = 0 # count of words
emb = []
for line in vocbFile:
	items = line.split(' ')
	word = items[0]
	l = len(word)
	vocb[word] = cntWord
	vocbList.append( word )	# assume that there is no duplicated word in vocabulary dictionary
	cntWord = cntWord + 1

	vec = line[l+1:].split(' ')
	emb.append( map( float, vec ) ) 
	#vocbSave.writelines( line[l+1:] )

emb.append( [.0] * d ) 	# the end symbol of sequences assign to the last word in the dictinoary

np.array( emb ).astype('float32').tofile( outDir + '/embMat' + str(d) + '.bin' )
vocbFile.close()
#vocbSave.close() 


# sentence normalization: transform capital letter into lowcase letter and remove all charactors out of alphabet ( a-z )
def sentenceNorm( sent ):
	sent = sent.lower()
	l = len( sent )
	s = ''
	for i in range(l):
		if( sent[i]=='\'' or sent[i]=='\\' or sent[i]=='/' ):
			if( i+1 < len and sent[i+1]!=' '):
				s = s + ' '
		elif( sent[i]==',' or sent[i]=='.' ):
			continue
		else:
			s = s + sent[i]
	return s

sentAIdx = 1
sentBIdx = 2
relatedScore = 4

dataPath = outDir + '/inputs.txt'

outFile = open( dataPath, "w" )
oov = set()	# store all out of vocabulary words that occurs in SICK dataset

cnt = 0	# count of lines
for line in inFile:
	cnt = cnt + 1;
	if( cnt == 1 ): 
		continue	# discard the first line, which is the header of the tables
	items = line.split('\t')
	sentA = sentenceNorm( items[ sentAIdx] )
	sentB = sentenceNorm( items[ sentBIdx] )
	score = items[ relatedScore ]

	tokenA = []
	for word in sentA.split(' '):
		if word == "":
			continue
		if word in vocb:
			tokenA.append( str(vocb[word]) )
		else:
			print 'oov: ' + word + ' in sentence: \n' + sentA 
			oov.add(word)
	tokenB = []
	for word in sentB.split(' '):
		if word == "":
			continue
		if word in vocb:
			tokenB.append( str(vocb[word]) )	
		else:
			print 'oov: ' + word + ' in sentence: \n' + sentB 
			oov.add(word)
	
	# debug
	if( cnt == 9258 ):
		print 'line 9258'
		print sentA, sentB
		print tokenA, tokenB

	outFile.writelines( " ".join( tokenA ) + '\t' + " ".join(tokenB) + '\t' + score + '\n' )
	if( cnt % 500 == 0 ):
		print '.'

inFile.close()
outFile.close()

# save the oov
oovPath = outDir + '/oov.txt'
oovFile = open( oovPath, "w" )
for key in oov:
	oovFile.writelines( str(key)+'\n' )
oovFile.close()

# save the vocabulary 
vocPath = outDir + '/vocb.txt'
vocFile = open( vocPath, "w" )
for word in vocbList:
	vocFile.writelines( word + '\t' +  str(vocb[word]) + '\n' )
vocFile.close()

