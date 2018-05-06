'''
This python script extract the sentences pairs and their label score from the WikiQA dataset and hash each word in to an interger based on a given vocabulary dictinoary
for the convenience of resuing.
'''
import os
import numpy as np
import utils

inFileDir = '../data/WikiQACorpus'
outFileDir = '../data/expQACorpus'
dataset = ['train','dev','test']
vocbPath = '../data/glove.6B/glove.6B.300d.txt'
if not os.path.exists( outFileDir ):
	os.makedirs( outFileDir )

d = 300
# load the vocabulary, of which row is began with a word followed by n-dimensional embedding vector
vocbFile = open( vocbPath, "r" )
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

emb.append( [.0] * d ) 	# the end symbol of sequences assign to the last word in the dictinoary
embMat_bin_path = outFileDir + '/embMat' + str(d) + '.bin'
if  not os.path.exists(embMat_bin_path):
	np.array( emb ).astype('float32').tofile( embMat_bin_path )
vocbFile.close()

# loop over all subset
for subset in dataset:
	inFilePath = inFileDir + '/WikiQA-' + subset + '.txt'	
	inFile = open( inFilePath, "r" )

	sentAIdx = 0
	sentBIdx = 1
	relatedScore = 2

	outPath = outFileDir + '/' + subset + '.txt'

	outFile = open( outPath, "w" )
	oov = set()	# out of vocabulary words

	cnt = 0	# count of lines
	previous_question = ''
	q_id = 0
	write_buffer = []
	write_flag = False
	for line in inFile:
		cnt = cnt + 1;
		items = line.strip().split('\t')
		sentA = utils.sentenceNorm( items[ sentAIdx] )
		sentB = utils.sentenceNorm( items[ sentBIdx] )
		score = items[ relatedScore ]

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
				tokenA.append( str(vocb[word]) )
			else:
				print 'oov in line %d sentence A:\t%s\n' % (cnt, word) + sentA 
				oov.add(word)
		tokenB = []
		for word in sentB.split(' '):
			if word == "":
				continue
			if word in vocb:
				tokenB.append( str(vocb[word]) )	
			else:
				print 'oov in line %d sentence B:\t%s\n' % (cnt, word) + sentB 
				oov.add(word)

		if (len(tokenA) == 0 or len(tokenB) == 0):
			continue
		write_buffer.append( " ".join( tokenA ) + '\t' + " ".join(tokenB) + '\t' + score + '\t' + str(q_id) + '\t' + str(write_flag) + '\n' )
		if( cnt % 500 == 0 ):
			print '.'
	
	if (write_flag):
		for l in write_buffer:
			outFile.writelines(l)
	
	inFile.close()
	outFile.close()

	# save the oov
	oovPath = outFileDir + '/' + subset + '-oov.txt'
	oovFile = open( oovPath, "w" )
	for key in oov:
		oovFile.writelines( str(key)+'\n' )
	oovFile.close()

	# save the vocabulary 
	vocPath = outFileDir + '/' + subset + '-vocb.txt'
	vocFile = open( vocPath, "w" )
	for word in vocbList:
		vocFile.writelines( word + '\t' +  str(vocb[word]) + '\n' )
	vocFile.close()
