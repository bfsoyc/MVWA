import random
import math
import numpy as np

class WikiQAData():
	rawDataA = []
	slenA = []		# sentence length
	rawDataB = []
	slenB = []
	rawLabel = []	
	trainSet = []
	validSet = []
	testSet = []
	# costume parameter for WikiQA
	candidateGroup = {}
	questionId = []	
	maxLen = 0

	def __init__( self  ):
		self.nextIdx = 0
		self.epoch = 0
		self.tokenMap = {}

	def getNextBatch( self, batchSize ):
		n = len( self.trainSet )
		if( batchSize > n ):
			raise Exception("batchSize should not bigger than total number of training samples")
		elif( self.nextIdx + batchSize > n ):
			retA = [ self.rawDataA[i] for i in self.trainSet[ self.nextIdx: ] ] + [ self.rawDataA[i] for i in self.trainSet[ : (self.nextIdx+batchSize)%n ]]
			retB = [ self.rawDataB[i] for i in self.trainSet[ self.nextIdx: ] ] + [ self.rawDataB[i] for i in self.trainSet[ : (self.nextIdx+batchSize)%n ]]
			label = [ self.rawLabel[i] for i in self.trainSet[ self.nextIdx: ] ] + [ self.rawLabel[i] for i in self.trainSet[ : (self.nextIdx+batchSize)%n ]]		
			lenA = [ self.slenA[i] for i in self.trainSet[ self.nextIdx: ] ] + [ self.slenA[i] for i in self.trainSet[ : (self.nextIdx+batchSize)%n ]]	
			lenB = [ self.slenB[i] for i in self.trainSet[ self.nextIdx: ] ] + [ self.slenB[i] for i in self.trainSet[ : (self.nextIdx+batchSize)%n ]]
			self.epoch = self.epoch + 1
			if (self.epoch % 2 == 0):
				self.shuffleTrainSet()	# shuffle the training set each epoch	
		else:
			retA = [ self.rawDataA[i] for i in self.trainSet[ self.nextIdx:self.nextIdx+batchSize ] ] 
			retB = [ self.rawDataB[i] for i in self.trainSet[ self.nextIdx:self.nextIdx+batchSize ] ] 
			label = [ self.rawLabel[i] for i in self.trainSet[ self.nextIdx:self.nextIdx+batchSize ] ]
			lenA = [ self.slenA[i] for i in self.trainSet[ self.nextIdx:self.nextIdx+batchSize ] ] 
			lenB = [ self.slenB[i] for i in self.trainSet[ self.nextIdx:self.nextIdx+batchSize ] ] 
		self.nextIdx = (self.nextIdx + batchSize ) % n
		return retA, retB, label , lenA, lenB
	
	def getValidSet( self ):
		retA = [ self.rawDataA[i] for i in self.validSet ]
		retB = [ self.rawDataB[i] for i in self.validSet ]
		label = [ self.rawLabel[i] for i in self.validSet ]
		lenA = [ self.slenA[i] for i in self.validSet ]
		lenB = [ self.slenB[i] for i in self.validSet ]
		return retA, retB, label, lenA, lenB, self.validSet

	def getTestSet( self ):
		retA = [ self.rawDataA[i] for i in self.testSet ]
		retB = [ self.rawDataB[i] for i in self.testSet ]
		label = [ self.rawLabel[i] for i in self.testSet ]
		lenA = [ self.slenA[i] for i in self.testSet ]
		lenB = [ self.slenB[i] for i in self.testSet ]
		return retA, retB, label, lenA, lenB, self.testSet

	def shuffleTrainSet( self ):
		random.shuffle( self.trainSet )
		self.nextIdx = 0
		print 'training set get shuffled@@@@@@@@@@@@@@@@@@@@@@@@@@@@epoch:\t %d' % self.epoch

	def printInfo(self):
		pass

	def loadVocb( self, path ):
		file_to_read = open( path ,'r')
		for line in file_to_read:
			word, word_id = line.split('\t')
			self.tokenMap[int(word_id)] = word
		file_to_read.close()

	def displaySent( self, sent, lens = None ):
		if lens==None:
			lens = range( len(sent) )
		s = []
		for token_id, itr in zip( sent, range(lens) ):
			if self.tokenMap.has_key( token_id ):
				s.append( self.tokenMap[token_id] )
			else:
				s.append( '<oov>' )
		print ' '.join( s )
		return s
	
	def evaluateOn(self, predict, dataset):
		if not dataset in ['dev','test']:
			raise Exception('argument dataset must be either \'dev\' or \'test\'')	
		if len(y) != len(self.validSet):
			raise Exception('length of predict is inconsistent with loaded dev dataset')		
		
		if (dataset == 'dev'):
			offset = min(self.validSet)
		else:
			offset = min(self.testSet)

		mrr = 0
		for group in self.candidateGroup[dataset]:
			pred = [predict[i-offset] for i in  group]
			idx = sorted( range(len(pred)), key = lambda k:pred[k])
			# find the first answer with score 1
			r = next( x for x in range(len(idx)) if int(self.rawLabel[group[idx[x]]]) == 1 )
			mrr = mrr + 1/(r+1)

		mrr = mrr / len(self.candidateGroup[dataset])
		return mrr			
		

def loadData( dataPathPrefix, vocbSize = 2, select = 'ALL' ):
	data = WikiQAData()	# a class to capsule all datas
	dataset = {'ALL':['train','dev','test'], 'TRAIN':['train'], 'DEV':['dev'], 'TEST':['test'] }[select]
	
	for subset in dataset:
		inFilePath = dataPathPrefix + subset + '.txt'
		inFile = open( inFilePath, "r")
		#refPath = refPathPrefix + subset+ '.txt'
		#refFile = open( refPath, "r")

		candidates = {}
		maxLen = 0	# maximum length of sentences
		bgIdx = len(data.rawLabel)
		for cnt, line in enumerate(inFile):
			items = line.split('\t')			# items = [sentA \t sentB \t score \t question_id]
			data.rawLabel.append(float(items[2]))
			q_id = int(items[3])
			data.questionId.append(q_id)
			candiates.setdefault(q_id, []).append(cnt)
			sentA = []
			for token in items[0].split(' '):
				sentA.append( int(token) )
			sentB = []
			for token in items[1].split(' '):
				sentB.append( int(token) )
			data.rawDataA.append( sentA )
			data.slenA.append( len(sentA) )
			data.rawDataB.append( sentB )
			data.slenB.append( len(sentB) )
			maxLen = max( max( maxLen, len(sentA) ),  len(sentB) )
			
		inFile.close()

		edIdx = len(data.rawLabel)
		# adjust each sentence to same length by appeding end symbol to the end
		# token 'vocbSize-1' represents for the end symbol of sequences, whose word vector is vector of zeros. 
		data.rawDataA[bgIdx:edIdx] = [ sent + [vocbSize-1]*(maxLen-len(sent)) for sent in data.rawDataA[bgIdx:edIdx] ]
		data.rawDataB[bgIdx:edIdx] = [ sent + [vocbSize-1]*(maxLen-len(sent)) for sent in data.rawDataB[bgIdx:edIdx] ]
		data.maxLen = maxLen
		

		if (subset == 'train'):
			data.trainSet = range(bgIdx, edIdx)
			data.candidateGroup['train'] = candidates
		elif (subset == 'dev'):
			data.validSet = range(bgIdx, edIdx)
			data.candidateGroup['dev'] = candidates
		elif (subset == 'test'):
			data.testSet == range(bgIdx, edIdx)
			data.candidateGroup['test'] = candidates

		

	random.seed( 5233 ) # fixed random seed 520
	random.shuffle( data.trainSet )
	
	data.printInfo()
	return data
