import random
import math
import numpy as np
import utils

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
	maxLen = 0

	def __init__( self  ):
		self.nextIdx = 0
		self.epoch = 0
		self.tokenMap = {}

	def getNextBatch( self, batchSize):
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
				pass
				#self.shuffleTrainSet()	# shuffle the training set each epoch	
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
	
	def truncate(self, len_limit):
	 	self.rawDataA = [sent[:len_limit] for sent in self.rawDataA]
		self.rawDataB = [sent[:len_limit] for sent in self.rawDataB]
		self.slenA = map(lambda x:min(x,len_limit), self.slenA)
		self.slenB = map(lambda x:min(x,len_limit), self.slenB)
		self.maxLen = max(max(self.slenA), max(self.slenB))
		print '|--------------------text truncate %3d--------------------|' % (len_limit)
		self.printInfo()

	def printInfo(self):
		print '======================WikiQA statistic====================='
		print 'subset\t#Q\t#Cand\tAve.Q\tAve.C'
		for subset in self.candidateGroup:
			if (subset == 'train'):
				setIdx = self.trainSet
			elif (subset == 'test'):
				setIdx = self.testSet
			elif (subset == 'dev'):
				setIdx = self.validSet
			else:
				raise Exception('subset %s not found' % subset)
			groups = self.candidateGroup[subset]
			qNum = len(groups)
			cNum = len(setIdx)
			qAveLen = 0.0
			for qId in groups:
				qAveLen += self.slenA[groups[qId][0]]
			qAveLen /= qNum
			cAveLen = 0.0
			for i in setIdx:
				cAveLen += self.slenB[i]
			cAveLen /= cNum
			print '%s\t%d\t%d\t%.2f\t%.2f' % (subset, qNum, cNum, qAveLen, cAveLen)

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
	
	def displayQuestion(self, predict, randomNum, dataset):
		if not dataset in ['dev','test']:
			raise Exception('argument dataset must be either \'dev\' or \'test\'')	
				
		if (dataset == 'dev'):
			if len(predict) != len(self.validSet):
				raise Exception('length of predict is inconsistent with loaded dev dataset')		
			offset = min(self.validSet)
		else:
			if len(predict) != len(self.testSet):
				raise Exception('length of predict is inconsistent with loaded test dataset')	
			offset = min(self.testSet)
		key = self.candidateGroup[dataset].keys()[randomNum]
		group = self.candidateGroup[dataset][key]
		pred = [predict[i-offset] for i in  group]
		idx = sorted( range(len(pred)), key = lambda k:pred[k], reverse = True)
		sorted_group = [group[i] for i in idx]
		print '***************Question *************** :'
		self.displaySent(self.rawDataA[group[i]], self.slenA[group[i]])
		r = next( x for x in range(len(idx)) if int(self.rawLabel[group[idx[x]]]) == 1 )
		print 'the first relevant answer ranked at %d' % (r + 1)
		for r,i in enumerate(sorted_group):
			print '+--rank %d answer:\t(label:%d\tprediction:%f)' % (r+1, self.rawLabel[i], predict[i-offset])
			#self.displaySent(self.rawDataA[i], self.slenA[i])
			self.displaySent(self.rawDataB[i], self.slenB[i])

	def evaluateOn(self, predict, dataset):
		if not dataset in ['dev','test']:
			raise Exception('argument dataset must be either \'dev\' or \'test\'')	
				
		if (dataset == 'dev'):
			if len(predict) != len(self.validSet):
				raise Exception('length of predict is inconsistent with loaded dev dataset')		
			offset = min(self.validSet)
		else:
			if len(predict) != len(self.testSet):
				raise Exception('length of predict is inconsistent with loaded test dataset')	
			offset = min(self.testSet)

		RR = 0.0	# summation of reciprocal rank
		AP = 0.0	# summation of average percision
		qCnt = 0	# distinct question count
		rnk = []
		for key in self.candidateGroup[dataset]:
			group = self.candidateGroup[dataset][key]
			pred = [predict[i-offset] for i in  group]
			relevant = [int(self.rawLabel[i]) for i in group]
			RR += utils.computeRR(pred, relevant)
			AP += utils.computeAP(pred, relevant)
			qCnt += 1
		MRR = RR / qCnt
		MAP = AP / qCnt
		return MRR, MAP
		

def loadData( dataPathPrefix, vocbSize = 2, select = 'ALL' ):
	data = WikiQAData()	# a class to capsule all datas
	dataset = {'ALL':['train','dev','test'], 'TRAIN':['train'], 'DEV':['dev'], 'TEST':['test'] }[select]
	
	maxLen = 0	# maximum length of sentences
	cnt = 0
	for subset in dataset:
		inFilePath = dataPathPrefix + subset + '.txt'
		inFile = open( inFilePath, "r")

		candidates = {}
		q_ids = []		
		bgIdx = len(data.rawLabel)
		for line in inFile:
			items = line.strip().split('\t')			# items = [sentA \t sentB \t score \t question_id]
			data.rawLabel.append(float(items[2]))
			q_id = int(items[3])
			candidates.setdefault(q_id, []).append(cnt)
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
			cnt = cnt + 1
			
		inFile.close()

		edIdx = len(data.rawLabel)
		if (subset == 'train'):
			data.trainSet = range(bgIdx, edIdx)
			data.candidateGroup['train'] = candidates
		elif (subset == 'dev'):
			data.validSet = range(bgIdx, edIdx)
			data.candidateGroup['dev'] = candidates
		elif (subset == 'test'):
			data.testSet = range(bgIdx, edIdx)
			data.candidateGroup['test'] = candidates

	# adjust each sentence to same length by appeding end symbol to the end
	# token 'vocbSize-1' represents for the end symbol of sequences, whose word vector is vector of zeros. 
	data.rawDataA = [ sent + [vocbSize-1]*(maxLen-len(sent)) for sent in data.rawDataA ]
	data.rawDataB = [ sent + [vocbSize-1]*(maxLen-len(sent)) for sent in data.rawDataB ]
	data.maxLen = maxLen		


	random.seed( 5233 ) # fixed random seed 520
	#random.shuffle( data.trainSet )
	
	data.printInfo()
	return data
