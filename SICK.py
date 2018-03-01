import random
import math
import numpy as np

class SICKData():
	# static variable, shared by all object of this class	
	rawDataA = []
	slenA = []		# sentence length
	rawDataB = []
	slenB = []
	rawLabel = []	
	trainSet = []
	validSet = []
	testSet = []	
	maxLen = 0

	def __init__( self  ):
		self.nextIdx = 0

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

	def printInfo(self):
		print 'train set : ', len( self.trainSet )
		print 'valid set : ', len( self.validSet )
		print 'test set : ' , len( self.testSet )
		# statistic
		cnt = [0 for i in range(self.maxLen+1)]
		for l in self.slenA:
			cnt[l] = cnt[l]+1
		for l in self.slenB:
			cnt[l] = cnt[l]+1

		n = len( self.slenA )*2
		if n != np.sum(cnt):
			print 'n = %d ,  total count = %d, some bad sentences' % (n, np.sum(cnt) )
		print '======================SICK statistic====================='
		print 'length\tcount\tpercentage'		
		for i in range(1,self.maxLen+1):
			print '%d\t%d\t%.2f%%' %(i, cnt[i], 100.0*cnt[i]/n)
		
		


def loadData( path ):
	data = SICKData()	# a class to capsule all datas

	inFile = open( path, "r" )
	maxLen = 0	# maximum length of sentences
	for line in inFile:
		items = line.split('\t')
		data.rawLabel.append(float(items[2]))
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

	# adjust each sentence to same length by appeding token 2 to the end
	# token 2 represents for word '.' in the vacabulary. 
	data.rawDataA = [ sent + [2]*(maxLen-len(sent)) for sent in data.rawDataA ]
	data.rawDataB = [ sent + [2]*(maxLen-len(sent)) for sent in data.rawDataB ]
	data.maxLen = maxLen

	samplesCnt = len(data.rawLabel)
	rk = range( samplesCnt )

	random.seed( 2333 ) # fixed random seed 520
	random.shuffle( rk )

	
	# split dataset
	tr = 4500 # around 9/20 * samplesCnt
	va = 1000
	#ts = 4927
	data.trainSet = rk[0:tr]
	data.validSet = rk[tr:tr+va]
	data.testSet = rk[tr+va:]
	
	data.printInfo()
	return data

# score label to a sparse target distribution p	
def scoreLabel2p( score ):
	'''
	the relatedness score is range from 1 to 5, for a certian score
	p[i] = score - floor(score),			for i = floor(score)+1
	     = floor(score) + 1 - score, 		for i = floor(score)
	     = 0								otherwise
	e.g 
		score = 4.2 corresponds to the following polynomial distribution

						| x=1 | x=2 | x=3 | x=4 | x=5 |
						+-----+-----+-----+-----+-----+	
	probabilty of p(x) 	|  0  |  0  |  0  | 0.8 | 0.2 |
	'''
	P = np.zeros( [ len( score ), 5 ] )
	for idx,s in enumerate(score):
		i = int( math.floor(s) ) + 1
		if i <= 5:		# deal with corner case i==6
			P[idx][i-1] = s - math.floor(s)	
		P[idx][i-2] = 1 - (s - math.floor(s))
	return P
	
