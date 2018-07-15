'''
Copyright @2018 PKU Calis.
All rights reserved.

Functions about library book abstract dataset.
refer to SICK.py
'''

import random
import math
import numpy as np
import utils

class LBAData():

	def __init__(self):
		self.nextIdx = 0
		self.epoch = 0
		self.maxLen = 0		

		self.tokenMap = {}
				
		self.rawData = []
		self.rawLabel = []
		self.slen = []
		self.trainSet = {}  # trainSet[ClassName] is the list of idx of train set with class 'ClassName' in rawLabel. 
		self.validSet = {}
		self.testSet = {}
		self.categories = {}
		self.digitLabel = {}

	def getNextBatch(self, batchSize):
		Categories = self.trainSet.keys()
		n = len(Categories)
		retA = []
		retB = []
		label = []
		lenA = []
		lenB = []
		# we sample positive sample one third of the batch
		num_of_pos_sample = int(batchSize / 3)
		for i in range(num_of_pos_sample):
			posCat = random.choice(Categories)  # postive category in batch
			a, b = random.sample(self.trainSet[posCat], 2)
			retA.append(self.rawData[a])
			lenA.append(self.slen[a])
			retB.append(self.rawData[b])
			lenB.append(self.slen[b])
			label.append(1.0)			

		for i in range(num_of_pos_sample, batchSize):
			cat1 = random.choice(Categories)
			cat2 = random.choice(Categories)
			while (cat1 == cat2):
				cat2 = random.choice(Categories)
			a = random.choice(self.trainSet[cat1])
			retA.append(self.rawData[a])
			lenA.append(self.slen[a])
			b = random.choice(self.trainSet[cat2])
			retB.append(self.rawData[b])
			lenB.append(self.slen[b])
			label.append(0)

		return retA, retB, label , lenA, lenB

	def truncate(self, len_limit):
		self.rawData = [sent[:len_limit] for sent in self.rawData]
		self.slen = map(lambda x:min(x, len_limit), self.slen)
		self.maxLen = max(self.slen)
		print '|------------------text truncate %3d-------------------|' % (len_limit)
		self.printInfo()

	def randomEvalOnValid(self):
		Categories = self.trainSet.keys()
		pickCat = random.choice(Categories)
		a = random.choice(self.validSet[pickCat])
		
		retA = []
		retB = []
		cat = []
		label = []
		lenA = []
		lenB = []
		for key in self.trainSet:
			B = random.sample(self.trainSet[key], 15)
			for b in B:
				retA.append(self.rawData[a])
				lenA.append(self.slen[a])
				retB.append(self.rawData[b])
				lenB.append(self.slen[b])
				if (key == pickCat):
					label.append(1.0)
				else:
					label.append(0)
				cat.append(key)
		return retA, retB, label, lenA, lenB, cat, pickCat

	def getEvalSet(self, subset = 'dev', label_set = 'all'):
		# evaluation set can be the dev subset or the test subset or the combination of them
		# assume we have N train samples and M evaluation sample.
		# then we can construct M*N sentence pairs, whose token are stored in retA and retB.
		# sentence pair is consist of sentence A and sentence B.
		# sentence A is from evaluation set while sentence B is from train samples.
		# the sentence pair reference label is the label of sentence B, this label is use for voting.
		if not subset in ['dev', 'test', 'both']:
			raise Exception('invalid subset')
		if label_set == 'all':
			label_set = ''.join(self.trainSet.keys())
		retA = []
		retB = []
		eval_label = []  # labels of samples in evaluation set
		sent_pair_ref_label = []  # sentence pair reference label
		label = []
		lenA = []
		lenB = []
		trainSet = []
		train_label = []
		for key in self.trainSet:
			trainSet = trainSet + self.trainSet[key]
			train_label = train_label + [key for i in range(len(self.trainSet[key]))]

		for key in label_set:
			evalSet = (self.validSet[key] if not subset == 'test' else []) + (self.testSet[key] if not subset == 'dev' else [])
			for a in evalSet:
				for b in trainSet:
					retA.append(self.rawData[a])
					lenA.append(self.slen[a])
					retB.append(self.rawData[b])
					lenB.append(self.slen[b])
					label.append(1.0)
				eval_label.append(key)
				sent_pair_ref_label = sent_pair_ref_label + train_label			
		return retA, retB, label, lenA, lenB, eval_label, sent_pair_ref_label, len(train_label)

	def shuffleTrainSet(self):
		random.shuffle(self.trainSet)
		self.nextIdx = 0
		print 'training set get shuffled@@@@@@@@@@@@@@@@@@@@@@@@@@@@epoch:\t %d' % self.epoch

	def printInfo(self):
		print '======================LBA statistic====================='
		print 'total categories: \t%d' % len(self.trainSet)
		print 'rough average length: \t%f' % (1.0 * sum(self.slen) / len(self.slen))	# this is not the accurate value of average length of {trainSet U validSet U testSet} 
		print 'Cat.\ttrain\tdev\ttest'	
		for key in self.trainSet:
			print '%s\t%d\t%d\t%d' % (key, len(self.trainSet[key]), len(self.validSet[key]), len(self.testSet[key]))
		
	def loadVocb(self, path):
		file_to_read = open(path, 'r')
		for line in file_to_read:
			word, word_id = line.split('\t')
			self.tokenMap[int(word_id)] = word
		file_to_read.close()

	def displaySent(self, sent, lens = None):
		if lens == None:
			lens = len(sent)
		s = []
		for token_id, itr in zip(sent, range(lens)):
			if self.tokenMap.has_key(token_id):
				s.append(self.tokenMap[token_id])
			else:
				s.append('<oov>')
		print ' '.join(s)
		return s

	# display all book abstract with given label in given subset.
	def inspectSentByLabel(self, subset, label):
		if (subset == 'dev'):
			sentSet = self.validSet[label]
		elif (subset == 'test'):
			sentSet = self.testSet[label]

		for i in sentSet:
			self.displaySent(self.rawData[i], self.slen[i])

	def save4FastText(self, saveDir):
		# save the corups for FastText model classification, noted that oov words have been deleted.
		f = open(saveDir + 'LBA4fastText-train.txt', 'w')
		for key in self.trainSet:
			ls = self.trainSet[key]
			for idx in ls:
				s = []
				for token_id, itr in zip(self.rawData[idx], range(self.slen[idx])):
					if self.tokenMap.has_key(token_id):
						s.append(self.tokenMap[token_id])
					else:
						s.append('<oov>')
				f.writelines(' '.join(s) + ' __label__' + key + '\n')
		f.close()

		f = open(saveDir + 'LBA4fastText-valid.txt', 'w')
		for key in self.validSet:
			ls = self.validSet[key]
			for idx in ls:
				s = []
				for token_id, itr in zip(self.rawData[idx], range(self.slen[idx])):
					if self.tokenMap.has_key(token_id):
						s.append(self.tokenMap[token_id])
					else:
						s.append('<oov>')
				f.writelines(' '.join(s) + ' __label__' + key + '\n')
		f.close()

		f = open(saveDir + 'LBA4fastText-test.txt', 'w')
		for key in self.testSet:
			ls = self.testSet[key]
			for idx in ls:
				s = []
				for token_id, itr in zip(self.rawData[idx], range(self.slen[idx])):
					if self.tokenMap.has_key(token_id):
						s.append(self.tokenMap[token_id])
					else:
						s.append('<oov>')
				f.writelines(' '.join(s) + ' __label__' + key + '\n')
		f.close()

def loadData(dataPath, vocbSize = 2, level = 1):
	data = LBAData()	

	inFile = open(dataPath, "r")
	maxLen = 0	# maximum length of sentences
	for r, line in enumerate(inFile):
		items = line.split('\t')
		label = utils.retriveMajorCLC(items[1], level)
		sent = map(int, items[0].split(' '))		
		data.rawData.append(sent)
		data.slen.append(len(sent))
		data.rawLabel.append(label)
		data.categories.setdefault(label, []).append(r)
		maxLen = max(maxLen, len(sent))

	data.rawData = [sent + [vocbSize - 1] * (maxLen - len(sent)) for sent in data.rawData]
	data.maxLen = maxLen

	# split the whole dataset by their categry, which is depending on the level specified.
	# e.g.
	# a sample with original label I274 is belong to categry 'I' when level is set to 1,
	#														 'I2' when level is set to 2 				
	# train/valid/test = 7/1/2	
	random.seed(5233) 
	#label_set = 'ADJKPQR'
	#label_set = 'AOQ'
	label_set = 'AEHJOPQUX'
	for key in data.categories:
		if not key in label_set:
			continue
		ls = data.categories[key]
		if (len(ls) < 200):
			print 'too few sample on category %s' % key
			continue
		rk = range(len(ls))
		random.shuffle(rk)
		N = min(len(ls), 1500)
		tr = int(N * 0.7)
		va = int(N * 0.1)
		data.trainSet[key] = [ls[i] for i in rk[:tr]]
		data.validSet[key] = [ls[i] for i in rk[tr:tr+va]]
		data.testSet[key] = [ls[i] for i in rk[tr+va:N]]
		
	# assign digit label to each category
	Categories = data.trainSet.keys()
	Categories.sort()
	for r, key in enumerate(Categories):
		data.digitLabel[key] = r
	data.printInfo()		
	return data

	
