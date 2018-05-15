class Parameters:
	def __init__( self, config):
		# some paramters have to be initialized
		self.TrainFlag = 0
		self.PredictFlag = 0

		inFile = open(config, 'r')
		for line in inFile:
			line = line.strip(' \n')
			if (len(line) <= 4):
				continue	# ignore empty line
			if (line[0] == '#'):
				continue	# ignore comments

			key, value = line.split('\t')[0:2]
			if (key == 'TrainFlag'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'PredictFlag'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'trainEmbedding'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'use_annotation_orthgonal_loss'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'embeddingSize'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'batchSize'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'cellSize'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'similarityMetric'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'learningRate'):
				exec 'self.%s = %f' % (key, float(value))
			elif (key == 'keepProb'):
				exec 'self.%s = %f' % (key, float(value))
			elif (key == 'penalty_strength'):
				exec 'self.%s = %f' % (key, float(value))
			elif (key == 'modelType'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'penaltyType'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'rnnCellType'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'forgetBias'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'activationType'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'modelLoadVersion'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'modelSavePeriod'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'itr'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'attention_aspect'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'finalFeaSize'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'hiddenSize'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'sentenceTruncate'):
				exec 'self.%s = %d' % (key, int(value))
			elif (key == 'dataset'):
				exec 'self.%s = \'%s\'' % (key, str(value))
			else:
				raise Exception("unrecognized parameter '%s' in configuration file"  % key)
		inFile.close()

		# varify some parameter
		if (self.dataset == 'WikiQA' and self.modelType != 6):
			raise Exception('only experimental model could be used on WikiQA dataset')
		if (not self.dataset in ['SICK','WikiQA','LBA']):
			raise Exception('unkonwn dataset')

		# set up some path
		modelDir = 'model/%s' % self.dataset
		logDir = 'log/%s' % self.dataset 
		extend = ''
		if (self.modelType == 1):
			modelName = '/meanFactory'
		elif (self.modelType == 2):
			if (self.similarityMetric == 1):
				modelName = '/basicRnn'
			elif (self.similarityMetric == 2):
				modelName = '/MaRNN'		
		elif (self.modelType == 3):
			modelName = '/unrollBasicRnn'
		elif (self.modelType == 4):
			modelName = '/gridRnn2D'
		elif (self.modelType == 5):
			modelName = '/selfAttentionRnn' 
		elif (self.modelType == 6):
			modelName = '/expModel'

		if (self.modelType == 2 or self.modelType==3 or self.modelType==4 or self.modelType == 5):
			if (self.rnnCellType == 1):
				extend = 'LSTM'
			elif (self.rnnCellType == 2):
				extend = 'GRU'
			else:
				raise Exception("unrecognized rnnCellType")
			extend += str( self.cellSize)

		self.modelSaveDir = modelDir + modelName + extend
		self.modelSavePath = self.modelSaveDir + '/myModel'
		self.logSaveDir = logDir + modelName + extend
		self.logSavePath = self.logSaveDir + '/myLog' 
		self.modelLoadPath = self.modelSavePath + '-' + str( self.modelLoadVersion)

	def printAll(self):
		print '======================display configuration===================== '
		for a in dir(self):
			if not a.startswith('__') and not callable( getattr(self,a)):
				print '%s:\t%s' % (a, str(getattr(self,a)))

