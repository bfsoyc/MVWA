class Parameters:
	def __init__( self, config ):
		# some paramters have to be initialized
		self.TrainFlag = 0
		self.PredictFlag = 0


		inFile = open( config, 'r' )
		for line in inFile:
			line = line.strip(' \n')
			if( len(line) <= 4 ):
				continue	# ignore empty line
			if( line[0] == '#' ):
				continue	# ignore comments

			key, value = line.split('\t')[0:2]
			if( key == 'TrainFlag' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'PredictFlag' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'trainEmbedding' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'embeddingSize' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'batchSize' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'cellSize' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'similarityMetric' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'learningRate' ):
				exec 'self.%s = %f' % (key, float(value) )
			elif( key == 'keepProb' ):
				exec 'self.%s = %f' % (key, float(value) )
			elif( key == 'modelType' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'rnnCellType' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'forgetBias' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'activationType' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'modelLoadVersion' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'modelSavePeriod' ):
				exec 'self.%s = %d' % (key, int(value) )
			elif( key == 'itr' ):
				exec 'self.%s = %d' % (key, int(value) )
			else:
				raise Exception("unrecognized parameter '%s' in configuration file"  % key)
		inFile.close()

		# set up some path
		modelDir = 'model'
		logDir = 'log'
		extend = ''
		if( self.modelType == 1 ):
			modelName = '/meanFactory'
		elif( self.modelType == 2 ):
			if( self.similarityMetric == 1 ):
				modelName = '/basicRnn'
			elif( self.similarityMetric == 2 ):
				modelName = '/MaRNN'		
		elif( self.modelType == 3 ):
			modelName = '/unrollBasicRnn'
		
		if( self.modelType == 2 or self.modelType==3 ):
			if( self.rnnCellType == 1 ):
				extend = 'LSTM'
			elif( self.rnnCellType == 2 ):
				extend = 'GRU'
			else:
				raise Exception("unrecognized rnnCellType")
			extend += str( self.cellSize )

		self.modelSavePath = modelDir + modelName + extend + '/myModel'
		self.logSaveDir = logDir + modelName + extend
		self.logSavePath = self.logSaveDir + '/myLog' 
		self.modelLoadPath = self.modelSavePath + '-' + str( self.modelLoadVersion )

