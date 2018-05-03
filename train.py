# my own module
import SICK
import WikiQA
import utils
import NLPModel
import ConfigReader as confrd
import random
import argparse

import numpy as np
import tensorflow as tf

class Exp:
	# static    !!!! never try to run more than 1 model in one python script
	para = 0
	datas = 0
	embMat = 0
	placehodlers = 0
	loss = 0
	prob = 0
	pred = 0
	prob_mse = 0
	mse = 0
	pr = 0

	def __init__( self, para):
		Exp.para = para
		
		dataDir = '../data/expCorpus'
			
		# load embedding matrix
		d = para.embeddingSize
		emb = np.fromfile( dataDir+'/embMat' + str(d) + '.bin', dtype = np.float32)
		emb.shape = -1,d

		# load data
		if (para.dataset == 'SICK'):
			Exp.datas = SICK.loadData( dataDir + '/inputs.txt', emb.shape[0])
		elif (para.dataset == 'WikiQA'):
			Exp.datas = WikiQA.loadData( dataDir + '/inputs.txt', emb.shape[0])
		Exp.datas.loadVocb( dataDir + '/vocb.txt')

		

		with tf.variable_scope( "embeddingLayer"):
			Exp.embMat = tf.get_variable( "embedding", initializer = emb, trainable = para.trainEmbedding)

		if (para.modelType == 1):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse , Exp.pr, Exp.emb1, Exp.ave = NLPModel.averageModel( Exp.datas.maxLen, Exp.para)
		elif (para.modelType == 2):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr = NLPModel.rnnModel( Exp.datas.maxLen, Exp.para)
		elif (para.modelType == 3):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr, Exp.ah1, Exp.ac1, Exp.ah2, Exp.ac2 \
= NLPModel.selfRnnModel( Exp.datas.maxLen, Exp.para)[0:13]
		elif (para.modelType == 4):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr, Exp.st = NLPModel.gridRnnModel( Exp.datas.maxLen, Exp.para)
		elif (para.modelType == 5):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr = NLPModel.selfAttentionRnnModel( Exp.datas.maxLen, Exp.para)
		elif (para.modelType == 6):
			Exp.placehodlers, Exp.tensorDict = NLPModel.expModel( Exp.datas.maxLen, Exp.para)

	def train(self):
		with tf.name_scope( 'train'):
			#train_step = tf.train.AdagradOptimizer( learning_rate = learningRate).minimize( loss)
			for key in self.tensorDict:
				print key
			train_step = tf.train.AdamOptimizer( self.para.learningRate ).minimize( self.tensorDict['loss'])

		saveVar = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES)	
		saver = tf.train.Saver( saveVar, max_to_keep = 50)

		sess = tf.Session()
		sess.run( tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))	# local variables for pearson correlation calculation

		# create a summary writer, add the 'graph' to the event file.	
		path = self.para.logSavePath
		fileDir = self.para.logSaveDir
		if tf.gfile.Exists( fileDir):
			tf.gfile.DeleteRecursively(fileDir)
		else:
			tf.gfile.MakeDirs(fileDir)
		writer = tf.summary.FileWriter( path, sess.graph)
		merged = tf.summary.merge_all()	

		itr = Exp.para.itr
		batchSize = Exp.para.batchSize
		print '=======================train phase=================================='
		print 'max sequence_length: %d ' % self.datas.maxLen
		for i in range( itr):
			s1,s2, score, slen1, slen2 = Exp.datas.getNextBatch( batchSize)
			feedDatas = [s1, s2, np.reshape(score,(-1,1)), slen1, slen2 ]	# reshape score from 1-d list to a ?-by-1 2-d numpy array
			_loss, _ = sess.run( [self.tensorDict['loss'], train_step], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas) })
				
			# print info and save
			if (i % self.para.modelSavePeriod != 0):
				continue	
			# different metric for different dataset
			if (self.para.dataset == 'SICK'):
				# valid set
				s1,s2, score, slen1, slen2, idx = Exp.datas.getValidSet()
				sc = np.reshape(score,(-1,1))
				feedDatas = [s1, s2, sc, slen1, slen2 ]
				_loss, _prob, _y, _merged, _prob_mse, _mse, _pr, _a1, _a2 = sess.run( [self.tensorDict['loss'], self.tensorDict['prob'], self.tensorDict['y'], merged, self.tensorDict['prob_mse'], self.tensorDict['mse'], self.tensorDict['pearson_r'], self.tensorDict['sent1_attention'], self.tensorDict['sent2_attention'] ], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas) })
				#_loss, _prob, _y, _merged, _prob_mse, _mse, _pr = sess.run( [self.loss, self.prob, self.pred, merged, self.prob_mse, self.mse, self.pr ], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas) })
				print 'valid loss: ' + str(_loss)
				print 'prob_MSE ' + str( _prob_mse)
				print '**** MSE: ' + str( _mse) + '****'
				print 'pearson_r: ' , _pr
				print 'spearman_rho: ', utils.spearman_rho( _y, sc)			 

				iid = random.randint(0,len(s1))
				'''			
				Exp.datas.displaySent( s1[iid] , slen1[iid])
				Exp.datas.displaySent( s2[iid] , slen2[iid])

				print 'attention perspective one:'				
				print np.reshape( _a1[iid,:slen1[iid],0], newshape = [-1])
				print np.reshape( _a2[iid,:slen2[iid],0], newshape = [-1])
				print 'another perspective:'
				print np.reshape( _a1[iid,:slen1[iid],1], newshape = [-1])
				print np.reshape( _a2[iid,:slen2[iid],1], newshape = [-1])
				'''				

				# we also examine on test set very time we save a model
				s1,s2, score, slen1, slen2, idx = Exp.datas.getTestSet()
				sc = np.reshape(score,(-1,1))
				feedDatas = [s1, s2, sc, slen1, slen2 ]
				_loss, _prob, _y, _merged, _prob_mse, _mse, _pr = sess.run( [self.tensorDict['loss'], self.tensorDict['prob'], self.tensorDict['y'], merged, self.tensorDict['prob_mse'], self.tensorDict['mse'], self.tensorDict['pearson_r'] ], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas) })
				
				print 'test loss: ' + str(_loss)
				print 'prob_MSE ' + str( _prob_mse)
				print 'MSE: ' + str( _mse)
				print 'pearson_r: ' , _pr
				print 'spearman_rho: ', utils.spearman_rho( _y, sc)	

			elif (self.para.dataset == 'WikiQA'):
				# valid set
				s1,s2, score, slen1, slen2, idx = Exp.datas.getValidSet()
				sc = np.reshape(score,(-1,1))
				feedDatas = [s1, s2, sc, slen1, slen2 ]
				
				_loss, _prob_pos, _merged = sess.run( [self.tensorDict['loss'], self.tensorDict['prob_of_positive'], merged ], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas) })
				print 'valid mrr:\t%f' % Exp.datas.evaluateOn(_prob_pos, 'dev')


			# save log
			writer.add_run_metadata( tf.RunMetadata() , 'itr:%d' % i)
			writer.add_summary( _merged, i)

			# save model
			path = self.para.modelSavePath
			if (i % self.para.modelSavePeriod == 0):
				if not tf.gfile.Exists(self.para.modelSaveDir):
					tf.gfile.MakeDirs(self.para.modelSaveDir)
				saver.save( sess, path, global_step = i , write_meta_graph = False)
				print 'model saved at %s with global step of %d' % (path ,i) 


	def predict(self):
		sess = tf.Session()
		sess.run( tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))	
		
		saveVar = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES)	
		saver = tf.train.Saver( saveVar)
		path = self.para.modelLoadPath	
		saver.restore( sess , path)

		# test set
		s1,s2, score, slen1, slen2,idx = Exp.datas.getTestSet()
		sc = np.reshape(score,(-1,1))
		feedDatas = [s1, s2, sc, slen1, slen2 ]
		_loss, _prob, _y, _prob_mse, _mse, _pr, _a1, _a2 = sess.run( [self.loss, self.prob, self.pred, self.prob_mse, self.mse, self.pr, self.d['sent1_attention'], self.d['sent2_attention']], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas) })

		print '=======================test phase=================================='
		self.datas.printInfo()
		print 'test loss:\t' + str(_loss)
		print 'prob_MSE:\t' + str( _prob_mse)
		print 'MSE:\t' + str( _mse)
		print 'pearson_r:\t' , _pr
		print 'spearman_rho:\t', utils.spearman_rho( _y, sc)
		
		if (self.para.modelType == 6):
			# inspect the annotation matrix
			for i in range(50):
				iid = random.randint(0,len(s1))
				sent1 = Exp.datas.displaySent( s1[iid] , slen1[iid])
				sent2 = Exp.datas.displaySent( s2[iid] , slen2[iid])
				annotation1 = np.squeeze( np.transpose(_a1[iid,:slen1[iid],:]))
				annotation2 = np.squeeze( np.transpose(_a2[iid,:slen2[iid],:]))
				utils.displayAttentionMat( sent1, annotation1, sent2, annotation2)

if __name__ == '__main__':
	print 'tensorflow version in use:  ' + tf.__version__
	
	parser = argparse.ArgumentParser( description='train model')
	parser.add_argument( '-config', type = str, nargs='+', help = 'path to the config file')
	args = parser.parse_args()

	config = args.config[0]
	print config
	para = confrd.Parameters( config)
	para.printAll()
	model = Exp(para)
	if (para.TrainFlag):
		model.train()
	if (para.PredictFlag):
		model.predict()
 

	

