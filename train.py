# my own module
import SICK
import NLPModel
import ConfigReader as confrd

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

	def __init__( self, para ):
		Exp.para = para
		
		dataDir = '../data/expCorpus'
			
		# load embedding matrix
		d = para.embeddingSize
		emb = np.fromfile( dataDir+'/embMat' + str(d) + '.bin', dtype = np.float32 )
		emb.shape = -1,d

		# load data
		Exp.datas = SICK.loadData( dataDir + '/inputs.txt', emb.shape[0] )

		

		with tf.variable_scope( "embeddingLayer" ):
			Exp.embMat = tf.get_variable( "embedding", initializer = emb, trainable = para.trainEmbedding )

		if( para.modelType == 1 ):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse , Exp.pr, Exp.emb1, Exp.ave = NLPModel.averageModel( Exp.datas.maxLen )
		elif( para.modelType == 2 ):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr = NLPModel.rnnModel( Exp.datas.maxLen, Exp.para )
		elif( para.modelType == 3):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr, Exp.ah1, Exp.ac1, Exp.ah2, Exp.ac2 \
= NLPModel.selfRnnModel( Exp.datas.maxLen, Exp.para )[0:13]
		elif( para.modelType == 4 ):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr, Exp.st = NLPModel.gridRnnModel( Exp.datas.maxLen, Exp.para )
		elif( para.modelType == 5 ):
			Exp.placehodlers, Exp.loss, Exp.prob, Exp.pred, Exp.prob_mse, Exp.mse, Exp.outputs1, Exp.outputs2, Exp.pr, Exp.ac, Exp.w, Exp.f = NLPModel.selfAttentionRnnModel( Exp.datas.maxLen, Exp.para )
		
	
	# compute the spearman's rho
	def __spearman_rho( self, predict, label ):
		EPS = 1e-6
		p = predict.flatten()
		l = label.flatten()
		args_p = p.argsort()
		args_l = l.argsort()
		
		n = len(args_p)
		front = 0
		rnk = 1
		rk_p = np.zeros( shape=(n,1) )
		while front < n:
			tail = front
			while( tail < n and abs(p[args_p[tail]]-p[args_p[front]]) < EPS ):
				tail = tail + 1
			r = rnk + 0.5*(tail-1-front)
			for j in range(front,tail):
				rk_p[args_p[j]] = r
			front = tail
			rnk = front + 1
	
		front = 0
		rnk = 1
		rk_l = np.zeros( shape=(n,1) )
		while front < n:
			tail = front
			while( tail < n and abs(l[args_l[tail]]-l[args_l[front]]) < EPS ):
				tail = tail + 1
			r = rnk + 0.5*(tail-1-front)
			#print 'tail, front, r', tail, front, r
			for j in range(front,tail):
				rk_l[args_l[j]] = r
			front = tail
			rnk = front + 1
		
		corrArr = np.corrcoef( np.column_stack( (rk_p, rk_l )) , rowvar = 0)

		return corrArr[0,1]


	def train(self):
		with tf.name_scope( 'train' ):
			#train_step = tf.train.AdagradOptimizer( learning_rate = learningRate ).minimize( loss )
			train_step = tf.train.AdamOptimizer( self.para.learningRate  ).minimize( self.loss )

		saveVar = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )	
		saver = tf.train.Saver( saveVar, max_to_keep = 50 )

		sess = tf.Session()
		sess.run( tf.group(tf.global_variables_initializer(), tf.local_variables_initializer() ) )	# local variables for pearson correlation calculation

		# create a summary writer, add the 'graph' to the event file.	
		path = self.para.logSavePath
		fileDir = self.para.logSaveDir
		if tf.gfile.Exists( fileDir ):
			tf.gfile.DeleteRecursively( fileDir )
		writer = tf.summary.FileWriter( path, sess.graph )
		merged = tf.summary.merge_all()	

		itr = Exp.para.itr
		batchSize = Exp.para.batchSize
		print '=======================train phase=================================='
		print 'max sequence_length: %d ' % self.datas.maxLen
		for i in range( itr ):
			s1,s2, score, slen1, slen2 = Exp.datas.getNextBatch( batchSize )
			#y = SICK.scoreLabel2p( score )
			feedDatas = [s1, s2, np.reshape(score,(-1,1)), slen1, slen2 ]	# reshape score from 1-d list to a ?-by-1 2-d numpy array
			_loss, _ = sess.run( [self.loss, train_step], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas ) } )
				
			# print info and save
			if( i % self.para.modelSavePeriod == 0 ):	
				print 'train loss: ' + str(_loss)
				# valid set
				s1,s2, score, slen1, slen2, idx = Exp.datas.getValidSet( )
				sc = np.reshape(score,(-1,1))
				feedDatas = [s1, s2, sc, slen1, slen2 ]
				_loss, _prob, _y, _merged, _prob_mse, _mse, _pr = sess.run( [self.loss, self.prob, self.pred, merged, self.prob_mse, self.mse, self.pr ], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas ) } )
				print 'valid loss: ' + str(_loss)
				print 'prob_MSE ' + str( _prob_mse )
				print '**** MSE: ' + str( _mse ) + '****'
				print 'pearson_r: ' , _pr
				print 'spearman_rho: ', self.__spearman_rho( _y, sc )			 

				# we also examine on test set very time we save a model
				s1,s2, score, slen1, slen2, idx = Exp.datas.getTestSet( )
				sc = np.reshape(score,(-1,1))
				feedDatas = [s1, s2, sc, slen1, slen2 ]
				_loss, _prob, _y, _merged, _prob_mse, _mse, _pr, _ac, _w, _f = sess.run( [self.loss, self.prob, self.pred, merged, self.prob_mse, self.mse, self.pr, self.ac, self.w, self.f ], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas ) } )

				print 'test loss: ' + str(_loss)
				print 'prob_MSE ' + str( _prob_mse )
				print 'MSE: ' + str( _mse )
				print 'pearson_r: ' , _pr
				print 'spearman_rho: ', self.__spearman_rho( _y, sc )				
				
	
				print 'slen1: %d, slen2: %d' % (slen1[1], slen2[1] )
				max1 = np.max( slen1 )
				print np.reshape( _ac[:,1,0], newshape = [ -1] )
				print np.reshape( _w[:,1,1], newshape = [-1] )
				print  _f[:2,0] 
				#print _emb1[9,:,:1]
				#print _ave[9,:1]				
				#max1 = np.max( slen1 )
				#max2 = np.max( slen2 )
				#print _state.shape
				#print np.reshape( _state[:,:,1:2, 1:2], newshape = [ max2+1, max1 ]) 
				#print  slen2[:20]
				#print idx[:20]
				#print _y[:20].tolist()
				#print sc[:20].tolist()
				#print _prob

				# save log
				writer.add_run_metadata( tf.RunMetadata() , 'itr:%d' % i )
				writer.add_summary( _merged, i )

				# save model
				path = self.para.modelSavePath
				if( i % self.para.modelSavePeriod == 0 ):
					#saver.save( sess, path, global_step = i , write_meta_graph = False )
					print 'model saved at %s with global step of %d' % (path ,i ) 

				Exp.datas.shuffleTrainSet()

	def predict(self):
		sess = tf.Session()
		sess.run( tf.group(tf.global_variables_initializer(), tf.local_variables_initializer() ) )	
		
		saveVar = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES )	
		saver = tf.train.Saver( saveVar )
		path = self.para.modelLoadPath	
		saver.restore( sess , path)

		# test set
		s1,s2, score, slen1, slen2,idx = Exp.datas.getTestSet( )
		sc = np.reshape(score,(-1,1))
		feedDatas = [s1, s2, sc, slen1, slen2 ]
		_loss, _prob, _y, _prob_mse, _mse, _pr = sess.run( [self.loss, self.prob, self.pred, self.prob_mse, self.mse, self.pr], feed_dict = { placeholder: feedData  for placeholder,feedData in zip( self.placehodlers, feedDatas ) } )

		print '=======================test phase=================================='
		self.datas.printInfo()
		print 'test loss: ' + str(_loss)
		print 'prob_MSE: ' + str( _prob_mse )
		print 'MSE: ' + str( _mse )
		print 'pearson_r: ' , _pr
		print 'spearman_rho: ', self.__spearman_rho( _y, sc )
		#variable = np.column_stack( (sc, _y) )
		#print 'cov: ',  np.corrcoef(variable,rowvar=0)  



if __name__ == '__main__':
	print 'tensorflow version in use:  ' + tf.__version__
	
	config = 'config.txt'
	para = confrd.Parameters( config )
	para.printAll()
	model = Exp(para)
	if( para.TrainFlag ):
		model.train()
	if( para.PredictFlag ):
		model.predict()
 

	

