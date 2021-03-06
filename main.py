'''
Copyright @2018 PKU Calis.
All rights reserved.

The main python script to run the model.

Usage:
run the following command in terminal
	python main.py -config path_to_config_file
'''
import SICK
import WikiQA
import LBA
import utils
import NLPModel
import ConfigReader as confrd

import random
import argparse
import numpy as np
import tensorflow as tf
import os

class modelManeger:

	def __init__(self, para):
		self.para = para
		
		# load embedding matrix
		d = para.embeddingSize
		emb = np.fromfile(para.dataPath + '/embMat' + str(d) + '.bin', dtype = np.float32)
		emb.shape = -1,d

		# load data
		if (para.dataset == 'SICK'):
			self.datas = SICK.loadData(para.dataPath + '/token.txt', emb.shape[0])
		elif (para.dataset == 'WikiQA'):
			self.datas = WikiQA.loadData(para.dataPath, emb.shape[0])
			self.datas.truncate(len_limit = para.sentenceTruncate)
		elif (para.dataset == 'LBA'):
			self.datas = LBA.loadData(para.dataPath + '/token.txt', emb.shape[0])
			self.datas.truncate(len_limit = para.sentenceTruncate)
		self.datas.loadVocb(para.dataPath + '/vocb.txt')

		with tf.variable_scope("embeddingLayer"):
			self.embMat = tf.get_variable("embedding", initializer = emb, trainable = para.trainEmbedding)

		if (para.modelType == 1):
			self.placehodlers, self.loss, self.prob, self.pred, self.prob_mse, self.mse , self.pr, self.emb1, self.ave = NLPModel.averageModel(self.datas.maxLen, self.para)
		elif (para.modelType == 2):
			self.placehodlers, self.loss, self.prob, self.pred, self.prob_mse, self.mse, self.outputs1, self.outputs2, self.pr = NLPModel.rnnModel(self.datas.maxLen, self.para)
		elif (para.modelType == 3):
			self.placehodlers, self.loss, self.prob, self.pred, self.prob_mse, self.mse, self.outputs1, self.outputs2, self.pr, self.ah1, self.ac1, self.ah2, self.ac2 \
= NLPModel.selfRnnModel(self.datas.maxLen, self.para)[0:13]
		elif (para.modelType == 4):
			self.placehodlers, self.loss, self.prob, self.pred, self.prob_mse, self.mse, self.outputs1, self.outputs2, self.pr, self.st = NLPModel.gridRnnModel(self.datas.maxLen, self.para)
		elif (para.modelType == 5):
			self.placehodlers, self.loss, self.prob, self.pred, self.prob_mse, self.mse, self.outputs1, self.outputs2, self.pr = NLPModel.selfAttentionRnnModel(self.datas.maxLen, self.para)
		elif (para.modelType == 6):
			self.placehodlers, self.tensorDict = NLPModel.expModel(self.datas.maxLen, self.para)

	def train(self):
		with tf.name_scope('train'):
			train_step = tf.train.AdamOptimizer(self.para.learningRate).minimize(self.tensorDict['loss'])

		variable_to_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)	
		saver = tf.train.Saver(variable_to_save, max_to_keep = 50)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True  # dynamic allow gpu resource to the program
		sess = tf.Session(config = config)
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))  # local variables is for pearson correlation calculation

		# create a summary writer, add the 'graph' to the event file.	
		logPath = self.para.logSavePath
		logDir = self.para.logSaveDir
		if tf.gfile.Exists(logDir):
			tf.gfile.DeleteRecursively(logDir)
		else:
			tf.gfile.MakeDirs(logDir)
		writer = tf.summary.FileWriter(logPath, sess.graph)
		merged = tf.summary.merge_all()  # error arise if no summary to merge

		itr = self.para.itr
		batchSize = self.para.batchSize
		save_flag = False
		print '=======================train phase======================='
		for i in range(itr):
			s1, s2, score, slen1, slen2 = self.datas.getNextBatch(batchSize)
			feedDatas = [s1, s2, np.reshape(score,(-1,1)), slen1, slen2]	# reshape score from 1-d list to a ?-by-1 2-d numpy array
			_loss, _ = sess.run([self.tensorDict['loss'], train_step], feed_dict = {placeholder : feedData for placeholder, feedData in zip(self.placehodlers, feedDatas)})
			
			if (i % self.para.modelSavePeriod != 0):
				continue	
			# print train step info(different metric for different dataset) and save the model
			if (self.para.dataset == 'SICK'):
				# valid set
				s1, s2, score, slen1, slen2, idx = self.datas.getValidSet()
				sc = np.reshape(score, (-1, 1))
				feedDatas = [s1, s2, sc, slen1, slen2]
				_loss, _prob, _y, _merged, _prob_mse, _mse, _pr, _a1, _a2 = sess.run([self.tensorDict['loss'], self.tensorDict['prob'], self.tensorDict['y'], merged, self.tensorDict['prob_mse'], self.tensorDict['mse'], self.tensorDict['pearson_r'], self.tensorDict['sent1_annotation'], self.tensorDict['sent2_annotation']], feed_dict = {placeholder : feedData for placeholder,feedData in zip(self.placehodlers, feedDatas)})
				print 'valid set loss: \t' + str(_loss)
				print 'valid set prob_MSE: \t' + str(_prob_mse)
				print 'valid set score_MSE: \t' + str(_mse)
				print 'valid set pearson_r: \t', _pr
				print 'valid set spearman_rho: \t', utils.spearman_rho(_y, sc)			 
			
				# we also examine on test set for demonstration(slow down the training phase at the same time). This part may cause OOM(out of memory) if you use GPU
				s1, s2, score, slen1, slen2, idx = self.datas.getTestSet()
				sc = np.reshape(score,(-1,1))
				feedDatas = [s1, s2, sc, slen1, slen2]
				_loss, _prob, _y, _merged, _prob_mse, _mse, _pr = sess.run([self.tensorDict['loss'], self.tensorDict['prob'], self.tensorDict['y'], merged, self.tensorDict['prob_mse'], self.tensorDict['mse'], self.tensorDict['pearson_r']], feed_dict = {placeholder : feedData for placeholder, feedData in zip(self.placehodlers, feedDatas)})
				print 'test set loss: \t' + str(_loss)
				print 'test set prob_MSE: \t' + str(_prob_mse)
				print 'test set score_MSE: \t' + str(_mse)
				print 'test set pearson_r: \t', _pr
				print 'test set spearman_rho: \t', utils.spearman_rho(_y, sc)

			elif (self.para.dataset == 'WikiQA'):
				# valid set
				s1, s2, score, slen1, slen2, idx = self.datas.getValidSet()
				sc = np.reshape(score, (-1, 1))
				feedDatas = [s1, s2, sc, slen1, slen2]
				_loss, _prob_pos, _merged = sess.run([self.tensorDict['loss'], self.tensorDict['prob_of_positive'], merged], feed_dict = {placeholder : feedData for placeholder, feedData in zip(self.placehodlers, feedDatas)})
				MRR, MAP = self.datas.evaluateOn(_prob_pos, 'dev')
				print 'valid set loss: \t%f' % _loss
				print 'valid set MRR: \t%f' % MRR
				print 'valid set MAP: \t%f' % MAP

				# test set
				s1, s2, score, slen1, slen2, idx = self.datas.getTestSet()
				sc = np.reshape(score, (-1, 1))
				feedDatas = [s1, s2, sc, slen1, slen2]
				_loss, _prob_pos, _merged = sess.run([self.tensorDict['loss'], self.tensorDict['prob_of_positive'], merged], feed_dict = {placeholder : feedData  for placeholder, feedData in zip(self.placehodlers, feedDatas)})
				MRR, MAP = self.datas.evaluateOn(_prob_pos, 'test')
				print 'test set loss: \t%f' % _loss
				print 'test set MRR: \t%f' % MRR
				print 'test set MAP: \t%f' % MAP

			elif (self.para.dataset == 'LBA'):
				print 'train loss:\t%f' % _loss
				# save
				if not save_flag:
					save_flag = True
					dataPath = '../data/expLBA/fastText.txt'
					self.datas.save4FastText(dataPath)
				_, _prob_pos, _merged = sess.run([self.tensorDict['loss'], self.tensorDict['prob_of_positive'], merged], feed_dict = {placeholder : feedData for placeholder,feedData in zip(self.placehodlers, feedDatas)})			

			# save log
			writer.add_run_metadata(tf.RunMetadata(), 'itr:%d' % i)
			writer.add_summary(_merged, i)

			# save model
			if not tf.gfile.Exists(self.para.modelSaveDir):
				tf.gfile.MakeDirs(self.para.modelSaveDir)
			saver.save(sess, self.para.modelSavePath, global_step = i, write_meta_graph = False)
			print 'model saved at %s with global step of %d' % (self.para.modelSavePath, i) 

	def predict(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))	
		
		variable_to_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)	
		saver = tf.train.Saver(variable_to_save)
		saver.restore(sess, self.para.modelLoadPath)

		if (self.para.dataset == 'SICK'):
			s1, s2, score, slen1, slen2,idx = self.datas.getTestSet()
			sc = np.reshape(score, (-1, 1))
			feedDatas = [s1, s2, sc, slen1, slen2]
			_loss, _prob, _y, _prob_mse, _mse, _pr, _a1, _a2 = sess.run([self.tensorDict['loss'], self.tensorDict['prob'], self.tensorDict['y'], self.tensorDict['prob_mse'], self.tensorDict['mse'], self.tensorDict['pearson_r'], self.tensorDict['sent1_annotation'], self.tensorDict['sent2_annotation']], feed_dict = { placeholder: feedData for placeholder, feedData in zip(self.placehodlers, feedDatas)})
			print '=======================test phase========================'
			print 'test set loss: \t' + str(_loss)
			print 'test set prob_MSE: \t' + str(_prob_mse)
			print 'test set score_MSE: \t' + str(_mse)
			print 'test set pearson_r: \t', _pr
			print 'test set spearman_rho: \t', utils.spearman_rho(_y, sc)
		
			utils.analysisBatchMatrixDependency(_a1, slen1)  # measure of redundancy of annotation matrix

			if (self.para.modelType == 6):
				# inspect the annotation matrix
				for i in range(10):
					iid = random.randint(0,len(s1))
					sent1 = self.datas.displaySent(s1[iid] , slen1[iid])
					sent2 = self.datas.displaySent(s2[iid] , slen2[iid])
					annotation1 = np.squeeze(np.transpose(_a1[iid,:slen1[iid],:]))
					annotation2 = np.squeeze(np.transpose(_a2[iid,:slen2[iid],:]))
					utils.displayAttentionMat(sent1, annotation1, sent2, annotation2)

		elif (self.para.dataset == 'WikiQA'):
			# test set
			s1, s2, score, slen1, slen2, idx = self.datas.getTestSet()
			sc = np.reshape(score, (-1, 1))
			feedDatas = [s1, s2, sc, slen1, slen2]
			_loss, _prob_pos, _annotation = sess.run([self.tensorDict['loss'], self.tensorDict['prob_of_positive'], self.tensorDict['sent2_annotation']], feed_dict = {placeholder : feedData for placeholder, feedData in zip(self.placehodlers, feedDatas)})
			MRR, MAP = self.datas.evaluateOn(_prob_pos, 'test')
			print 'test set loss: \t%f' % _loss
			print 'test set MRR: \t%f' % MRR
			print 'test set MAP: \t%f' % MAP
			
			utils.analysisBatchMatrixDependency(_annotation, slen2)
	
			# randomly inspect some questions
			for i in range(10):
				iid = random.randint(1, 100)
				self.datas.displayQuestion(_prob_pos, iid, dataset = 'test')
				raw_input('Press Enter to continue...')

		elif (self.para.dataset == 'LBA'):
				#self.datas.inspectSentByLabel('test', 'Q')  # debug use
				s1, s2, score, slen1, slen2, evalSet_label, ref_label, L = self.datas.getEvalSet('both', label_set = 'all') # inspect all incorrect classificated sample with label Q
				
				sc = np.reshape(score, (-1, 1))
				labelMap = self.datas.digitLabel
				M = np.zeros((len(labelMap), len(labelMap)) , dtype = int)	# M[i][j]: the number of samples predicted to be category j while true label is i, original label are sorted by their lexicographical order
				for k in range(len(evalSet_label)):
					feedDatas = [s1[k * L:(k + 1) * L], s2[k * L:(k + 1) * L], sc[k * L:(k + 1) * L], slen1[k * L:(k + 1) * L], slen2[k * L:(k + 1) * L]]
					prob_list = []
					# for each samples in evaluation set, we have L sentences pairs.
					# we split these L records in small batches to process since L is too large.
					for batch_idx in range(int(L / 500 + 1)):
						batch_datas = [fd[batch_idx * 500 : (batch_idx + 1) * 500] for fd in feedDatas]
						_, _prob_pos = sess.run([self.tensorDict['loss'], self.tensorDict['prob_of_positive']], feed_dict = {placeholder : feedData for placeholder, feedData in zip(self.placehodlers, batch_datas)})
						prob_list.append(_prob_pos)
					_prob_pos = np.concatenate(prob_list, axis = 0)
					pred, rk = utils.vote(ref_label[k * L:(k + 1) * L], _prob_pos, top_k = 15)
					true_label = labelMap[evalSet_label[k]]
					pred_label = labelMap[pred]
					M[true_label][pred_label] += 1
					if (k % (len(evalSet_label) / 100) == 0):
						print 'progress: %f' % (1.0 * k / len(evalSet_label))
						print M
				print 'confusion matrix:\t '
				print np.array2string(M)			
				print 'over all accuracy: %f' % (1.0 * np.sum(np.diag(M)) / np.sum(M))
				
if __name__ == '__main__':
	print 'tensorflow version in use:  ' + tf.__version__
	
	parser = argparse.ArgumentParser(description='train model')
	parser.add_argument('-config', type = str, nargs='+', help = 'path to the config file')
	args = parser.parse_args()

	config = args.config[0]
	para = confrd.Parameters(config)
	para.printAll()
	if (para.use_cpu == 1):
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	tf.logging.set_verbosity(tf.logging.WARN)

	model = modelManeger(para)
	if (para.TrainFlag):
		model.train()
	if (para.PredictFlag):
		model.predict()

