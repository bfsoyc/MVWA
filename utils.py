'''
Copyright @2018 PKU Calis.
All rights reserved.

This python script defines many useful utilities.
'''

import numpy as np
import matplotlib.pyplot as plt

# find top k cloest vector from vocabulary matrix
def cloestVector(vec, emb, topK = 10, metric = 'Eular'):
	d = []	
	for i in range(emb.shape[0]):
		if (metric == 'Eular'):
			d.append(np.linalg.norm(vec - emb[i, :]))
	idx = sorted(range(len(d)), key = lambda k:d[k])
	return idx[:topK]

# majority vote
def vote(label, score, top_k = 10):
	assert len(label) == len(score)
	idx = sorted(range(len(score)), key = lambda k:score[k], reverse = True)
	top_list = [label[i] for i in idx[:top_k]]
	winner = max(set(top_list), key = top_list.count)
	return winner, idx[:top_k]  # return the winner and index of top k votes

# retrive the truncated index based on Chinese Library Classification
# CLC is a hierachical classification, this function extract the major category from its full index.
def retriveMajorCLC(CLCIndex, level):
	CLCIndex = CLCIndex.upper()
	if not level in [1,2]:
		raise Exception('one 1 and 2 are supported level')
	MajorIndex = 'ABCDEFGHIJKNOPQRSTUVXZ'
	if not CLCIndex[0] in MajorIndex:
		raise Exception('invalid CLC index %s' % CLCIndex)
	return CLCIndex[:level]

# compute and return the eigen values sorted by descendant order and the corresponding eigen vectors
def eigenDecomposition(A):
	evals, evecs = np.linalg.eig(A)
	energy = evals**2
	idx = np.argsort(-energy)
	return evals[idx], evecs[:, idx]

# a naive analysis of row dependency of a batch of matrix.
def analysisBatchMatrixDependency(batchMat, batchLen):
	maxLen = max(batchLen)
	Majority_k = [[] for i in range(maxLen+1)]
	energy_at_second_last = [[] for i in range(maxLen+1)]

	bz = batchMat.shape[0]
	for i in range(bz):
		_mat = batchMat[i, :]
		length = batchLen[i]
		U,Sigma,V = np.linalg.svd(_mat)	
		S = np.zeros(max(length, Sigma.shape[0]))	
		S[:Sigma.shape[0]] = Sigma
		S /= sum(S)
		first_idx_greater_than_95 = length
		for j in range(1,length):
			S[j] += S[j-1]
			if (first_idx_greater_than_95 == length and S[j] > 0.95):
				first_idx_greater_than_95 = j
		Majority_k[length].append(first_idx_greater_than_95)
		energy_at_second_last[length].append(S[length - 2])
	
	print 'length\tcount\tMajor_at_k\tenergy[len-1]'
	for i in range(2, maxLen+1):
		if (len(Majority_k[i]) == 0):
			continue
		print '%d\t%d\t%f\t%f' % (i, len(Majority_k[i]), np.mean(Majority_k[i]) + 1, np.mean(energy_at_second_last[i]))
		

# compute the average precision(AP) of the given rank list
# relevant is a binary mask where entity with value 1 represent a positive sample
# AP:= summation{ precision(k)*(recall(k)-recall(k-1)) | k = 1,2,3... }
def computeAP(score, relevant):
	assert isinstance(relevant[0], int)
	idx = sorted(range(len(score)), key = lambda k:score[k], reverse = True)
	hit = 0.0
	AP = 0.0
	for k,i in enumerate(idx):
		if (relevant[i] == 1):
			hit += 1
			precision_at_k = hit / (k+1)
			AP += precision_at_k
	AP /= sum(relevant)
	return AP

# compute the reciprocal rank(RR) for the given rank list
# relevant is a binary mask where entity with value 1 represent a positive sample
def computeRR(score, relevant):
	assert isinstance(relevant[0], int)
	idx = sorted(range(len(score)), key = lambda k:score[k], reverse = True)
	try:
		r = next(x for x in range(len(idx)) if relevant[idx[x]] == 1)  # find the first answer with score 1
	except StopIteration:
		'Invalid argument: relevant. No entity in relevant is one'
	RR = 1.0 / (r + 1)
	return RR

def spearman_rho(predict, label, approximate = False):
	if (approximate):
		# use the pupular formula
		# The formula only holds when both predict and label are distinct set, which usually is not the case of label set
		n = len(predict)
		_, rank_p = np.unique(predict, return_inverse = True)
		_, rank_l = np.unique(label, return_inverse = True)
		d = 0
		for rx, ry in zip(rank_p + 1, rank_l + 1):
			d = d + (rx - ry)**2
		rho = 1 - 6.0 * d / n / (n*n - 1)
	else:  # computed by definition
		EPS = 1e-6
		p = predict.flatten()
		l = label.flatten()
		args_p = p.argsort()
		args_l = l.argsort()
	
		n = len(args_p)
		front = 0
		rnk = 1
		rk_p = np.zeros(shape=(n,1))
		while front < n:
			tail = front
			while(tail < n and abs(p[args_p[tail]] - p[args_p[front]]) < EPS):
				tail = tail + 1
			r = rnk + 0.5 * (tail - 1 - front)
			for j in range(front,tail):
				rk_p[args_p[j]] = r
			front = tail
			rnk = front + 1

		front = 0
		rnk = 1
		rk_l = np.zeros(shape=(n, 1))
		while front < n:
			tail = front
			while(tail < n and abs(l[args_l[tail]] - l[args_l[front]]) < EPS):
				tail = tail + 1
			r = rnk + 0.5 * (tail - 1 - front)
			for j in range(front, tail):
				rk_l[args_l[j]] = r
			front = tail
			rnk = front + 1
	
		corrArr = np.corrcoef(np.column_stack((rk_p, rk_l)) , rowvar = 0)
		rho = corrArr[0,1]
	return rho

# Uasge request :
# 1 each row in w1 and w2 is an aspect of attention
# 2 both sen1 and sen2 is a list of words
def displayAttentionMat(sen1, w1, sen2, w2):
	_figArr, axArr = plt.subplots(nrows = 1, ncols = 2, figsize = (20,20), num = 'press any key to resume' )
	fontDict = {'fontsize':16}

	i_step = int(256 / w1.shape[0])
	j_step = int(256 / w1.shape[1])
	w,h = (i_step * w1.shape[0], j_step * w1.shape[1])
	img = np.zeros(shape = [w, h,3], dtype = np.float32)
	maxv = np.max(w1)
	for i in range(w1.shape[0]):
		for j in range(w1.shape[1]):
			img[i * i_step : (i + 1) * i_step, j * j_step : (j + 1) * j_step, : ] = 64 + w1[i, j] / maxv * 128 # rescale the grey image such that each pixel value is in range [64,196]
		axArr[0].imshow(img.astype(np.uint8))
		axArr[0].set_xticks([j * j_step + j_step / 2 for j in range(w1.shape[1])])
		axArr[0].set_xticklabels(sen1, fontdict = fontDict, rotation = 40)  
		axArr[0].get_xaxis().set_ticks_position('top')
		axArr[0].get_yaxis().set_visible(False)
		axArr[0].tick_params(axis = 'x', length = 0)  # remove the ticks line on axis 

	# do the same for sen2
	i_step = int(256 / w2.shape[0])
	j_step = int(256 / w2.shape[1])
	w, h = (i_step * w2.shape[0], j_step * w2.shape[1])
	img = np.zeros(shape = [w, h, 3], dtype = np.float32)
	maxv = np.max(w2)
	for i in range(w2.shape[0]):
		for j in range(w2.shape[1]):
			img[i * i_step : (i + 1) * i_step, j * j_step : (j + 1) * j_step, : ] = 64 + w2[i, j] / maxv * 128 
		axArr[1].imshow(img.astype(np.uint8))
		axArr[1].set_xticks([j * j_step + j_step / 2 for j in range(w2.shape[1])])
		axArr[1].set_xticklabels(sen2, fontdict = fontDict, rotation = 40)  
		axArr[1].get_xaxis().set_ticks_position('top')
		axArr[1].get_yaxis().set_visible(False)

	axArr[1].tick_params(axis = 'x', length = 0)  
	plt.ion()	# trun the interactive mode on to avoid plt.show() blocking everything.
	plt.show()
	plt.waitforbuttonpress(0)
	plt.close('all')

# for data transform
# sentence normalization: transform capital letter into lowcase letter and remove all charactors out of alphabet (a-z) except for some parentheses.
def sentenceNorm(sent):
	sent = sent.lower()
	l = len(sent)
	s = ''
	for i in range(l):
		if (sent[i] == '\'' or sent[i] == '\\' or sent[i] == '/'):
			if (i+1 < l and sent[i+1] != ' '):
				s = s + ' '
		elif (sent[i] == ':'):
			if (i+1 < l and sent[i+1] != ' '):
				s = s + ' '
		elif (sent[i] == ',' or sent[i]=='.'):
			continue
		else:
			s = s + sent[i]
	return s

