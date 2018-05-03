import numpy as np
import matplotlib.pyplot as plt

def spearman_rho( predict, label, approximate = False ):
	if (approximate):
		# use the pupular formula
		# this is only hold when both predict and label are distinct set, which usually is not the case of label 
		n = len(predict)
		_, rank_p = np.unique(predict, return_inverse = True)
		_, rank_l = np.unique(label, return_inverse = True)
		d = 0
		for rx,ry in zip( rank_p+1, rank_l+1 ):
			d = d + (rx-ry)**2
		rho = 1 - 6.0 * d / n / (n*n - 1)
	else:
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
		rho = corrArr[0,1]
	return rho

# assume :
# 1 each row in w1 is an aspect of attention
# 2 sen1 is a list of words
def displayAttentionMat( sen1, w1, sen2, w2 ):
  _figArr, axArr = plt.subplots( nrows = 1, ncols = 2, figsize = (20,20)  )
  fontDict = { 'fontsize':16}
  
  i_step = int(256/w1.shape[0])
  j_step = int(256/w1.shape[1])
  w,h = (i_step * w1.shape[0], j_step * w1.shape[1])
  img = np.zeros( shape = [w,h,3], dtype=np.float32)
  maxv = np.max(w1)
  for i in range(w1.shape[0]):
    for j in range(w1.shape[1]):
      img[i*i_step:(i+1)*i_step, j*j_step:(j+1)*j_step, : ] = 64 + w1[i,j]/maxv*128 # rescale the grey image such that each pixel value is in range [64,196]
  axArr[0].imshow(img.astype(np.uint8))
  axArr[0].set_xticks( [j*j_step+j_step/2 for j in range(w1.shape[1])] )
  axArr[0].set_xticklabels(sen1, fontdict = fontDict, rotation = 40)  
  axArr[0].get_xaxis().set_ticks_position('top')
  axArr[0].get_yaxis().set_visible(False)
  axArr[0].tick_params( axis = 'x', length = 0)  # remove the ticks line on axis 
  
  # do the same for sen2
  i_step = int(256/w2.shape[0])
  j_step = int(256/w2.shape[1])
  w,h = (i_step * w2.shape[0], j_step * w2.shape[1])
  img = np.zeros( shape = [w,h,3], dtype=np.float32)
  maxv = np.max(w2)
  for i in range(w2.shape[0]):
    for j in range(w2.shape[1]):
      img[i*i_step:(i+1)*i_step, j*j_step:(j+1)*j_step, : ] = 64 + w2[i,j]/maxv*128 
  axArr[1].imshow(img.astype(np.uint8))
  axArr[1].set_xticks( [j*j_step+j_step/2 for j in range(w2.shape[1])] )
  axArr[1].set_xticklabels(sen2, fontdict = fontDict, rotation = 40)  
  axArr[1].get_xaxis().set_ticks_position('top')
  axArr[1].get_yaxis().set_visible(False)
  
  axArr[1].tick_params( axis = 'x', length = 0)  
  plt.show()

# for data transform
# sentence normalization: transform capital letter into lowcase letter and remove all charactors out of alphabet ( a-z )
def sentenceNorm( sent ):
	sent = sent.lower()
	l = len( sent )
	s = ''
	for i in range(l):
		if (sent[i] == '\'' or sent[i] == '\\' or sent[i] == '/'):
			if (i+1 < l and sent[i+1] != ' '):
				s = s + ' '
		if (sent[i] == ':'):
			if (i+1 < l and sent[i+1] != ' '):
				s = s + ' '
		elif (sent[i] == ',' or sent[i]=='.'):
			continue
		else:
			s = s + sent[i]
	return s

