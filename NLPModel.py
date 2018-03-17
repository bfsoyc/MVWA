import tensorflow as tf

def rnnModel( maxLen , para ):
	hiddenSize = para.cellSize
	
	lens1 = tf.placeholder( dtype = tf.float32, shape = [None] )
	lens2 = tf.placeholder( dtype = tf.float32, shape = [None] )
	sent1 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	sent2 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	score = tf.placeholder( dtype = tf.float32, shape = [None, 1 ] )
	

	with tf.variable_scope( "embeddingLayer" ):
		tf.get_variable_scope().reuse_variables()
		embMat = tf.get_variable( "embedding" )
	embeddingSize = embMat.get_shape().as_list()[1]
	wordEmb1 = tf.nn.embedding_lookup( embMat, sent1 )
	wordEmb2 = tf.nn.embedding_lookup( embMat, sent2 )


	if( para.activationType == 1 ):
		activeFuc = tf.tanh
	elif( para.activationType == 2 ):
  		activeFuc = tf.identity

	if( para.rnnCellType == 1 ):
		cell = tf.contrib.rnn.BasicLSTMCell( hiddenSize , activation = activeFuc, forget_bias = para.forgetBias)
	elif( para.rnnCellType == 2 ):
		cell = tf.contrib.rnn.GRUCell( hiddenSize )
	#cell = tf.contrib.rnn.DropoutWrapper( cell=cell, output_keep_prob = para.keepProb )
	outputs1, states1 = tf.nn.dynamic_rnn( cell, wordEmb1, sequence_length = lens1, dtype=tf.float32 )	# states is a tuple with final outputs(.h ) and cellstate (.c)
	outputs2, states2 = tf.nn.dynamic_rnn( cell, wordEmb2, sequence_length = lens2, dtype=tf.float32 )
	if( para.rnnCellType == 1 ):
		feas1 = [states1.c]
		feas2 = [states2.c]
	elif( para.rnnCellType == 2 ):
		feas1 = [states1]
		feas2 = [states2]


	if( para.similarityMetric == 1 ):
		# In "softmax" metric, we use the final state of LSTM to predict the distribution of similarity 	
		feaVec, w_a, w_d = __feaTransform( feas1, feas2, outputSize = hiddenSize )

		# softmax layer
		w = tf.get_variable( 'softmax_weight', initializer = tf.truncated_normal( shape =  [hiddenSize, 5], stddev = 0.1 ) )
		b = tf.get_variable( 'softmax_bias' , initializer = tf.truncated_normal( shape =  [5], stddev = 0.1 ) )
		logits = tf.matmul( feaVec, w ) + b 
		prob = tf.nn.softmax( logits )

		r = tf.constant( range(1,6), shape = [5,1] , dtype = tf.float32 )
		y =  tf.matmul( prob, r ) 
		label = __scoreLabel2p( score )	

		
		wPenalty = __applyL2Norm()
		#wPenalty = L2Strength * ( tf.reduce_sum( tf.square( w )) + tf.reduce_sum( tf.square(w_a) ) + tf.reduce_sum( tf.square(w_d)) )	# weight decay penalty
		#-------------------
		loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = label, logits = logits, name = 'cross_entropy_loss' ) ) + wPenalty
		pearson_r = __pearson_r( y, score )

		# add summary for tensorboard visulization
		prob_mse = tf.reduce_mean(   tf.reduce_sum( (prob-label)**2 , axis = 1) )	
		mse = tf.reduce_mean( tf.square(y - score) )
		tf.summary.scalar( 'probability_MSE' , prob_mse )
		tf.summary.scalar( 'similarity_MSE' , mse )
		tf.summary.scalar( 'pearson_r', pearson_r )
		tf.summary.scalar( 'loss' , loss )

	elif( para.similarityMetric == 2 ):
		# In "Manhattan" metric, we derive the estimation of similarity directly from the Mahhattan distance of the final states of LSTM 		
		prob = tf.exp( -tf.reduce_sum( tf.abs( feas1[0] - feas2[0] ), axis = 1 , keep_dims = True) )
		a = tf.get_variable( 'regression_coef_a', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		b = tf.get_variable( 'regression_coef_b', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		c = tf.get_variable( 'regression_coef_c', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		prob = a*prob**2 + b*prob + c
		label = __rescaleLabel( score )
		wPenalty = __applyL2Norm()
		loss = tf.losses.mean_squared_error( labels = label, predictions = prob ) + wPenalty

		y = __rescaleLabel( prob , True)
		pearson_r = __pearson_r( y, score )
		prob_mse = mse = loss*16	# advanced computation

		# add summary for tensorboard visulization
		tf.summary.scalar( 'loss', loss )
		tf.summary.scalar( 'similarity_MSE' , mse )


	return [sent1, sent2, score, lens1, lens2], loss, prob, y, prob_mse, mse, outputs1, outputs2, pearson_r


def selfRnnModel( maxLen , para ):
	hiddenSize = para.cellSize
	
	lens1 = tf.placeholder( dtype = tf.int32, shape = [None] )
	lens2 = tf.placeholder( dtype = tf.int32, shape = [None] )
	sent1 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	sent2 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	score = tf.placeholder( dtype = tf.float32, shape = [None, 1 ] )
	

	with tf.variable_scope( "embeddingLayer" ):
		tf.get_variable_scope().reuse_variables()
		embMat = tf.get_variable( "embedding" )
	embeddingSize = embMat.get_shape().as_list()[1]
	wordEmb1 = tf.nn.embedding_lookup( embMat, sent1 )
	wordEmb2 = tf.nn.embedding_lookup( embMat, sent2 )
	
	# transform to time-major data
	t_wordEmb1 = tf.transpose( wordEmb1, perm = [1,0,2] )
	t_wordEmb2 = tf.transpose( wordEmb2, perm = [1,0,2] )
	
	def  _dynamic_rnn( cell, length, inputs ):
		# the implement of dynamic_rnn:
		# min_seq_length and max_seq_length is the minimum and maximum length of sequences within a batch 
		# we loop over the time 
		#	if( time <= min_seq_length ):
		#		calculate the next state from previous state for all samples in the batchSize
		# 		( just for the efficience of computation over batch? )		
		#	else:
		#		calculate the next state from previous state for all samples in the batchSize and set to zero_state for those which have done.  
		#
		# noted that we have deprecated using term 'state' and 'output' to represent the cell state and the cell hypothesis output
		# now we use a tuple, named state, to capsule all these array:  state = tuple( h = output, c = cell_state )   
		min_seq_length = tf.reduce_min( length )
		max_seq_length = tf.reduce_max( length )
		input_shape = tf.shape(inputs)	# tf.shape() return a tf.Tensor
		input_TShape = inputs.get_shape()	# this method return a tf.TensorShape, which is a tuple of "Dimension".		

		maxTime = input_shape[0]
		batchSize = input_shape[1]

		# create zero states
		zero_state = cell.zero_state( batchSize, dtype = tf.float32 )

		# create tensor to store all temporary variable while processing the RNN
		shp = [None, cell.output_size]	# element shape
		# create an array of tensor 
		h_ta = tf.TensorArray( dtype = tf.float32, size = max_seq_length, element_shape = shp )
		c_ta = tf.TensorArray( dtype = tf.float32, size = max_seq_length, element_shape = shp )	

		def _time_step( time, state, h_ta, c_ta ):
			''' 
				take a single time step of the dynamic rnn, used as body for tf.while_loop()
			
			'''
			input_t = tf.slice( inputs, begin = [ time, 0,0 ], size = [1,-1,-1] )
			input_t = tf.squeeze( input_t, axis = 0)
			call_cell = lambda: cell( inputs = input_t, state = state )
			( _, new_state ) = call_cell()
			# broadcasting select to determine which value should get the previous state & zero output , 
			# and which values should get a calculated state & output.
			copy_cond = (time >= length )	
			new_h = tf.where( copy_cond, zero_state.h, new_state.h )	# zero output or new calculated output.
			new_c = tf.where( copy_cond, state.c, new_state.c )				# privious state or new calculated state.

			# store cell output and state of all timestamp
			h_ta = h_ta.write( time, new_h )
			c_ta = c_ta.write( time, new_c )

			return ( time+1, tf.contrib.rnn.LSTMStateTuple( c=new_c, h=new_h ), h_ta, c_ta )

		#
		time = tf.constant( 0, dtype = tf.int32, name = 'time' )
		_,  final_state, h_ta, c_ta = tf.while_loop( cond = lambda time, *_ : time < max_seq_length,
														body = _time_step, loop_vars = ( time, zero_state, h_ta, c_ta ) )

		all_H = h_ta.stack()
		all_C = c_ta.stack()
		return final_state, all_H, all_C

	if( para.activationType == 1 ):
		activeFuc = tf.tanh
	elif( para.activationType == 2 ):
  		activeFuc = tf.identity

	if( para.rnnCellType == 1 ):
		cell = tf.contrib.rnn.BasicLSTMCell( hiddenSize , activation = activeFuc, forget_bias = para.forgetBias)
	elif( para.rnnCellType == 2 ):
		cell = tf.contrib.rnn.GRUCell( hiddenSize )
	states1, ah1, ac1 = _dynamic_rnn( cell, lens1, t_wordEmb1 )
	states2, ah2, ac2 = _dynamic_rnn( cell, lens2, t_wordEmb2 )

	feas1 = [states1.c] + __extractFeaVec( tf.transpose( ac1, [1,0,2] ) , lens1)
	feas2 = [states2.c] + __extractFeaVec( tf.transpose( ac2, [1,0,2] ) , lens2)
	
	if( para.similarityMetric == 1 ):
		# In "softmax" metric, we use the final state of LSTM to predict the distribution of similarity 	
		feaVec, w_a, w_d = __feaTransform( feas1, feas2, outputSize = hiddenSize )

		# softmax layer
		w = tf.get_variable( 'softmax_weight', initializer = tf.truncated_normal( shape =  [hiddenSize, 5], stddev = 0.1 ) )
		b = tf.get_variable( 'softmax_bias' , initializer = tf.truncated_normal( shape =  [5], stddev = 0.1 ) )
		logits = tf.matmul( feaVec, w ) + b 
		prob = tf.nn.softmax( logits )

		r = tf.constant( range(1,6), shape = [5,1] , dtype = tf.float32 )
		y =  tf.matmul( prob, r ) 
		label = __scoreLabel2p( score )	

		
		wPenalty = __applyL2Norm()
		#wPenalty = L2Strength * ( tf.reduce_sum( tf.square( w )) + tf.reduce_sum( tf.square(w_a) ) + tf.reduce_sum( tf.square(w_d)) )	# weight decay penalty
		#-------------------
		loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = label, logits = logits, name = 'cross_entropy_loss' ) ) + wPenalty
		pearson_r = __pearson_r( y, score )

		# add summary for tensorboard visulization
		prob_mse = tf.reduce_mean(   tf.reduce_sum( (prob-label)**2 , axis = 1) )	
		mse = tf.reduce_mean( tf.square(y - score) )
		tf.summary.scalar( 'probability_MSE' , prob_mse )
		tf.summary.scalar( 'similarity_MSE' , mse )
		tf.summary.scalar( 'pearson_r', pearson_r )
		tf.summary.scalar( 'loss' , loss )

	elif( para.similarityMetric == 2 ):
		# In "Manhattan" metric, we derive the estimation of similarity directly from the Mahhattan distance of the final states of LSTM 		
		prob = tf.exp( -tf.reduce_sum( tf.abs( feas1[0] - feas2[0] ), axis = 1 , keep_dims = True) )
		a = tf.get_variable( 'regression_coef_a', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		b = tf.get_variable( 'regression_coef_b', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		c = tf.get_variable( 'regression_coef_c', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		prob = a*prob**2 + b*prob + c
		label = __rescaleLabel( score )
		wPenalty = __applyL2Norm()
		loss = tf.losses.mean_squared_error( labels = label, predictions = prob ) + wPenalty

		y = __rescaleLabel( prob , True)
		pearson_r = __pearson_r( y, score )
		prob_mse = mse = loss*16	# advanced computation

		# add summary for tensorboard visulization
		tf.summary.scalar( 'loss', loss )
		tf.summary.scalar( 'similarity_MSE' , mse )


	outputs1 = tf.constant(0,dtype=tf.int32)
	outputs2 = tf.constant(0,dtype=tf.int32)	# useless
	return [sent1, sent2, score, lens1, lens2], loss, prob, y, prob_mse, mse, outputs1, outputs2, pearson_r, ah1, ac1, ah2, ac2

def selfAttentionRnnModel( maxLen , para ):
	hiddenSize = para.cellSize
	
	lens1 = tf.placeholder( dtype = tf.int32, shape = [None] )
	lens2 = tf.placeholder( dtype = tf.int32, shape = [None] )
	sent1 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	sent2 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	score = tf.placeholder( dtype = tf.float32, shape = [None, 1 ] )
	

	with tf.variable_scope( "embeddingLayer" ):
		tf.get_variable_scope().reuse_variables()
		embMat = tf.get_variable( "embedding" )
	embeddingSize = embMat.get_shape().as_list()[1]
	wordEmb1 = tf.nn.embedding_lookup( embMat, sent1 )
	wordEmb2 = tf.nn.embedding_lookup( embMat, sent2 )
	
	# transform to time-major data
	t_wordEmb1 = tf.transpose( wordEmb1, perm = [1,0,2] )
	t_wordEmb2 = tf.transpose( wordEmb2, perm = [1,0,2] )
	
	

	def  _self_attention_rnn( cell, length, inputs ):
		# the implement of dynamic_rnn:
		# min_seq_length and max_seq_length is the minimum and maximum length of sequences within a batch 
		# we loop over the time 
		#	if( time <= min_seq_length ):
		#		calculate the next state from previous state for all samples in the batchSize
		# 		( just for the efficience of computation over batch? )		
		#	else:
		#		calculate the next state from previous state for all samples in the batchSize and set to zero_state for those which have done.  
		#
		# noted that we have deprecated using term 'state' and 'output' to represent the cell state and the cell hypothesis output
		# now we use a tuple, named state, to capsule all these array:  state = tuple( h = output, c = cell_state )   
		min_seq_length = tf.reduce_min( length )
		max_seq_length = tf.reduce_max( length )
		input_shape = tf.shape(inputs)	# tf.shape() return a tf.Tensor
		input_TShape = inputs.get_shape()	# this method return a tf.TensorShape, which is a tuple of "Dimension".		

		maxTime = input_shape[0]
		batchSize = input_shape[1]

		# create zero states
		zero_state = cell.zero_state( batchSize, dtype = tf.float32 )
		INF = 1e6
		mask_weight = -INF * tf.ones( [batchSize, para.attention_aspect] , dtype=tf.float32 )

		# create attention weights
		with tf.variable_scope( 'attention_matrix', reuse = tf.AUTO_REUSE ):
			attention_W = tf.get_variable( 'attention_weights', initializer = tf.truncated_normal( shape =  [cell.output_size, para.attention_aspect], stddev = 0.1 ) ) 

		# create tensor to store all temporary variable while processing the RNN
		shp = [None, cell.output_size]	# element shape
		# create an array of tensor 
		h_ta = tf.TensorArray( dtype = tf.float32, size = max_seq_length, element_shape = shp )
		c_ta = tf.TensorArray( dtype = tf.float32, size = max_seq_length, element_shape = shp )	
		w_ta = tf.TensorArray( dtype = tf.float32, size = max_seq_length, element_shape = [None, para.attention_aspect] )		

		def _time_step( time, state, h_ta, c_ta , w_ta):
			''' 
				take a single time step of the dynamic rnn, used as body for tf.while_loop()
			
			'''
			input_t = tf.slice( inputs, begin = [ time, 0,0 ], size = [1,-1,-1] )
			input_t = tf.squeeze( input_t, axis = 0)
			call_cell = lambda: cell( inputs = input_t, state = state )
			( _, new_state ) = call_cell()

			# broadcasting select to determine which value should get the previous state & zero output , 
			# and which values should get a calculated state & output.
			copy_cond = (time >= length )	
			new_h = tf.where( copy_cond, zero_state.h, new_state.h )	# zero output or new calculated output.
			new_c = tf.where( copy_cond, state.c, new_state.c )				# privious state or new calculated state.

			weights = tf.matmul( new_state.h, attention_W )
			a_weight = tf.where( copy_cond, mask_weight, weights )
	
			# store cell output and state of all timestamp
			h_ta = h_ta.write( time, new_h )
			c_ta = c_ta.write( time, new_c )
			w_ta = w_ta.write( time, a_weight )

			return ( time+1, tf.contrib.rnn.LSTMStateTuple( c=new_c, h=new_h ), h_ta, c_ta , w_ta )

		#
		time = tf.constant( 0, dtype = tf.int32, name = 'time' )
		_,  final_state, h_ta, c_ta, w_ta = tf.while_loop( cond = lambda time, *_ : time < max_seq_length,
														body = _time_step, loop_vars = ( time, zero_state, h_ta, c_ta, w_ta ) )

		all_H = h_ta.stack()
		all_C = c_ta.stack()
		Ws = w_ta.stack()
		return final_state, all_H, all_C, Ws

	if( para.activationType == 1 ):
		activeFuc = tf.tanh
	elif( para.activationType == 2 ):
  		activeFuc = tf.identity

	if( para.rnnCellType == 1 ):
		cell = tf.contrib.rnn.BasicLSTMCell( hiddenSize , activation = activeFuc, forget_bias = para.forgetBias)
	elif( para.rnnCellType == 2 ):
		cell = tf.contrib.rnn.GRUCell( hiddenSize )
	states1, ah1, ac1, ws1 = _self_attention_rnn( cell, lens1, t_wordEmb1 )
	states2, ah2, ac2, ws2 = _self_attention_rnn( cell, lens2, t_wordEmb2 )

	# applying attention weight to calculate the weighted sum of cell states
	norm_w1 = tf.nn.softmax( ws1, axis = 0 )
	norm_w2 = tf.nn.softmax( ws2, axis = 0 )
	ext_wordEmb1 = tf.tile( tf.expand_dims( ac1, axis = -1 ), multiples = [1,1,1,para.attention_aspect] )
	ext_wordEmb2 = tf.tile( tf.expand_dims( ac2, axis = -1 ), multiples = [1,1,1,para.attention_aspect] )
	ext_w1 = tf.tile( tf.expand_dims( norm_w1, axis = 2 ), multiples = [1,1,cell.output_size,1] )
	ext_w2 = tf.tile( tf.expand_dims( norm_w2, axis = 2 ), multiples = [1,1,cell.output_size,1] )
	feas1 = tf.unstack( tf.reduce_sum( ext_wordEmb1 * ext_w1, axis = 0 ) , axis = -1 ) 
	feas2 = tf.unstack( tf.reduce_sum( ext_wordEmb2 * ext_w2, axis = 0 ) , axis = -1 )
	
	if( para.similarityMetric == 1 ):
		# In "softmax" metric, we use the final state of LSTM to predict the distribution of similarity 	
		feaVec, w_a, w_d = __feaTransform( feas1, feas2, outputSize = hiddenSize )

		# softmax layer
		w = tf.get_variable( 'softmax_weight', initializer = tf.truncated_normal( shape =  [hiddenSize, 5], stddev = 0.1 ) )
		b = tf.get_variable( 'softmax_bias' , initializer = tf.truncated_normal( shape =  [5], stddev = 0.1 ) )
		logits = tf.matmul( feaVec, w ) + b 
		prob = tf.nn.softmax( logits )

		r = tf.constant( range(1,6), shape = [5,1] , dtype = tf.float32 )
		y =  tf.matmul( prob, r ) 
		label = __scoreLabel2p( score )	

		
		wPenalty = __applyL2Norm()
		#wPenalty = L2Strength * ( tf.reduce_sum( tf.square( w )) + tf.reduce_sum( tf.square(w_a) ) + tf.reduce_sum( tf.square(w_d)) )	# weight decay penalty
		#-------------------
		loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = label, logits = logits, name = 'cross_entropy_loss' ) ) + wPenalty
		pearson_r = __pearson_r( y, score )

		# add summary for tensorboard visulization
		prob_mse = tf.reduce_mean(   tf.reduce_sum( (prob-label)**2 , axis = 1) )	
		mse = tf.reduce_mean( tf.square(y - score) )
		tf.summary.scalar( 'probability_MSE' , prob_mse )
		tf.summary.scalar( 'similarity_MSE' , mse )
		tf.summary.scalar( 'pearson_r', pearson_r )
		tf.summary.scalar( 'loss' , loss )

	elif( para.similarityMetric == 2 ):
		# In "Manhattan" metric, we derive the estimation of similarity directly from the Mahhattan distance of the final states of LSTM 		
		prob = tf.exp( -tf.reduce_sum( tf.abs( feas1[0] - feas2[0] ), axis = 1 , keep_dims = True) )
		a = tf.get_variable( 'regression_coef_a', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		b = tf.get_variable( 'regression_coef_b', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		c = tf.get_variable( 'regression_coef_c', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		prob = a*prob**2 + b*prob + c
		label = __rescaleLabel( score )
		wPenalty = __applyL2Norm()
		loss = tf.losses.mean_squared_error( labels = label, predictions = prob ) + wPenalty

		y = __rescaleLabel( prob , True)
		pearson_r = __pearson_r( y, score )
		prob_mse = mse = loss*16	# advanced computation

		# add summary for tensorboard visulization
		tf.summary.scalar( 'loss', loss )
		tf.summary.scalar( 'similarity_MSE' , mse )


	outputs1 = tf.constant(0,dtype=tf.int32)
	outputs2 = tf.constant(0,dtype=tf.int32)	# useless
	return [sent1, sent2, score, lens1, lens2], loss, prob, y, prob_mse, mse, outputs1, outputs2, pearson_r, ac2, norm_w2, feas2[1]

def gridRnnModel( maxLen , para ):
	hiddenSize = para.cellSize
	
	lens1 = tf.placeholder( dtype = tf.int32, shape = [None] )
	lens2 = tf.placeholder( dtype = tf.int32, shape = [None] )
	sent1 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	sent2 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	score = tf.placeholder( dtype = tf.float32, shape = [None, 1 ] )
	

	with tf.variable_scope( "embeddingLayer" ):
		tf.get_variable_scope().reuse_variables()
		embMat = tf.get_variable( "embedding" )
	embeddingSize = embMat.get_shape().as_list()[1]
	wordEmb1 = tf.nn.embedding_lookup( embMat, sent1 )
	wordEmb2 = tf.nn.embedding_lookup( embMat, sent2 )
	
	# transform to time-major data
	t_wordEmb1 = tf.transpose( wordEmb1, perm = [1,0,2] )
	t_wordEmb2 = tf.transpose( wordEmb2, perm = [1,0,2] )
	
	def _2dRNN( cell, s1, s2, lens1, lens2 ):
		cell2 = tf.contrib.rnn.BasicLSTMCell( num_units = cell._num_units, activation = cell._activation, forget_bias = cell._forget_bias)
		
		dim1_maxlen = tf.reduce_max( lens1 )
		dim2_maxlen = tf.reduce_max( lens2 ) 
		
		input_shape = tf.shape(s1)	# tf.shape() return a tf.Tensor	
		batchSize = input_shape[1]
		# create zero states
		zero_state1 = cell.zero_state( batchSize, dtype = tf.float32 )
		zero_state2 = cell.zero_state( batchSize, dtype = tf.float32 )


		'''
		image the total network as a 2 dimension grid network
		the grid at postion (i,j) encode the relatedness of string pair of s1[:i] and s2[:j]
		the cell state of grid at (i,j) with i > len1 ans j <= len2 should be same as the grid on its left
		the cell state of grid at (i,j) with j > len2 should be same as the grid on its top		
		'''
		# loop_vars		
		ta_shp = [None, None, cell.output_size ]	# ta_shp = [ dim1_maxlen, batchSize, cell.output_size ]
		H_ta_hori = tf.TensorArray( dtype = tf.float32, size = dim2_maxlen + 1, element_shape = ta_shp , clear_after_read = False )	# dim2_maxlen + 1 : extra space for initial zero states
		C_ta_hori = tf.TensorArray( dtype = tf.float32, size = dim2_maxlen + 1, element_shape = ta_shp , clear_after_read = False)	 		
		H_ta_vert = tf.TensorArray( dtype = tf.float32, size = dim2_maxlen + 1, element_shape = ta_shp , clear_after_read = False)
		C_ta_vert = tf.TensorArray( dtype = tf.float32, size = dim2_maxlen + 1, element_shape = ta_shp , clear_after_read = False)	

		# initialize the tensorArray at index 0 with zeros
		init_array = tf.zeros( shape = [dim1_maxlen, batchSize, cell.output_size ] , dtype = tf.float32 )
		H_ta_hori = H_ta_hori.write(0, init_array )
		C_ta_hori = C_ta_hori.write(0, init_array )
		H_ta_vert = H_ta_vert.write(0, init_array )
		C_ta_vert = C_ta_vert.write(0, init_array )


		def _dim2_step( dim2, H_ta_hori, C_ta_hori, H_ta_vert, C_ta_vert ):
			input_t_dim2 = tf.slice( s2, begin = [ dim2, 0,0 ] , size = [1,-1,-1] )
			input_t_dim2 = tf.squeeze( input_t_dim2, axis = 0 )
			
			copy_cond_2 = ( dim2 >= lens2 )	# use as constant in dim1_step()		

			# create a set of new tensor array for _dim1_step() using 
			shp = [None, cell.output_size]	# element shape
			h_ta_hori = tf.TensorArray( dtype = tf.float32, size = dim1_maxlen, element_shape = shp )
			c_ta_hori = tf.TensorArray( dtype = tf.float32, size = dim1_maxlen, element_shape = shp )	 		
			h_ta_vert = tf.TensorArray( dtype = tf.float32, size = dim1_maxlen, element_shape = shp )
			c_ta_vert = tf.TensorArray( dtype = tf.float32, size = dim1_maxlen, element_shape = shp )	

			# retrieve the previous states
			prev_H_hori = H_ta_hori.read( dim2 )
			prev_C_hori = C_ta_hori.read( dim2 )
			prev_H_vert = H_ta_vert.read( dim2 )
			prev_C_vert = C_ta_vert.read( dim2 )

			def _dim1_step( dim1, dim1_state , dim2_state, h_ta_hori, c_ta_hori, h_ta_vert, c_ta_vert):
				input_t_dim1 = tf.slice( s1, begin = [ dim1, 0,0 ] , size = [1,-1,-1] )
				input_t_dim1 = tf.squeeze( input_t_dim1, axis = 0 )
		

				upper_h_vert = prev_H_vert[ dim1 ]
				input_combine = tf.concat( [input_t_dim1, input_t_dim2, dim1_state.h, upper_h_vert ], axis = 1 )
				# compute the new state in two direction
				# horizontal
				call_horizontal = lambda: cell( inputs = input_combine, state = dim1_state )
		
				# vertical					
				upper_c_vert = prev_C_vert[ dim1 ] 
				upper_state_vert = tf.contrib.rnn.LSTMStateTuple( c = upper_c_vert, h = upper_h_vert )
				call_vertical = lambda: cell2( inputs = input_combine, state = upper_state_vert )

				with tf.variable_scope('horizontal'):	# you got to know when the framework of tensorflow actually create weights
					( _, new_state_hori ) = call_horizontal()
				with tf.variable_scope('vertical'):	
					( _, new_state_vert ) = call_vertical()
				copy_cond = (dim1 >= lens1 )	
				new_h_hori = tf.where( copy_cond, zero_state1.h, new_state_hori.h )	
				new_c_hori = tf.where( copy_cond, dim1_state.c, new_state_hori.c )
				new_h_vert = tf.where( copy_cond, zero_state2.h, new_state_vert.h )	
				new_c_vert = tf.where( copy_cond, dim2_state.c, new_state_vert.c )

				# copy the upper state or not
				upper_c_hori = prev_C_hori[ dim1 ] 
				new_h_hori = tf.where( copy_cond_2, zero_state1.h, new_h_hori )	
				new_c_hori = tf.where( copy_cond_2, upper_c_hori, new_c_hori )
				new_h_vert = tf.where( copy_cond_2, zero_state2.h, new_h_vert )	
				new_c_vert = tf.where( copy_cond_2, upper_c_vert, new_c_vert )

				h_ta_hori = h_ta_hori.write( dim1, new_h_hori )
				c_ta_hori = c_ta_hori.write( dim1, new_c_hori )
				h_ta_vert = h_ta_vert.write( dim1, new_h_vert )
				c_ta_vert = c_ta_vert.write( dim1, new_c_vert )
				
				return ( dim1+1,  tf.contrib.rnn.LSTMStateTuple( c = new_c_hori, h = new_h_hori) , tf.contrib.rnn.LSTMStateTuple( c = new_c_vert, h = new_h_vert ),
						 h_ta_hori, c_ta_hori, h_ta_vert, c_ta_vert)


			dim1 = tf.constant( 0, dtype = tf.int32, name = 'dim1_time' )
			_,_,_,h_ta_hori, c_ta_hori, h_ta_vert, c_ta_vert = tf.while_loop( cond = lambda time, *_ : time < dim1_maxlen, parallel_iterations = 10,
													body = _dim1_step, loop_vars = ( dim1, zero_state1 , zero_state2 , h_ta_hori, c_ta_hori, h_ta_vert, c_ta_vert ) )

			# stack the tensorArray into a single tensor and feed it to the higher level tensorArray			
			h_ts_hori = h_ta_hori.stack()
			c_ts_hori = c_ta_hori.stack()
			h_ts_vert = h_ta_vert.stack()
			c_ts_vert = c_ta_vert.stack()

			# write to higher level tensorArray
			H_ta_hori = H_ta_hori.write( dim2+1, h_ts_hori )
			C_ta_hori = C_ta_hori.write( dim2+1, c_ts_hori )
			H_ta_vert = H_ta_vert.write( dim2+1, h_ts_vert )
			C_ta_vert = C_ta_vert.write( dim2+1, c_ts_vert )


			return  (dim2+1,H_ta_hori, C_ta_hori, H_ta_vert, C_ta_vert)  

		dim2 = tf.constant( 0, dtype = tf.int32, name = 'dim2_time' )
		_ ,H_ta_hori, C_ta_hori, H_ta_vert, C_ta_vert= tf.while_loop( cond = lambda time, *_: time < dim2_maxlen, parallel_iterations = 1,
											body = _dim2_step, loop_vars = (dim2, H_ta_hori, C_ta_hori, H_ta_vert, C_ta_vert) )

		#
		final_c_hori = C_ta_hori.read( dim2_maxlen )
		final_c_vert = C_ta_vert.read( dim2_maxlen )
		return  tf.concat( [final_c_hori[-1], final_c_vert[-1]] , axis = 1 ) , C_ta_vert.stack(), H_ta_hori.stack(), H_ta_vert.stack()


	if( para.activationType == 1 ):
		activeFuc = tf.tanh
	elif( para.activationType == 2 ):
  		activeFuc = tf.identity

	if( para.rnnCellType == 1 ):
		cell = tf.contrib.rnn.BasicLSTMCell( hiddenSize , activation = activeFuc, forget_bias = para.forgetBias)		
	elif( para.rnnCellType == 2 ):
		cell = tf.contrib.rnn.GRUCell( hiddenSize )
	
	feaVec, h_ta = _2dRNN( cell, t_wordEmb1, t_wordEmb2, lens1, lens2 )[:2]
	
	if( para.similarityMetric == 1 ):
		# softmax layer
		w = tf.get_variable( 'softmax_weight', initializer = tf.truncated_normal( shape =  [hiddenSize*2, 5], stddev = 0.1 ) )	# in this case, feature vector is twice of hiddenSize
		b = tf.get_variable( 'softmax_bias' , initializer = tf.truncated_normal( shape =  [5], stddev = 0.1 ) )
		logits = tf.matmul( feaVec, w ) + b 
		prob = tf.nn.softmax( logits )

		r = tf.constant( range(1,6), shape = [5,1] , dtype = tf.float32 )
		y =  tf.matmul( prob, r ) 
		label = __scoreLabel2p( score )	

		
		wPenalty = __applyL2Norm()
		#wPenalty = L2Strength * ( tf.reduce_sum( tf.square( w )) + tf.reduce_sum( tf.square(w_a) ) + tf.reduce_sum( tf.square(w_d)) )	# weight decay penalty
		#-------------------
		loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = label, logits = logits, name = 'cross_entropy_loss' ) ) + wPenalty
		pearson_r = __pearson_r( y, score )

		# add summary for tensorboard visulization
		prob_mse = tf.reduce_mean(   tf.reduce_sum( (prob-label)**2 , axis = 1) )	
		mse = tf.reduce_mean( tf.square(y - score) )
		tf.summary.scalar( 'probability_MSE' , prob_mse )
		tf.summary.scalar( 'similarity_MSE' , mse )
		tf.summary.scalar( 'pearson_r', pearson_r )
		tf.summary.scalar( 'loss' , loss )

	elif( para.similarityMetric == 2 ):
		# In "Manhattan" metric, we derive the estimation of similarity directly from the Mahhattan distance of the final states of LSTM 		
		prob = tf.exp( -tf.reduce_sum( tf.abs( feas1[0] - feas2[0] ), axis = 1 , keep_dims = True) )
		a = tf.get_variable( 'regression_coef_a', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		b = tf.get_variable( 'regression_coef_b', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		c = tf.get_variable( 'regression_coef_c', initializer = tf.truncated_normal( shape =  [1, 1], stddev = 0.1 ) )
		prob = a*prob**2 + b*prob + c
		label = __rescaleLabel( score )
		wPenalty = __applyL2Norm()
		loss = tf.losses.mean_squared_error( labels = label, predictions = prob ) + wPenalty

		y = __rescaleLabel( prob , True)
		pearson_r = __pearson_r( y, score )
		prob_mse = mse = loss*16	# advanced computation

		# add summary for tensorboard visulization
		tf.summary.scalar( 'loss', loss )
		tf.summary.scalar( 'similarity_MSE' , mse )


	outputs1 = tf.constant(0,dtype=tf.int32)
	outputs2 = tf.constant(0,dtype=tf.int32)	# useless
	return [sent1, sent2, score, lens1, lens2], loss, prob, y, prob_mse, mse, outputs1, outputs2, pearson_r, h_ta

def averageModel( maxLen  ):		
	hiddenSize = 150;	

	lens1 = tf.placeholder( dtype = tf.float32, shape = [None] )
	lens2 = tf.placeholder( dtype = tf.float32, shape = [None] )
	sent1 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	sent2 = tf.placeholder( dtype = tf.int32, shape = [None,maxLen] )
	score = tf.placeholder( dtype = tf.float32, shape = [None, 1 ] )
	label = __scoreLabel2p( score )

	with tf.variable_scope( "embeddingLayer" ):
		tf.get_variable_scope().reuse_variables()
		embMat = tf.get_variable( "embedding" )
	embeddingSize = embMat.get_shape().as_list()[1]
	wordEmb1 = tf.nn.embedding_lookup( embMat, sent1 )
	wordEmb2 = tf.nn.embedding_lookup( embMat, sent2 )

	with tf.name_scope( 'sentence1' ):
		feas1 = __extractFeaVec( wordEmb1, lens1)
	with tf.name_scope( 'sentence2' ):
		feas2 = __extractFeaVec( wordEmb2, lens2 )


	feaVec, w_a, w_d = __feaTransform( feas1, feas2, hiddenSize )
	
	# softmax layer
	w = tf.get_variable( 'softmax_weight', initializer = tf.truncated_normal( shape =  [hiddenSize, 5], stddev = 0.1 ) )
	b = tf.get_variable( 'softmax_bias' , initializer = tf.truncated_normal( shape =  [5], stddev = 0.1 ) )
	logits = tf.matmul( feaVec, w ) + b 
	prob = tf.nn.softmax( logits )
	r = tf.constant( range(1,6), shape = [5,1] , dtype = tf.float32 )
	y =  tf.matmul( prob, r ) 
	prob_mse = tf.reduce_mean(   tf.reduce_sum( (prob-label)**2 , axis = 1) )	
	mse = tf.reduce_mean( tf.square(y - score) )
	#pearson_r, _ = tf.contrib.metrics.streaming_pearson_correlation( y, score )	# the second return value is update_op
	pearson_r = __pearson_r( y, score )

	L2Strength = 1e-2	# L2 regulariztion strength
	wPenalty = L2Strength * ( tf.reduce_sum( tf.square( w )) + tf.reduce_sum( tf.square(w_a) ) + tf.reduce_sum( tf.square(w_d)) )	# weight decay penalty
	loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = label, logits = logits, name = 'cross_entropy_loss' ) ) + wPenalty

	# add summary for tensorboard visulization
	tf.summary.scalar( 'probability_MSE' , prob_mse )
	tf.summary.scalar( 'similarity_MSE' , mse )
	tf.summary.scalar( 'pearson_r', pearson_r )
	tf.summary.scalar( 'loss' , loss )

	return [sent1, sent2, score, lens1, lens2], loss, prob, y, prob_mse, mse, pearson_r , wordEmb1, feas1[0]

def __getWb( shape , tensorName ):
	w = tf.get_variable( tensorName + '_weight', initializer = tf. truncated_normal( shape = shape, stddev = 0.1 ) )
	b = tf.get_variable( tensorName + '_bias' , initializer = tf. truncated_normal( [ shape[1] ] , stddev = 0.1 ) )
	return w,b

def __extractFeaVec( wordEmb , lens):
	lens = tf.cast( lens, tf.float32 )	

	sumpool = tf.reduce_sum( wordEmb, axis=1, name = 'sum_pooling' )
	maxpool = tf.reduce_max( wordEmb, axis=1, name = 'max_pooling' )
	minpool = tf.reduce_min( wordEmb, axis=1, name = 'min_pooling' )
	
	batchSize,embeddingSize = sumpool.get_shape().as_list()
	# scaling sum_pooling to get mean-pooling in the sense of length normalization
	factor = tf.tile( tf.expand_dims( 1/lens, 1 )	, multiples = [1,embeddingSize] )
	print '#############'	
	print factor.get_shape()
	print sumpool.get_shape()
	ave = tf.multiply( factor, sumpool , name = 'length_specified_mean_pooling')
	
	#return [ave, maxpool, minpool ]
	return [maxpool,minpool]
	
def __feaTransform( feas1, feas2, outputSize ):
	feas_a = []	
	feas_d = []
	for fea1,fea2 in zip( feas1, feas2 ):		
		h_angle = tf.multiply( fea1, fea2 )	# get angle feature
		h_dist = tf.abs( fea1-fea2 )		# get distance feature
		feas_a.append( h_angle )
		feas_d.append( h_dist )

	H_angle = tf.concat( feas_a, axis = 1, name = 'angle_feature' )
	H_dist = tf.concat( feas_d, axis = 1, name = 'distance_feature' )
	srcFeaSize = feas1[0].get_shape().as_list()[1]
	feasCnt = len( feas1 )
	w_a = tf.get_variable( 'angleFea' + '_weight', initializer = tf. truncated_normal( shape = [srcFeaSize*feasCnt,outputSize], stddev = 0.1 ) )
	w_d = tf.get_variable( 'distFea' + '_weight', initializer = tf. truncated_normal( shape = [srcFeaSize*feasCnt,outputSize], stddev = 0.1 ) )
	b_s = tf.get_variable( 'hs' + '_bias', initializer = tf. truncated_normal( shape = [outputSize], stddev = 0.1 ) )
	feaVec = tf.sigmoid( tf.matmul( H_angle, w_a ) + tf.matmul( H_dist, w_d ) + b_s , name = 'feature_vector')	
	return feaVec, w_a, w_d

# score label to a sparse target distribution p	, this implement takes Tensor as input and return a Tensor
def __scoreLabel2p( score ):
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
	score -= 1		
	i = range(5)
	d_pos = tf.minimum( 1.0, score-i )
	d_neg = tf.minimum( 1.0, -d_pos )
	p = tf.minimum( 1.0-d_pos, 1.0-d_neg )
	return p
	
# rescale the label to the range from 0 to 1 
def  __rescaleLabel( score , inverse = False):
	'''
	the relatedness score is range from 1 to 5, rescale it with the following formula:
	scaled_score  = (score - 1) / 4
	this is an linear trasform
	'''
	if( inverse ):
		return score*4+1
	else:
		return (score-1)/4

	
# compute the pearson correlation coefficient
def __pearson_r( X, Y ):
	meanX = tf.reduce_mean( X )
	meanY = tf.reduce_mean( Y )
	covXY = tf.reduce_mean( (X - meanX) * ( Y - meanY ) )
	covXX = tf.reduce_mean( (X - meanX) **2 )
	covYY = tf.reduce_mean( (Y - meanY) **2 )

	return covXY / (tf.sqrt(covXX)*tf.sqrt(covYY))

# apply L2 regularization on trainable variables
def __applyL2Norm( L2Strength = 3e-4):
	ls = tf.trainable_variables()
	print 'variable to be regulized:'
	for ts in ls:
		print ts.name
	wPenalty = tf.contrib.layers.apply_regularization( regularizer = tf.contrib.layers.l2_regularizer(L2Strength) , weights_list = ls )
	return wPenalty
