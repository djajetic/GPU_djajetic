#Damir Jajetic, 2016, MIT licence

def predict (LD, output_dir, basename):

	import os
	import numpy as np
	import random
	import data_converter
	from sklearn import preprocessing, decomposition
	from sklearn.utils import shuffle
	import time
	from sklearn.externals import joblib

	
	from lasagne import layers
	from lasagne.updates import nesterov_momentum
	from lasagne.updates import norm_constraint
	import lasagne
	import theano
	import theano.tensor as T
	from lasagne.regularization import regularize_layer_params, regularize_layer_params_weighted, l2, l1
	np.random.seed(0)
	random.seed(0)

	
	LD.data['X_train'], LD.data['Y_train'] = shuffle(LD.data['X_train'], LD.data['Y_train'] , random_state=1)
	X_train = LD.data['X_train']
	X_valid = LD.data['X_valid']
	X_test = LD.data['X_test']
	
	
	X_train = X_train.toarray()
	X_train = np.float32(X_train)
	X_valid = X_valid.toarray()
	X_valid = np.float32(X_valid)
	X_test = X_test.toarray()
	X_test = np.float32(X_test)
	
	y_train = LD.data['Y_train']
	y_train = np.float32(y_train)
	y_train = np.reshape(y_train, (-1, 1))
	
	
	fs = decomposition.PCA()
	fs.fit(X_train)
	X_train2 = fs.transform(X_train)
	X_valid2 = fs.transform(X_valid)
	X_test2 = fs.transform(X_test)
	
	
	X_train = np.hstack([X_train, X_train2])
	X_valid = np.hstack([X_valid, X_valid2])
	X_test = np.hstack([X_test, X_test2])
	
	def batches(X, y, csize, rs):
		X, y = shuffle(X, y, random_state=rs)
		for cstart in range(0, len(X) - csize+1, csize):
			Xc = X[cstart:cstart+csize] 
			yc = y[cstart:cstart+csize]
			yield  Xc, yc
	
	input_var = T.matrix('inputs')
	target_var = T.matrix('targets')
	
	l_in = lasagne.layers.InputLayer(shape=(None, X_train.shape[1]),
	     input_var=input_var,
	     )
	     
	l_hid1 = lasagne.layers.DenseLayer(
	    l_in, num_units= 5000,
	    nonlinearity=lasagne.nonlinearities.rectify,
	    W=lasagne.init.Sparse())
	
	Lnum_out_units = 1
	
	l_out = lasagne.layers.DenseLayer(
		l_hid1, num_units=Lnum_out_units,
		nonlinearity=None)

	network = l_out
	
	prediction = lasagne.layers.get_output(network)

	loss = lasagne.objectives.squared_error(prediction, target_var)
	loss = loss.mean()
	
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.sgd(loss, params, learning_rate=0.001)
	
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	for epoch in range(60):
		train_err = 0
		train_batches = 0
		for batch in batches(X_train, y_train, 20, epoch):
		    Xt, yt = batch
		    train_err += train_fn(Xt, yt)
		    train_batches += 1
		
	xml1 = T.matrix('xml1')
	Xlt1 = lasagne.layers.get_output(l_out, xml1, deterministic=True)
	f2 = theano.function([xml1], Xlt1)
		
	preds_valid = f2(X_valid)
	preds_test = f2(X_test)
			

	import data_io
	if  LD.info['target_num']  == 1:
		pass
								
	preds_valid = np.clip(preds_valid,0,1)
	preds_test = np.clip(preds_test,0,1)
	
	cycle = 0 
	filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
	filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_test), preds_test)

