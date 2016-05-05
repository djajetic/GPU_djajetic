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
	from lasagne.updates import norm_constraint, total_norm_constraint
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
	
	normx = preprocessing.StandardScaler()
	
	normx.fit(X_train)
	X_train = normx.transform(X_train)
	X_valid = normx.transform(X_valid)
	X_test = normx.transform(X_test)
	
	X_train = np.float32(X_train)
	X_valid = np.float32(X_valid)
	X_test = np.float32(X_test)
	
	y_train = np.copy(LD.data['Y_train'])
	y_train = np.int16(y_train)
	
	normy = preprocessing.MinMaxScaler(feature_range=(-1, 1))
	normy.fit(y_train)
	y_train = normy.transform(y_train)
	y_train = np.float32(y_train)
	
	y_train = y_train.reshape((-1, 1))
	
	def batches(X, y, csize, rs):
		X, y = shuffle(X, y, random_state=rs)
		for cstart in range(0, X.shape[0] - csize+1, csize):
			Xc = X[cstart:cstart+csize] 
			yc = y[cstart:cstart+csize]
			yield  Xc, yc
	
	input_var = T.matrix('inputs')
	target_var = T.matrix('targets')
	
	l_in = lasagne.layers.InputLayer(shape=(None, X_train.shape[1]),
	     input_var=input_var,
	     nonlinearity=lasagne.nonlinearities.tanh,
	     W=lasagne.init.Sparse()
	     )		
	
	l_hid1 = lasagne.layers.DenseLayer(
	    l_in, num_units= 100,
	    nonlinearity=lasagne.nonlinearities.tanh,
	    W=lasagne.init.Sparse())
	
	
	l_hid2 = lasagne.layers.DenseLayer(
	    l_hid1, num_units= 100,
	    nonlinearity=lasagne.nonlinearities.tanh,
	    W=lasagne.init.Sparse()
	    )
	    
	
	l_hid3 = lasagne.layers.DenseLayer(
	    l_hid2, num_units= 100,
	    nonlinearity=lasagne.nonlinearities.tanh,
	    W=lasagne.init.Sparse()
	    )
	
	Lnum_out_units = 1

	l_out = lasagne.layers.DenseLayer(
		l_hid3, num_units=Lnum_out_units,
		nonlinearity=None)

	network = l_out
	
	prediction = lasagne.layers.get_output(network)

	loss = lasagne.objectives.squared_error(prediction, target_var)
	loss = loss.mean()
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.sgd(loss, params, learning_rate=0.25)
		
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	for epoch in range(40):		
		train_err = 0
		train_batches = 0
		for batch in batches(X_train, y_train, epoch+40, epoch):
		    Xt, yt = batch
		    train_err += train_fn(Xt, yt)
		    train_batches += 1
		
	xml1 = T.matrix('xml1')
	Xlt1 = lasagne.layers.get_output(l_out, xml1, deterministic=True)
	f2 = theano.function([xml1], Xlt1)
	preds_valid = f2(X_valid)
	preds_valid = normy.inverse_transform(preds_valid)
	preds_test = f2(X_test)
	preds_test = normy.inverse_transform(preds_test)

	import data_io
	
	cycle = 0 
	filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
	filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_test), preds_test)

	