#Damir Jajetic, 2016, MIT licence

def predict (LD, output_dir, basename):
	
	import os
	import numpy as np
	import random
	import data_converter
	from sklearn import preprocessing, feature_selection, decomposition
	from sklearn.utils import shuffle
	import time
	from sklearn.externals import joblib
	from scipy import sparse
	
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
	
	fs = decomposition.TruncatedSVD(n_components=400, n_iter=5, random_state=1)
	fs.fit(X_train)
	X_train = fs.transform(X_train)
	X_valid = fs.transform(X_valid)
	X_test = fs.transform(X_test)
	
	
	normx = preprocessing.Normalizer()
	
	normx.fit(X_train)
	X_train = normx.transform(X_train)
	X_valid = normx.transform(X_valid)
	X_test = normx.transform(X_test)
	
	y_train = np.copy(LD.data['Y_train'])
	
	def batches(X, y, csize, rs):
		X, y = shuffle(X, y, random_state=rs)
		for cstart in range(0, X.shape[0] - csize+1, csize):
			Xc = X[cstart:cstart+csize] 
			yc = y[cstart:cstart+csize]
			
			Xc = np.float32(Xc)
			yc = np.float32(yc)
			yield  Xc, yc
	
	input_var = T.matrix('inputs')
	target_var = T.matrix('targets')
	
	l_in = lasagne.layers.InputLayer(shape=(None, X_train.shape[1]),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.Sparse(),
	     input_var=input_var)
	     
	l_hid1 = lasagne.layers.DenseLayer(
	    l_in, num_units= 600,
	    nonlinearity=lasagne.nonlinearities.rectify,
	    W=lasagne.init.Sparse()
	    )
	    
	l_hid2 = lasagne.layers.DenseLayer(
	    l_hid1, num_units= 600,
	    nonlinearity=lasagne.nonlinearities.rectify,
	    W=lasagne.init.Sparse()
	)
	
	Lnum_out_units = y_train.shape[1]
	
	
	l_out = lasagne.layers.DenseLayer(
		l_hid2, num_units=Lnum_out_units,
		nonlinearity=lasagne.nonlinearities.sigmoid)

	network = l_out
	
	prediction = lasagne.layers.get_output(network)

	loss = lasagne.objectives.squared_error(prediction, target_var)
	loss = loss.mean()
	
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.3, momentum=0.90)
	
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	for epoch in range(20):		
		train_err = 0
		train_batches = 0
		for batch in batches(X_train, y_train, epoch+1, epoch):
		    Xt, yt = batch
		    train_err += train_fn(Xt, yt)
		    train_batches += 1
		
	xml1 = T.matrix('xml1')
	Xlt1 = lasagne.layers.get_output(l_out, xml1, deterministic=True)
	f2 = theano.function([xml1], Xlt1)
		
	csize= 1000
	preds_valid = np.zeros((X_valid.shape[0], y_train.shape[1]))
	for cstart in range(0, X_valid.shape[0], csize):			
		Xo = X_valid[cstart:cstart+csize]
		Xo = np.float32(Xo)
		pp = f2(Xo)
		preds_valid[cstart:cstart+csize] = pp
	
	preds_test = np.zeros((X_test.shape[0], y_train.shape[1]))
	for cstart in range(0, X_test.shape[0], csize):			
		Xo = X_test[cstart:cstart+csize]
		Xo = np.float32(Xo)
		pp = f2(Xo)
		preds_test[cstart:cstart+csize] = pp
			

	import data_io
	if  LD.info['target_num']  == 1:
		preds_valid = preds_valid[:,1]
		preds_test = preds_test[:,1]
					

	eps = 0.0001
	preds_valid = np.round(np.clip(preds_valid,0+eps,1-eps),4)
	preds_test = np.round(np.clip(preds_test,0+eps,1-eps),4)
	
	
	cycle = 0 
	filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_valid), preds_valid)
	filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
	data_io.write(os.path.join(output_dir,filename_test), preds_test)

	
	