-------------------------------------------------------------------------------
-- This script is a modification of train-on-mnist.lua found at 
-- github.com/torch/demos
--
-- This script constructs MLPs with sigmoid neurons. The aim is to understand 
-- how the performance depends on whether the outputs are coded in a one-hot 
-- manner or binary.
-- Example: 
-- One hot encoding maps class 3 to [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
-- Binary encoding maps class 3 to [1, 1, 0, 0]
--
-- Rather than use softmax and log-loss we directly use squared loss, since the 
-- sigmoid neurons are expected to produce an output that is close to {0, 1}.
-- The purpose of this is educational rather than optimal training on MNIST
-------------------------------------------------------------------------------
require 'torch';
require 'nn';
require 'optim';
require 'dataset-mnist';
require 'pl';
require 'paths';

-------------------------------------------------------------------------------
-- parse command-line options
--
-- plotting has not been implemented
local opt = lapp[[
	-s, --save		(default "logs") 	subdirectory to save logs
	-n, --network		(default "")		reload pretrained network	
	-e, --encoding		(default "one-hot")	specify output encoding: one-hot | binary
	-f, --full					use the full dataset
	-p, --plot					plot while training
	-r, --learningRate	(default 4)		learning rate for SGD
	-b, --batchSize		(default 10)		batch size
	-m, --momentum		(default 0)		momentum for SGD
	--coefL1		(default 0)		L1 penalty on the weights
	--coefL2		(default 0)		L2 penalty on the weights
	-t, --threads		(default 4)		number of threads
	--ntrain		(default 2000)		number of training examples (ignored if the -f flag is set)
	--ntest			(default 1000)		number of test examples (ignored if the -f flag is set)
	-i, --iterations	(default 5)		number of training epochs
	--hidden1		(default nil)		number of units in first hidden layer
	--hidden2		(default nil)		number of units in second hidden layer
	-v, --verbosity		(default 1)		verbosity level
]]

-- Verbosity
-- 0 prints only set up information
-- 1 prints only training and test loss and epoch number
-- 2 prints training and test loss and classification error
-- 3 prints confusion matrices in addition to everything above
--

-- fix seed to get reproducible results
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb threads to ' .. torch.getnumthreads())

-- set default tensor type to float
torch.setdefaulttensortype('torch.FloatTensor')

-- MNIST specific definitions

mt = {}
mt.classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
mt.geometry = {32, 32}

-- determine number and sizes of hidden layers
if opt.hidden2 ~= 'nil' and opt.hidden1 == 'nil' then
	print("It's not nice to put a second hidden layer without putting a first one")
	os.exit(1)
elseif opt.hidden2 ~='nil' and opt.hidden1 ~= 'nil' then
	mt.hiddens = {tonumber(opt.hidden1), tonumber(opt.hidden2)} 
elseif opt.hidden2 == 'nil' and opt.hidden1 ~= 'nil' then
	mt.hiddens = {tonumber(opt.hidden1)}
else
	mt.hiddens = {}
end

-------------------------------------------------------------------------------
-- define class to define functions
mnist_train = {}

function mnist_train.makeModel(h_sizes)
	model = nn.Sequential()
	model:add(nn.Reshape(1024))
	local insize = 1024
	for i=1,#h_sizes do
		model:add(nn.Linear(insize, h_sizes[i]))
		model:add(nn.Sigmoid())
		insize = h_sizes[i]
	end
	local n_outputs = 10
	if opt.encoding == 'binary' then
		n_outputs = 4
	end
	model:add(nn.Linear(insize, n_outputs))
	model:add(nn.Sigmoid())
	return model
end

function mnist_train.initialize() 
	-- check that model has already been defined
	if model == nil then
		print('model not yet defined')
		return
	end

	-- reset epoch to 1
	epoch = 1

	-- Initialize the weights and biases of each layer
	-- If l^{th} layer has n_l neurons, then the weights between the fully
	-- connected layer between layer (l - 1) and l are all initialized to be
	-- normal random variables with mean 0 and variance 1/(n_l)
	local nlayers = #(model)
	for i =1, nlayers do
		if model.modules[i].weight ~= nil then
			local tsize = model.modules[i].weight:size()
			model.modules[i].weight:copy(torch.randn(tsize):mul(1/math.sqrt(tsize[2])))
		end
	end
	parameters, gradParameters = model:getParameters()

	-- Use the squared error
	criterion = nn.MSECriterion()

	-- Load training and testing data
	if opt.full then
		nbTrainingExamples = 60000
		nbTestingExamples = 10000
		print('Using the entire dataset; this may be some time to run.')
	else
		nbTrainingExamples = opt.ntrain
		nbTestingExamples = opt.ntest
		print('Using ' .. nbTrainingExamples .. ' for training and ' ..
		nbTestingExamples .. ' for testing.')
	end

	-- create training set and normalize
	trainData = mnist.loadTrainSet(nbTrainingExamples, mt.geometry)
	mean, std = trainData:normalize()

	-- create testing set and normalize
	testData = mnist.loadTestSet(nbTestingExamples, mt.geometry)
	testData:normalize(mean, std)

	-- if binary coding is to be used set flags to make training and testing
	-- data return outputs encoded in binary
	if opt.encoding == 'binary' then
		trainData.binary_output = true
		testData.binary_output = true
	end

	-- this matrix records confusion across classes
	confusion = optim.ConfusionMatrix(mt.classes)

	-- if log folder is not explicitly given then add timestamp to aovid overwriting
	if opt.save == 'logs' then
		opt.save = 'logs_' .. string.gsub(string.gsub(os.date(), ' ', '_'), ':', '-')
	end
	trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
	testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
	trainLossLogger = optim.Logger(paths.concat(opt.save, 'train_loss.log'))
	testLossLogger = optim.Logger(paths.concat(opt.save, 'test_loss.log'))
	regularizationLogger = optim.Logger(paths.concat(opt.save, 'regularization.log'))

	-- save opt
	torch.save(paths.concat(opt.save, 'opt_parameters.txt'), opt)
end

-- helper functions
-- when the output is 10 dimensional then it simply picks the largest value as label
-- when the output is binary, it rounds to 0, 1 each output and then outputs the resulting number in binary
-- since the classes are only between 1-10, 0 -> 1 and 11-15 -> 10
function convert_to_labels(preds)
	batchSize = preds:size(1)
	local pred_labels = torch.Tensor(batchSize)
	if opt.encoding == 'binary' then
		local bmul = torch.Tensor{1, 2, 4, 8}:reshape(4, 1)
		pred_labels = (preds:ge(0.5):typeAs(bmul)*bmul):squeeze():apply(function (x) return math.max(math.min(x, 10), 1) end)
	else
		_, pred_labels = preds:max(2)
	end
	return pred_labels
end
-- same as above except here attempt is made to find the closest binary encoding
-- preferred usage is this function
function convert_to_labels2(preds)
	batchSize = preds:size(1)
	local pred_labels = torch.Tensor(batchSize)
	if opt.encoding == 'binary' then
		local bcode = torch.Tensor(10, 4):zero()
		local distances = torch.Tensor(batchSize, 10)
		for i =1, 10 do
			if i%2 >=1 then bcode[i][1] = 1 end
			if i%4 >=2 then bcode[i][2] = 1 end
			if i%8 >=4 then bcode[i][3] = 1 end
			if i%16 >=8 then bcode[i][4] = 1 end
			distances[{{}, {i}}]:copy(torch.norm(bcode[i]:reshape(1, 4):expand(batchSize, 4) - preds, 2, 2))
		end
		_, pred_labels = distances:min(2)
	else
		_, pred_labels = preds:max(2)
	end
	return pred_labels
end

-- training function
function mnist_train.train(dataset)
	-- epoch tracker
	epoch = epoch or 1

 	-- local vars
 	local time = sys.clock()

	-- track training loss
	-- don't add regularization term to training loss
 	-- instead keep track separately
	local tr_loss = 0
	local regularization = 0
 
 	-- do one epoch
	if opt.verbosity >= 1 then
 		print('<mnist_train> on training set: epoch ' .. epoch)
		if opt.verbosity >=2 then
 			print('<mnist_train> [batchSize = ' .. opt.batchSize .. ']')
		end
	end
 	for t = 1,dataset:size(), opt.batchSize do
 		-- create mini batch
		-- first check size of minibatch (only required for last batch)
		local batchSize = math.min(dataset:size() - t + 1, opt.batchSize)
 		local inputs = torch.Tensor(batchSize, 1, mt.geometry[1], mt.geometry[2])
 		local targets = torch.Tensor(batchSize, dataset[1][2]:size(1))
		local target_labels = torch.Tensor(batchSize)
		-- copy the next batchSize elements to create a minibatch
 		local k = 1
 		for i = t, t + batchSize - 1 do
 			-- load new sample
 			local sample = dataset[i]
 			inputs[k] = sample[1]:clone()
 			targets[k] = sample[2]:clone()
			target_labels[k] = dataset.labels[i]
 			k = k + 1
 		end
 
 		-- create closure to evaluate f(X) and df/dX
 		local feval  = function(x)
  			-- just in case:
  			collectgarbage()
  
  			-- get new parameters
  			if x ~= parameters then
  				parameters:copy(x)
  			end
  
  			-- reset gradients
  			gradParameters:zero()
  
  			-- evaluate function for complete minibatch
  			local preds = model:forward(inputs)
  			local f = criterion:forward(preds, targets)

  			-- estimate df/dW
  			local df_do = criterion:backward(preds, targets)
  			model:backward(inputs, df_do)
  
  			-- Penalties (L1 and L2)
			if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
				-- Update the Gradients
				-- The training loss will be update to reflect the regularization
				-- term at the end of the epoch
				gradParameters:add(torch.sign(parameters):mul(opt.coefL1) +
				parameters:clone():mul(opt.coefL2))
			end

			-- Update training_loss
			-- nn.MSECriterion() returns mean squared error over mini batch
			tr_loss = tr_loss + f*batchSize

			-- convert targets into pred_labels
			local pred_labels = convert_to_labels2(preds):squeeze()
			-- update confusion matrix
			confusion:batchAdd(pred_labels, target_labels)

  
  			return f, gradParameters
		end

		-- optimize on current mini-batch
 		sgdState = sgdState or {
 			learningRate = opt.learningRate,
 			momentum = opt.momentum,
 			learningRateDecay = 5e-7
 		}
 		optim.sgd(feval, parameters, sgdState)
 
 		-- disp progress
 		xlua.progress(t, dataset:size())
 	end
 
 	-- time taken
 	time = sys.clock() - time
 	time = time/dataset:size()
	if opt.verbosity >=3 then
 		print("<trainer> time to learn 1 sample = " .. (time * 1000) .. 'ms')
	end

	-- adjust training loss and compute regularization terms 
	tr_loss = tr_loss/dataset:size()
	if opt.verbosity >=1 then
 		print("<trainer> training_loss = " .. tr_loss)
	end
	if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
		regularization = opt.coefL1*torch.norm(parameters, 1)
		regularization =  opt.coefL2*torch.norm(parameters, 2)^2/2
		if opt.verbosity >=2 then
			print("<trainer> regularization term = " .. regularization)
		end
	end


	-- print confusion matrix
	if opt.verbosity >=3 then
		print(confusion)
	end

	-- do logging
	
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	trainLossLogger:add{['criterion - squared error (train set)'] = tr_loss}
	regularizationLogger:add{['regularization term (train set)'] = regularization}

	-- logs model at every epoch
	local filename = paths.concat(opt.save, 'minst_epoch' .. epoch .. '.net')
	if opt.verbosity >= 3 then
		print('<trainer> saving network to ' .. filename)
	end
	torch.save(filename, model)

	-- next epoch and reset confusion
	epoch = epoch + 1
	confusion:zero()
end

-- test function
function mnist_train.test(dataset)
	-- rest confusion
	confusion:zero()

	-- local vars
	local time = sys.clock()

	-- track testing loss
	local tst_loss = 0

	-- test over given dataset
	if opt.verbosity >= 1 then	
		print('<trainer> on testing set:')
	end
	for t = 1,dataset:size(), opt.batchSize do
		-- disp process
		xlua.progress(t, dataset:size())

		-- create mini batch
		-- first check size of minibatch (only required for last batch)
		local batchSize = math.min(dataset:size() - t + 1,
		opt.batchSize)
		local inputs = torch.Tensor(batchSize, 1, mt.geometry[1],
		mt.geometry[2])
		local targets = torch.Tensor(batchSize, dataset[1][2]:size(1))
		local target_labels = torch.Tensor(batchSize)
		-- copy the next batchSize elements to create a minibatch
		local k = 1
		for i = t, t + batchSize - 1 do
			-- load new sample
			local sample = dataset[i]
			inputs[k] = sample[1]:clone()
			targets[k] = sample[2]:clone()
			target_labels[k] = dataset.labels[i]
			k = k + 1
		end

		-- test samples
		local preds = model:forward(inputs)
		tst_loss = tst_loss + criterion:forward(preds,targets) * batchSize

		-- predict labels
		local pred_labels = convert_to_labels2(preds)
		-- add to confusion matrix
		confusion:batchAdd(pred_labels, target_labels)
	end

	-- timing
	time = sys.clock() - time
	time = time / dataset:size()
	if opt.verbosity >=3 then
	 	print("<trainer> time to test 1 sample = " .. (time * 100) .. 'ms')
	end

	-- adjust testing loss 
	tst_loss = tst_loss/dataset:size()
	if opt.verbosity >=1 then
		print("<trainter> testing loss = " .. tst_loss)
	end

	-- print confusion matrix
	if opt.verbosity >= 3 then
		print(confusion)
	end

	-- do loggging
	testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	testLossLogger:add{['criterion - squared error (test set)'] = tst_loss}
end
-------------------------------------------------------------------------------
-- and train!
--
function do_all()
	mnist_train.makeModel(mt.hiddens)
	mnist_train.initialize()
	for i=1,opt.iterations do
		-- train/test
		mnist_train.train(trainData)
		mnist_train.test(testData)
		
		-- plot errors
		if opt.plot then
			trainLogger:style{['% mean class accuracy (train set)'] = '-'}
			testLogger:style{['% mean class accuracy (test set)'] = '-'}
			trainLossLogger:style{['criterion - squared error (train set)'] = '-'}
			testLossLogger:style{['criterion - squared error (test set)'] = '-'}
			trainLogger:plot()
			testLogger:plot()
			trainLossLogger:plot()
			testLossLogger:plot()
		end
	end
end

do_all()
--print('If you did dofile from th')
--print('First check opt and mt')
--print('Only then run do_all()')
---------------------------------------------------------------------------------------
