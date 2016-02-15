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
-- TODO: logging, networking, plotting has not been implemented
-- TODO: coefL1, coefL2 are not yet implemented
local opt = lapp[[
	-s, --save				(default "logs") 		subdirectory to save logs
	-n, --network			(default "")			reload pretrained network	
	-e, --encoding			(default "one-hot")	specify output encoding: one-hot | binary
	-f, --full											use the full detaset
	-p, --plot											plot while training
	-r, --learningRate	(default 4)				learning rate for SGD
	-b, --batchSize		(default 10)			batch size
	-m, --momentum			(default 0)				momentum for SGD
	--coefL1					(default 0)				L1 penalty on the weights
	--coefL2					(default 0)				L2 penalty on the weights
	-t, --threads			(default 4)				number of threads
	--ntrain					(default 2000)			number of training examples (ignored if the -f flag is set)
	--ntest					(default 1000)			number of test examples (ignored if the -f flag is set)
]]

-- fix seed to get reproducible results
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb threads to ' .. torch.getnumthreads())

-- set default tensor type to float
torch.setdefaulttensortype('torch.FloatTensor')

-- MNIST specific definitions
classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
geometry = {32, 32}
-- one-hot uses num_classes dimensional output layer
num_classes = 10
-- binary uses num_bits dimensionay output layer
num_bits = 4

-------------------------------------------------------------------------------
-- define class to define functions
mnist_train = {}

function mnist_train.makeModel(n_hidden, n_outputs)
	model = nn.Sequential()
	model:add(nn.Reshape(1024))
	model:add(nn.Linear(1024, n_hidden))
	model:add(nn.Sigmoid())
	model:add(nn.Linear(n_hidden, n_outputs))
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

	--
	-- Initialize the weights and biases of each layer
	-- If l^{th} layer has n_l neurons, then the weights between the fully
	-- connected layer between layer (l - 1) and l are all initialized to be
	-- normal random variables with mean 0 and variance 1/(n_l)
	-- TODO: The above is not yet implemented
	parameters, gradParameters = model:getParameters()
	parameters:copy(torch.randn(#parameters)):mul(0.1)

	-- Use the squared error
	criterion = nn.MSECriterion()

	-- Load training and testing data
	if opt.full then
		nbTrainingExamples = 60000
		nbTestingExamples = 10000
		print('Using the entire dataset; this may be some time to run.')
	else
		nbTrainingExamples = opt.ntrain
		nbTestingExamples = ot.ntest
		print('Using ' .. nbTrainingExamples .. ' for training and ' ..
		nbTestingExamples .. ' for testing.')
	end

	-- create training set and normalize
	trainData = mnist.loadTrainSet(nbTrainingExamples, geometry)
	mean, std = trainData:normalize()

	-- create testing set and normalize
	testData = mnist.loadTestSet(nbTestingExamples, geometry)
	testData:normalize(mean, std)

	-- if binary coding is to be used set flags to make training and testing
	-- data return outputs encoded in binary
	if opt.encoding == 'binary' then
		trainData.binary_output = true
		testData.binary_output = true
	end
end

-- training function
function mnist_train.train(dataset)
	-- epoch tracker
	epoch = epoch or 1

 	-- local vars
 	local time = sys.clock()

	-- track training loss
	-- TODO: initialize training_loss Tensors
	-- TODO: initialize the confusion matrix
	trainin_loss[epoch] = 0
 
 	-- do one epoch
 	print('<trainer> on training set: ')
 	print('<trainer> online epoch # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
 	for t = 1,dataset:size(), opt.batchSize do
 		-- create mini batch
 		local inputs = torch.Tensor(opt.batchSize, 1, geometry[1], geometry[2])
 		local targets = torch.Tensor(opt.batchSize, dataset[1][2]:size(1))
		local target_labels = torch.Tensor(opt.batchSize)
 		local k = 1
		-- TODO: CHECK LINES BELOW
		local t_batchSize = math.min(dataset:size() - t + 1, opt.batchSize)
 		for i = t, math.min(t + opt.batchSize - 1, dataset:size()) do
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
  			local outputs = model:forward(inputs)
  			local f = criterion:forward(outputs, targets)

  
  			-- estimate df/dW
  			local df_do = criterion:backward(outputs, targets)
  			model:backward(inputs, df_do)
  
  			-- Penalties (add later)
			-- TODO

			-- Update training_loss
			-- nn.MSECriterion() returns mean squared error over mini batch
			if 
			training_loss[epoch] = training_loss[epoch] + f*inputs:size(1)
  
  			return f, gradParameters
		end

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
	training_loss = training_loss/dataset:size()
 	print("<trainer> time to learn 1 sample = " .. (time * 1000) .. 'ms')
 	print("training_loss = " .. training_loss)

	epoch = epoch + 1
end

-- test function
function test(dataset)
	-- local vars
	local time = sys.clock()
	local cl_acc = 0
	test_loss = 0

	-- test over given dataset
	print('<trainer> on testing Set:')
	for t = 1,dataset:size(), opt.batchSize do
		-- dist process
		xlua.progress(t, dataset:size())

		-- create mini batch
		local inputs = torch.Tensor(opt.batchSize, 1, geometry[1], geometry[2])
		local targets = torch.Tensor(opt.batchSize, dataset[1][2]:size(1))
		local k = 1
		for i = t, math.min(t + opt.batchSize - 1, dataset:size()) do
			-- load new sample
			local sample = dataset[i]
			local input = sample[1]:clone()
			local target = sample[2]:clone()
			inputs[k] = input
			targets[k] = target
			k = k + 1
		end

		-- test samples
		local raw_preds = model:forward(inputs)
		test_loss = test_loss + criterion:forward(raw_preds, targets)*opt.batchSize
		preds = raw_preds:ge(0.5)
		cl_acc = cl_acc + preds:typeAs(targets):eq(targets):sum(2):eq(preds:size(2)):sum()
	end

	-- timing
	time = sys.clock() - time
	time = time / dataset:size()
	print("<trainer> time to test 1 sample = " .. (time * 100) .. 'ms')

	local cl_err = dataset:size() - cl_acc
	cl_err = cl_err / dataset:size()
	test_loss = test_loss/dataset:size()
	print ('test_loss  = ' .. test_loss)
	print ('Classification Error = ' .. cl_err)
end
-------------------------------------------------------------------------------
-- and train!
-- 
function trainRounds(n_rounds) 
	for i=1,n_rounds do
		-- train/test
		train(trainData)
		test(testData)
	end
end
-------------------------------------------------------------------------------

