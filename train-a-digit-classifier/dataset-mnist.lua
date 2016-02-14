require 'torch'
require 'paths'

mnist = {}

mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.path_dataset = 'mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

function mnist.download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.loadTrainSet(maxLoad, geometry)
   return mnist.loadDataset(mnist.path_trainset, maxLoad, geometry)
end

function mnist.loadTestSet(maxLoad, geometry)
   return mnist.loadDataset(mnist.path_testset, maxLoad, geometry)
end

function mnist.loadDataset(fileName, maxLoad)
   mnist.download()

   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local labels = f.labels

   local nExample = f.data:size(1)
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
   data = data[{{1,nExample},{},{},{}}]
   labels = labels[{{1,nExample}}]
   print('<mnist> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels
	dataset.encode_one_hot_targets = false
	dataset.encode_binary_targets = false

   function dataset:normalize(mean_, std_)
      local mean = mean_ or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)

		-- Several pixels are always 0 in mnist
		-- Only normalize those pixels that have some variance
		local stdinv = std:squeeze():clone()
		for i=1,std:size(2) do
			if std[1][i] > 0 then
				stdinv[i] = 1/std[1][i]
			end
		end

      for i=1,data:size(1) do
         data[i]:add(-mean[1])
			data[i]:cmul(stdinv)
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

	function dataset:makeOneHotEncoding() 
		if encode_one_hot_targets then
			print("Targets are coded as one hot vectors already!")
			return
		end
		encode_one_hot_targets = true
		local num_classes = 0
		for i =1,labels:size(1) do
			if tonumber(labels[i]) > num_classes then
				num_classes = tonumber(labels[i])
			end
		end
		self.targets = torch.FloatTensor(data:size(1), num_classes)
		for i=1,data:size(1) do
			self.targets[i]:zero()
			self.targets[i][tonumber(labels[i])] = 1
		end
	end

	function dataset:makeBinaryEncoding() 
		if encode_binary_targets then
			print("Targets are coded as binary vectors already!")
			return
		end
		encode_binary_targets = true
		local num_classes = 0
		for i =1,labels:size(1) do
			if tonumber(labels[i]) > num_classes then
				num_classes = tonumber(labels[i])
			end
		end
		local ndim = math.ceil(math.log(num_classes)/math.log(2))
		self.btargets = torch.FloatTensor(data:size(1), ndim)
		for i=1,data:size(1) do
			self.btargets[i]:zero()
			local pow2 = 1
			local numeric_label = tonumber(labels[i])
			for j=1,ndim do
				if numeric_label % math.pow(2, pow2) >= math.pow(2, pow2-1) then
					self.btargets[i][j] = 1
				end
				pow2 = pow2 + 1
			end
		end
	end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
                                       return example
   end})

   return dataset
end

