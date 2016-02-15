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
	dataset.binary_output = false

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

	function dataset:binaryOutput(class)
		local ovector = torch.Tensor(4):zero()
		if (class % 2 >= 1) then
			ovector[1] = 1
		end
		if (class % 4 >= 2) then
			ovector[2] = 1
		end
		if (class % 8 >= 4) then
			ovector[3] = 1
		end
		if (class % 16 >= 8) then
			ovector[4] = 1
		end
		return ovector
	end

   function dataset:ohOutput(class)
		local ovector = torch.Tensor(10):zero()
		ovector[class] = 1
		return ovector
	end

   setmetatable(dataset, {__index = function(self, index)
			     	local input = self.data[index]
			     	local class = self.labels[index]
				  	if self.binary_output then
						label = binaryOutput(class)
				  	else 
					  	label = ohOutput(class)
				  	end
			     	local example = {input, label}
              	return example
   				end})

   return dataset
end
