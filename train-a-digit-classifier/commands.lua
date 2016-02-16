dofile('dataset-mnist.lua');
opt = {}
opt.encoding='binary'

function convert_to_labels(preds)
	batchSize = preds:size(1)
	local pred_labels = torch.Tensor(batchSize)
	if opt.encoding == 'binary' then
		local bmul = torch.Tensor{1, 2, 4, 8}:reshape(4, 1)
		pred_labels = preds:ge(0.5):typeAs(bmul)*bmul
	else
		_, pred_labels = preds:max(2)
	end
	return pred_labels
end

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

dataset = mnist.loadTrainSet(10, {32, 32})
