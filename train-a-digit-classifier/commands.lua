dofile('dataset-mnist.lua');
opt = {}
opt.encoding='one-hot'

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

dataset = mnist.loadTrainSet(10, {32, 32})
