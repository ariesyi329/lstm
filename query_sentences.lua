stringx = require('pl.stringx')
require('nngraph')
require('base')
require('io')
require('torch')

--readline from user
function readline()
	local line = io.read("*line")
	if line == nil then error({code="EOF"}) end
	line = stringx.split(line)
	if tonumber(line[1]) == nil then error({code="init"}) end
	return line
end

--get index sequence of input words
function getIndex(line)
	local ind_seq = line:clone()
	for i = 2, #line do
		if word2ind[line[i]] == nil then
			ind_seq[i] = word2ind["<unk>"]
		else
			ind_seq[i] = word2ind[line[i]]
	return ind_seq
end

function makePrediction(ind_seq)
	local known_len = #ind_seq-1
	local pred_len = ind_seq[1]
	--TO DO!
	reset_state()
	g_disable_dropout(model.rnns)

	--loop the known part
	for i = 2, known_len do
		local x = ind_seq[i]
		local y = ind_seq[i+1]
		_, model.s[1], _ = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
		g_replace_table(model.s[0], model.s[1])
	end

	--setup a tensor to store prediction results
	local pred_seq = torch.Tensor(pred_len)
	--make prediction
	local x = ind_seq[known_len+1] --initialize x as the last word of input.
	local y = ind_seq[known_len+1] --since we won't use y further, give any number would be fine.
	for i = 1, pred_len do
		local x = ind_seq[i]
		_, model.s[1], pred = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
		_, pred_tensor = torch.max(pred, 2)
		x = pred_tensor[1][1]
		pred_seq[i] = x
		g_replace_table(model.s[0], model.s[1])
	end
	return pred_seq
end

function outputPrediction(input_seq, pred_seq)
	local output = input_seq[2]
	for i = 3, #input_seq do
		output = output .. " " .. input_seq[i]
	end
	for i = 1, pred_seq:size(1) do
		local ind = pred_seq[i]
		local word = ind2word[ind]
		output = output .. " " .. word
	end
end

print("==> Loading model ...")
local model_file = "./model/model.net"
local model = torch.load(model_file)
print("==> Loading lexicon ...")
local word2ind_file = "./map/"
local ind2word_file = "./map/"
local word2ind = torch.load(word2ind_file)
local ind2word = torch.load(ind2word_file)

while true do
	print("Query: len word1 word2 etc")
	local ok, line = pcall(readline)
	if not ok then
		if line.code == "EOF" then
			break -- end loop
		elseif line.code == "init" then
			print("Start with a number")
		else
			print(line)
			print("Failed, try again")
		end
	else
		local ind_seq = getIndex(line)
		local pred_seq = makePrediction(ind_seq)
		outputPrediction()
	end
end