stringx = require('pl.stringx')
require('nngraph')
require('base')
require('io')
require('torch')

--reset start state to zeros
function reset_state()
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * num_layers do
            model.start_s[d]:zero()
        end
    end
end

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
		end
	end
	return ind_seq
end

function makePrediction(ind_seq)
	local known_len = #ind_seq-1
	local pred_len = tonumber(ind_seq[1])

	--initialize start state
	reset_state(num_layers)
	g_disable_dropout(model.rnns)
	g_replace_table(model.s[0], model.start_s)

	--loop the known part
	for i = 2, known_len do
		local x = torch.DoubleTensor(batch_size):fill(ind_seq[i])
		local y = torch.DoubleTensor(batch_size):fill(ind_seq[i+1])
		_, model.s[1], _ = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
		g_replace_table(model.s[0], model.s[1])
	end

	--setup a tensor to store prediction results
	local pred_seq = torch.Tensor(pred_len)
	--make prediction
	local x = torch.DoubleTensor(batch_size):fill(ind_seq[known_len+1])
	local y = torch.DoubleTensor(batch_size):fill(ind_seq[known_len+1])
	for i = 1, pred_len do
		_, model.s[1], pred = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
		_, pred_tensor = torch.max(pred, 2)
		x = torch.DoubleTensor(batch_size):fill(pred_tensor[1][1])
		pred_seq[i] = pred_tensor[1][1]
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
	print("output...")
	print(output)
end

--load model and lexicon
print("==> Loading model ...")
local model_file = "./model/model.net"
model = torch.load(model_file)
print("==> Loading lexicon ...")
local word2ind_file = "./map/word2ind.t7"
local ind2word_file = "./map/ind2word.t7"
word2ind = torch.load(word2ind_file)
ind2word = torch.load(ind2word_file)

num_layers=2 --the same with settings in main.lua
batch_size=20 --the same with settings in main.lua

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
		outputPrediction(line, pred_seq)
	end
end