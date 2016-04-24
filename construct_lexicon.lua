require('torch')
ptb = require('data')

local dummy_batch_size = 20

local train = ptb.traindataset(dummy_batch_size)
local valid = ptb.validdataset(dummy_batch_size)
local test = ptb.testdataset(dummy_batch_size)

local word2ind = ptb.vocab_map
local ind2word = ptb.reverse_vocab_map

print("==> saving lexicon ...")
local save_path = "./map/"
torch.save(save_path.."word2ind.t7", word2ind)
torch.save(save_path.."ind2word.t7", ind2word)
print("==> saved.")