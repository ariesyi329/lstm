require('nngraph')
require('base')
require('torch')
require('xlua')

function reset_state(state)
    local d_num_layers = num_layers
    if model_type == "lstm" then
        d_num_layers = 2 * num_layers
    end
    if model ~= nil and model.start_s ~= nil then
        for d = 1, d_num_layers do
            model.start_s[d]:zero()
        end
    end
end

function load_test_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    --print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        x[i] = word2ind[data[i]]
    end
    return x
end

function testdataset(batch_size)
    if testfn then
        local x = load_test_data(testfn)
        x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
        return x
    end
end

function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        xlua.progress(i, len-1)
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
end

--model type: lstm or gru
model_type = "gru"

--load model
print("==> Loading model ...")
local model_file = "./model/"..model_type.."/model.net"
model = torch.load(model_file)
print("==> Loading lexicon ...")
local word2ind_file = "./map/word2ind.t7"
word2ind = torch.load(word2ind_file)

--parameters
local stringx = require('pl.stringx')
local file = require('pl.file')
local ptb_path = "./data/"
testfn  = ptb_path .. "ptb.test.txt"

num_layers=2
batch_size=20
state_test = {data=testdataset(batch_size)}

run_test()

