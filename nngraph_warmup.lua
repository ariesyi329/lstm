require 'torch'
require 'nngraph'
require 'nn'

--the function
function some_function(opt)
    local x = nn.Identity()()
    local y = nn.Identity()()
    local z = nn.Identity()()
    
    local left_term = nn.Square()(nn.Tanh()(nn.Linear(opt.x_size, opt.output_size)(x)))
    local right_term = nn.Square()(nn.Sigmoid()(nn.Linear(opt.y_size, opt.output_size)(y)))
    local mul = nn.CMulTable()({left_term, right_term})
    local a = nn.CAddTable()({mul, z})
    
    return nn.gModule({x,y,z},{a})
end

function propogation(module, input, goutput)
	forward_output = module:forward(input)
	backward_output = module:backward(input, goutput)
	return forward_output, backward_output
end

-- define variable sizes.
opt = {}
opt.x_size = 4
opt.y_size = 5
opt.output_size = 2

-- randomly choose x, y, z.
x = torch.rand(opt.x_size)
y = torch.rand(opt.y_size)
z = torch.rand(opt.output_size)
input = {x,y,z}
gOutput = torch.ones(opt.output_size)

some_function = some_function(opt)
forward_output, backward_output = propogation(some_function, input, gOutput)

--print results
print("input x is:")
print(x)
print("input y is:")
print(y)
print("input z is:")
print(z)
print("forward output is:")
print(forward_output)
print("gradient of output is:")
print(gOutput)
print("backward output of x is:")
print(backward_output[1])
print("backward output of y is:")
print(backward_output[2])
print("backward output of z is:")
print(backward_output[3])



