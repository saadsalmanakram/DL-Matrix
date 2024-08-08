import torch
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Define a tensor
x = torch.tensor([-1.0, 2.0, -3.0], requires_grad=True)

# Apply custom ReLU
relu = MyReLU.apply
y = relu(x)
y.sum().backward()

# Access gradients
print(x.grad)  # tensor([0., 1., 0.])
