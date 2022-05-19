import torch


def create_straight_through_function(f):
    class STFunction(torch.autograd.Function):
        """
        https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
        """
        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                return f(input)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return lambda x: STFunction.apply(x)


quantize_image_1_255 = create_straight_through_function(lambda x: torch.round(x * 255.) / 255.)
