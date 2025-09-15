import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result

def foo_grad(x):
    x = np.asarray(x)
    return 4 * x**3

def bar(x):
    return np.prod(x)

def bar_grad(x):
    x = np.asarray(x)
    if_zeros = (x == 0)
    zero_counts = if_zeros.sum()
    if zero_counts == 0:
         return np.prod(x)/x
    grad = np.zeros_like(x)
    if zero_counts == 1:
        grad[if_zeros] =  np.prod(x[~if_zeros])
    return grad


