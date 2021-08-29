# ampgo
Global optimization via adaptive memory programming with a scipy.optimize like API.

## Installation

## Example: Minimizing the six-hump camelback function in ampgo
```python
import ampgo

def obj(x):
    """Six-hump camelback function"""
    x1 = x[0]
    x2 = x[1]
    f = (4 - 2.1*(x1*x1) + (x1*x1*x1*x1)/3.0)*(x1*x1) + x1*x2 + (-4 + 4*(x2*x2))*(x2*x2)
    return f

bounds = [(-5, 5), (-5, 5)]
res = ampgo.ampgo(obj, bounds)
print(res.x)
print(res.fun)
```

## History
Coded by Andrea Gavana, andrea.gavana@gmail.com. Original hosted at https://code.google.com/p/ampgo/.  Made available under the MIT licence. Usage and installation modified by Daniel Schmitz.

Differences to original version:
* Support all of SciPy's local minimizers
* Return a OptimizeResult class like SciPy's global optimizers
* Require bounds instead of starting point
* Jacobian and Hessian support
* Support all of NLopt's local minimizers (requires [simplenlopt](https://simplenlopt.readthedocs.io/en/latest/index.html))