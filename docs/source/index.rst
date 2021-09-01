.. ampgo documentation master file, created by
   sphinx-quickstart on Sun Aug 29 08:09:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ampgo's documentation!
=================================
Adaptive memory programming is a powerful optimization algorithm for low-dimensional
global optimization problems. Its core is a stochastic algorithm which executes repeated 
local minimizations from heuristically chosen startpoints. 

This package implements the same API as scipy's global optimizers. All local minimizers from SciPy 
and from NLopt (requires `simplenlopt <https://simplenlopt.readthedocs.io/en/latest/index.html#>`_) can be employed for the local searches. 

Installation
=================================
.. code:: bash

   pip install ampgo

Example: Minimizing the six-hump camelback function in ampgo
==================
.. code-block:: python

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

Documentation
==================
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ampgo

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`