import numpy
from scipy.optimize import minimize
import warnings
try:
    from simplenlopt import minimize as minimize_nlopt
    simplenlopt_available = True
except IOError:
    simplenlopt_available = False

class OptimizeResult(dict):
    r""" Represents the optimization result.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    message : str
        Description of the cause of the termination.
    fun : float
        Value of the objective function.
    nfev : int
        Number of evaluations of the objective function.
    ntunnel: int
        Total umber of tunneling phases performed
    ntunnel_success: int
        Number of successfull tunneling phases performed
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

def ampgo(objfun, bounds, args=(), x0 = 'random', jac = None, hess = None, hessp = None, 
          local_minimizer='l-bfgs-b', local_minimizer_options={}, maxfunevals=None,
          totaliter=20, maxiter=5, glbtol=1e-5, eps1=0.02, eps2=0.1, tabulistsize=5,
          tabustrategy='farthest', fmin=-numpy.inf, disp=None):
    """
    Finds the global minimum of a function using the AMPGO (Adaptive Memory Programming for
    Global Optimization) algorithm. 

    Parameters
    ----------
    objfun : callable 
        The objective function to be minimized. Must be in the form ``objfun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``,
        defining the finite lower and upper bounds for the optimizing argument of ``fun``. 
        It is required to have ``len(bounds) == len(x)``.
    args : tuple, optional, default ()
        Further arguments to describe the objective function
    x0 : {ndarray, 'random'}, optional, default 'random'
        Starting guess for the decision variable. If 'random', picks a random point in the feasible region
    method : string, optional, default 'auto'
    jac : {callable,  '2-point', '3-point', True}, optional, default None
        If callable, must be in the form ``jac(x, *args)``, where ``x`` is the argument
        in the form of a 1-D array and args is a tuple of any additional fixed parameters 
        needed to completely specify the function.\n
        If '2-point' will use forward difference to approximate the gradient.\n
        If '3-point' will use central difference to approximate the gradient.\n
        If True, objfun must return a tuple of both objective function and an array holding the gradient information
    :param `objfun`: Function to be optimized, in the form ``f(x, *args)``.
    :type `objfun`: callable
    :param `args`: Additional arguments passed to `objfun`.
    :type `args`: tuple
    :param `local`: The local minimization method (e.g. ``"L-BFGS-B"``). It can be one of the available
     `scipy` local solvers or `OpenOpt` solvers.
    :type `local`: string
    :param `bounds`: A list of tuples specifying the lower and upper bound for each independent variable
     [(`xl0`, `xu0`), (`xl1`, `xu1`), ...]
    :type `bounds`: list
    :param `maxfunevals`: The maximum number of function evaluations allowed.
    :type `maxfunevals`: integer
    :param `totaliter`: The maximum number of global iterations allowed.
    :type `totaliter`: integer
    :param `maxiter`: The maximum number of `Tabu Tunnelling` iterations allowed during each global iteration.
    :type `maxiter`: integer
    :param `glbtol`: The optimization will stop if the absolute difference between the current minimum objective
     function value and the provided global optimum (`fmin`) is less than `glbtol`.
    :type `glbtol`: float
    :param `eps1`: A constant used to define an aspiration value for the objective function during the Tunnelling phase.
    :type `eps1`: float
    :param `eps2`: Perturbation factor used to move away from the latest local minimum at the start of a Tunnelling phase.
    :type `eps2`: float
    :param `tabulistsize`: The size of the tabu search list (a circular list).
    :type `tabulistsize`: integer
    :param `tabustrategy`: The strategy to use when the size of the tabu list exceeds `tabulistsize`. It can be
     'oldest' to drop the oldest point from the tabu list or 'farthest' to drop the element farthest from
     the last local minimum found.
    :type `tabustrategy`: string
    :param `fmin`: If known, the objective function global optimum value.
    :type `fmin`: float
    :param `disp`: If zero or defaulted, then no output is printed on screen. If a positive number, then status
     messages are printed.
    :type `disp`: integer
 
    :returns: A tuple of 5 elements, in the following order:

     1. **best_x** (`array_like`): the estimated position of the global minimum.
     2. **best_f** (`float`): the value of `objfun` at the minimum.
     3. **evaluations** (`integer`): the number of function evaluations.
     4. **msg** (`string`): a message describes the cause of the termination.
     5. **tunnel_info** (`tuple`): a tuple containing the total number of Tunnelling phases performed and the
        successful ones.

    :rtype: `tuple`

    The detailed implementation of AMPGO is described in the paper 
    "Adaptive Memory Programming for Constrained Global Optimization" located here:

    http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf

    Copyright 2014 Andrea Gavana
    """
   
#if local not in SCIPY_LOCAL_SOLVERS + OPENOPT_LOCAL_SOLVERS:
#        raise Exception('Invalid local solver selected: %s'%local)

#    if local in SCIPY_LOCAL_SOLVERS and not SCIPY:
#        raise Exception('The selected solver %s is not available as there is no scipy installation'%local)

#    if local in OPENOPT_LOCAL_SOLVERS and not OPENOPT:
#        raise Exception('The selected solver %s is not available as there is no OpenOpt installation'%local)
    
    if x0 == 'random':
        lower_bounds = numpy.asarray([b[0] for b in bounds])
        upper_bounds = numpy.asarray([b[1] for b in bounds])

        x0 =  lower_bounds + (upper_bounds - lower_bounds) * numpy.random.rand(len(lower_bounds))
    
    else:
        x0 = numpy.atleast_1d(x0)

    n = len(x0)

    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    low = [0]*n
    up = [0]*n
    for i in range(n):
        if bounds[i] is None:
            l, u = -numpy.inf, numpy.inf
        else:
            l, u = bounds[i]
            if l is None:
                low[i] = -numpy.inf
            else:
                low[i] = l
            if u is None:
                up[i] = numpy.inf
            else:
                up[i] = u

    if maxfunevals is None:
        maxfunevals = max(100, 10*len(x0))

    if tabulistsize < 1:
        raise ValueError('Invalid tabulistsize specified: %s. It should be an integer greater than zero.'%tabulistsize)
    if tabustrategy not in ['oldest', 'farthest']:
        raise ValueError('Invalid tabustrategy specified: %s. It must be one of "oldest" or "farthest"'%tabustrategy)

    iprint = 50
    if disp is None or disp <= 0:
        disp = 0
        iprint = -1

    low = numpy.asarray(low)
    up = numpy.asarray(up)

    tabulist = []
    best_f = numpy.inf
    best_x = x0
    
    global_iter = 0
    all_tunnel = success_tunnel = 0
    evaluations = 0

    if glbtol < 1e-8:
        local_tol = glbtol
    else:
        local_tol = 1e-8

    while 1:

        if disp > 0:
            print('\n')
            print('='*72)
            print('Starting MINIMIZATION Phase %-3d'%(global_iter+1))
            print('='*72)

        #else:
        '''
        options = {'maxiter': max(1, maxfunevals), 'disp': disp}
        if minimizer_kwargs is None:
            minimizer_kwargs={}
            minimizer_kwargs['method'] = 'L-BFGS-B'
        elif "method" in minimizer_kwargs == False:
            minimizer_kwargs['method'] = 'L-BFGS-B'
        '''
        if local_minimizer[:5] != 'nlopt':
            if local_minimizer_options is not None:
                local_minimizer_options['maxiter'] = max(1, maxfunevals)
            else:
                local_minimizer_options={}
                local_minimizer_options['maxiter'] = max(1, maxfunevals)

            res = minimize(objfun, x0, args=args, method = local_minimizer, 
                    jac = jac, hess = hess, hessp = hessp,
                    options = local_minimizer_options, bounds=bounds)

        elif local_minimizer[:5] == 'nlopt' and simplenlopt_available:
            if hess or hessp:
                warnings.warn("Hessian information given but nlopt solvers do not use them.", RuntimeWarning)

            res = minimize_nlopt(objfun, x0, args=args, jac = jac, bounds = bounds,
                method = local_minimizer[6:], **local_minimizer_options)
        
        elif local_minimizer[:5] == 'nlopt' and not simplenlopt_available:

            raise ValueError("simplenlopt not installed. Nlopt solvers not available!")

        else:

            raise ValueError("") 

        xf, yf, num_fun = res['x'], res['fun'], res['nfev']
        
        maxfunevals -= num_fun
        evaluations += num_fun

        if yf < best_f:
            best_f = yf
            best_x = xf

        if disp > 0:
            print('\n\n ==> Reached local minimum: %s\n'%yf)
        
        if best_f < fmin + glbtol:
            if disp > 0:
                print('='*72)
            return OptimizeResult(
                    x=best_x,
                    fun=best_f,
                    success = True,
                    message='Optimization terminated successfully',
                    nfev=evaluations,
                    ntunnel = all_tunnel,
                    ntunnel_succes = success_tunnel)
                    #best_x, best_f, evaluations, 'Optimization terminated successfully', (all_tunnel, success_tunnel)
        if maxfunevals <= 0:
            if disp > 0:
                print('='*72)
            return OptimizeResult(
                x=best_x,
                fun=best_f,
                success = False,
                message='Maximum number of function evaluations exceeded',
                nfev=evaluations,
                ntunnel = all_tunnel,
                ntunnel_succes = success_tunnel)
            #return best_x, best_f, evaluations, 'Maximum number of function evaluations exceeded', (all_tunnel, success_tunnel)

        tabulist = drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy)
        tabulist.append(xf)

        i = improve = 0

        while i < maxiter and improve == 0:

            if disp > 0:
                print('-'*72)
                print('Starting TUNNELLING   Phase (%3d-%3d)'%(global_iter+1, i+1))
                print('-'*72)

            all_tunnel += 1
            
            r = numpy.random.uniform(-1.0, 1.0, size=(n, ))
            beta = eps2*numpy.linalg.norm(xf)/numpy.linalg.norm(r)
            
            if numpy.abs(beta) < 1e-8:
                beta = eps2
                
            x0  = xf + beta*r

            x0 = numpy.where(x0 < low, low, x0)
            x0 = numpy.where(x0 > up , up , x0)

            aspiration = best_f - eps1*(1.0 + numpy.abs(best_f))

            tunnel_args = tuple([objfun, aspiration, tabulist] + list(args))


            #res = minimize(tunnel, x0, args=tunnel_args, method=local, bounds=bounds, tol=local_tol, options=options)
            if local_minimizer[:5] != 'nlopt':
                if local_minimizer_options is not None:
                    local_minimizer_options['maxiter'] = max(1, maxfunevals)
                else:
                    local_minimizer_options={}
                    local_minimizer_options['maxiter'] = max(1, maxfunevals)

                res = minimize(tunnel, x0, args=tunnel_args, method = local_minimizer, options = local_minimizer_options, bounds=bounds)

            elif local_minimizer[:5] == 'nlopt' and simplenlopt_available:

                res = minimize_nlopt(tunnel, x0, args=tunnel_args, bounds = bounds,
                    method = local_minimizer[6:], **local_minimizer_options)

            elif local_minimizer[:5] == 'nlopt' and not simplenlopt_available:

                raise ValueError("simplenlopt not installed. Nlopt solvers not available!")

            else:

                raise ValueError("")

            xf, yf, num_fun = res['x'], res['fun'], res['nfev']

            maxfunevals -= num_fun
            evaluations += num_fun

            yf = inverse_tunnel(xf, yf, aspiration, tabulist)

            if yf <= best_f + glbtol:
                oldf = best_f
                best_f = yf
                best_x = xf
                improve = 1
                success_tunnel += 1

                if disp > 0:
                    print('\n\n ==> Successful tunnelling phase. Reached local minimum: %s < %s\n'%(yf, oldf))

            if best_f < fmin + glbtol:
                return OptimizeResult(
                    x=best_x,
                    fun=best_f,
                    success = True,
                    message='Optimization terminated successfully',
                    nfev=evaluations,
                    ntunnel = all_tunnel,
                    ntunnel_succes = success_tunnel)

            i += 1
                        
            if maxfunevals <= 0:
                return OptimizeResult(
                    x=best_x,
                    fun=best_f,
                    success = False,
                    message='Maximum number of function evaluations exceeded',
                    nfev=evaluations,
                    ntunnel = all_tunnel,
                    ntunnel_succes = success_tunnel)

            tabulist = drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy)
            tabulist.append(xf)

        if disp > 0:
            print('='*72)

        global_iter += 1
        x0 = xf.copy()

        if global_iter >= totaliter:
            return OptimizeResult(
                x=best_x,
                fun=best_f,
                success = False,
                message='Maximum number of global iterations exceeded',
                nfev=evaluations,
                ntunnel = all_tunnel,
                ntunnel_succes = success_tunnel)

        if best_f < fmin + glbtol:
            return OptimizeResult(
                x=best_x,
                fun=best_f,
                success = True,
                message='Optimization terminated successfully',
                nfev=evaluations,
                ntunnel = all_tunnel,
                ntunnel_succes = success_tunnel)

def drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy):

    if len(tabulist) < tabulistsize:
        return tabulist
    
    if tabustrategy == 'oldest':
        tabulist.pop(0)
    else:
        distance = numpy.sqrt(numpy.sum((tabulist-xf)**2, axis=1))
        index = numpy.argmax(distance)
        tabulist.pop(index)

    return tabulist


def tunnel(x0, *args):

    objfun, aspiration, tabulist = args[0:3]

    fun_args = ()    
    if len(args) > 3:
        fun_args = tuple(args[3:])

    numerator = (objfun(x0, *fun_args) - aspiration)**2
    denominator = 1.0

    for tabu in tabulist:
        denominator = denominator*numpy.sqrt(numpy.sum((x0 - tabu)**2))

    ytf = numerator/denominator

    return ytf


def inverse_tunnel(xtf, ytf, aspiration, tabulist):

    denominator = 1.0

    for tabu in tabulist:
        denominator = denominator*numpy.sqrt(numpy.sum((xtf - tabu)**2))
    
    yf = aspiration + numpy.sqrt(ytf*denominator)
    return yf

'''
from scipy.optimize import rosen, rosen_der, rosen_hess
import numpy as np
x0=np.array([10.8, 0.7])
ampgo_res=ampgo(rosen, [(-1, 1.5), (-1, 1.5)], jac=rosen_der, hess = rosen_hess, local_minimizer='nlopt_lbfgs')#, local_minimizer_options={'gtol': 1e-3})
print(ampgo_res.x)

if __name__ == '__main__':

    import os
    import go_benchmark

    os.system('cls')

    for tests in ['Bird']:
        
        klass = getattr(go_benchmark, tests)()

        x0 = klass.generator()
        fmin = klass.fglob
        bounds = klass.bounds
        tolfun = 1e-6

        xf, yf, fun_evals, msg, tt = AMPGO(klass.evaluator, x0, args=(), local='L-BFGS-B', bounds=bounds,
                                       maxfunevals=20000, totaliter=2000, maxiter=5, eps1=0.02, eps2=0.1,
                                       tabulistsize=5, tabustrategy='farthest', fmin=fmin, disp=1, glbtol=tolfun)

        xb = numpy.asarray(klass.global_optimum)
        if xb.ndim == 2:
            xb = xb[0, :]
            
        print('\n\n')
        print('F_glob :', klass.evaluator(xb))
        print('F_best :', yf)
        print('X_best :', xf)
        print('F_evals:', fun_evals)
        print('Message:', msg)
        print('Tunnels:', tt)
'''