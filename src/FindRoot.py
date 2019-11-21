#!env python3

# pylint: disable=C0303,W0511
'''
A personal academic exercise code for elementary numerical analysis to study and 
explore well-known Root-Finder algorithms for continuous, univariate functions.

For production code, please consider using SciPy library.

TODO:
  1) Profile code to optimize? 
  2) Build unit test cases. (include coverage test)
  3) Think about how to adapt these methods to multivariate functions

Alex Lim - mathvfx.github.io - 2019
'''
import array

from typing import Callable


class FindRoot:
    '''A root-finding class for continuous, real-valued function of single
    variable using several functional iteration schemes.

    Note that it is user's responsibility to provide efficient mathematics
    function and its derivatives where necessary-- this class does not
    perform any symbolic operations.

    Requirements
    ------------
        1) All mathematical functions must be a continuous function over its
           domain.
    '''


    def __init__(self, math_expr: Callable[[float], float]):
        '''FindRoot class constructor.

        Parameters
        ----------
        math_expr : callable, required
                    A callable Mathematics function for which we will attempt to
                    find solution for the form f(x) = 0 for all real-value x.
                    For example: exp(x**2) + x**3 - 10 = 0, for which we want to
                    find numerical solution for x.
        '''
        if not callable(math_expr):
            raise TypeError(" >> 'math_expr' must be callable math function")
        self._math_func = math_expr
        self.Configure()


    @staticmethod
    def _sgn(val: float) -> int:
        '''Return signum value of the value.
        For an efficient signum function, consider using numpy.sign()
        '''
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0


    def Bisection(self, bracket: tuple, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics, real-valued
        function of the form f(x) = 0 by continuously halving the given interval
        (bracketing) until value is within error tolerance.

        Consider scipy.optimize.bisect() for production code.

        Parameters
        ----------
        bracket : 2-float, required.
                  Interval (a,b) with a <= b for which a real-valued solution 
                  may be located.
        sol_array : dictionary, optional.
                    A "pass-by-reference" way of retriving interim solutions
                    for later analysis.

        Returns
        ----------
        result : approximate solution to within error tolerance for mathmatics
                 function provided, if exists. Otherwise, None is returned.
        '''
        # Pre-check parameter type
        if not isinstance(bracket, (tuple, list)):
            raise TypeError(">> 'bracket' must be a 2-tuple.")
        if not isinstance(sol_array, dict) and sol_array is not None:
            raise TypeError(">> 'sol_array' must be a dict.") 

        # Initialize
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func

        a = bracket[0]
        b = bracket[1]
        if a > b:
            raise ValueError(" >> bracket (a,b) must satisfy a <= b")

        # Initial condition check
        sgn_FA = self._sgn(expr(a)) # Sign of expr evaluated at left interval
        sgn_FB = self._sgn(expr(b)) # SIgn of expr evaluated at right interval
        if sgn_FA * sgn_FB > 0:
            # FA and FB having same sign imply expr never crosses zero and thus
            # no solution within initial interval. This behavior is guaranteed
            # by Intermediate Value Theorem for continuous function.
            print(f">>> No real-value solution found on [{a}, {b}].")
            return None

        # Perform bisection search until max limit or within tolerance
        for i in range(1, self._max_iter):
            mid_pt = (b-a) / 2.0
            curr_pt = a + mid_pt
            sgn_FP = self._sgn(expr(curr_pt))

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(mid_pt)

            if mid_pt < self._tol or sgn_FP == 0:
                # Found solution
                print(f">> Bisection: {curr_pt} in {i} iteration.")
                if sol_array is not None:
                    sol_array["bisection"] = iter_array
                    sol_array["error"] = err_array
                return curr_pt

            # Update intervals for the next iteration
            if sgn_FA * sgn_FP > 0:
                a = curr_pt
            else:
                b = curr_pt

        # No solution
        print(f">>> Maximum reached. Either no solution or slow convergence.")
        print(f"    Last approximation: {curr_pt} after {i} attempt.")
        if sol_array is not None:
            sol_array["bisection"] = iter_array
            sol_array["error"] = err_array
        return None


    def FixedPoint(self, init_pt: float, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form g(p) = p or f(p) - p = 0 by functional iteration technique.

        Consider using scipy.optimize.fixed_point() for production code.

        Parameters
        ----------
        init_pt : float, required.
                  A guess of initial approximation point that should be as
                  close as possible to actual solution.
        sol_array : dictionary, optional.
                    A "pass-by-reference" way of retriving interim solutions
                    for later analysis.

        Returns
        ----------
        result : approximate solution to within error tolerance for mathmatics
                 function provided, if exists. Otherwise, None is returned.
        '''
        # Pre-check parameter type
        if not isinstance(init_pt, (int, float)):
            raise TypeError(">> 'init_pt' must be a numeric.")
        if not isinstance(sol_array, dict) and sol_array is not None:
            raise TypeError(">> 'sol_array' must be a dict.") 

        # Initialize
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func
        p0 = init_pt

        # Perform fixed-point iteration
        for i in range(1, self._max_iter):
            curr_pt = expr(p0)
            abs_diff = abs(curr_pt-p0)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(abs_diff)

            if abs_diff < self._tol:
                # Found a solution within erro tolerance
                print(f">> Fixed-Point: {curr_pt} after {i} iterations.")
                if sol_array is not None:
                    sol_array["fixed_point"] = iter_array
                    sol_array["error"] = err_array
                return curr_pt

            # Update for next iterations
            p0 = curr_pt

        # No solution
        print(f">>> FixedPoint: Maximum reached. Last evaluation: {curr_pt}")
        if sol_array is not None:
            sol_array["fixed_point"] = iter_array
            sol_array["error"] = err_array
        return None


    def Muller(self, init_pt: tuple, sol_array: dict = None) -> float:
        '''Perform iterative solution search to univariate mathematics function 
        of the form f(x) = 0 given three sequential initial approximation points.
        Muller's Method can find complex-valued roots.

        Consider scipy.optimize.newton for production code.

        Parameters
        ----------
        init_pt : 3-float, required.
                  A guess of initial approximation points upon which roots may 
                  exist. Closer approximation will provide faster convergence.
        sol_array : dictionary, optional.
                    A "pass-by-reference" way of retriving interim solutions
                    for later analysis. sol_array for Muller method will contain
                    both real array and imaginary array.

        Returns
        ----------
        result : approximate solution to within error tolerance for mathmatics
                 function provided, if exists. Otherwise, None is returned.

        Reference
        ----------
           Burden and Faires, NUMERICAL ANALYSIS(10e), pg97
        '''
        # Pre-check parameter type
        if not isinstance(init_pt, (tuple, list)):
            raise TypeError(">> 'init_pt' must be a 3-tuple of numerics.")
        if not isinstance(sol_array, dict) and sol_array is not None:
            raise TypeError(">> 'sol_array' must be a dict.") 

        # Initialize
        # pylint: disable=R0914
        iter_array = array.array(self._sol_array_dtype) # Store real part
        iter_array_im = array.array(self._sol_array_dtype) # Store imaginary part
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func
        p0 = init_pt[0]
        p1 = init_pt[1]
        p2 = init_pt[2]
        h1 = p1 - p0 # Can be thought of as "differential"
        h2 = p2 - p1 # Can be thought of as "differential"
        delta1 = (expr(p1)-expr(p0)) / h1 # Thought of as secant
        delta2 = (expr(p2)-expr(p1)) / h2 # Thought of as secant
        d = (delta2-delta1) / (h2+h1)

        # Perform Muller iterative search
        for i in range(1, self._max_iter):
            b = delta2 + d*h2
            FP2 = expr(p2)
            discrm = (b*b - 4.0*d*FP2)**0.5 # Check for complex val
            if abs(b-discrm) < abs(b+discrm):
                h = -2.0*FP2 / (b+discrm)
            else:
                h = -2.0*FP2 / (b-discrm)
            curr_pt = p2 + h

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt.real)
                if isinstance(curr_pt, complex):
                    iter_array_im.append(curr_pt.imag)
                err_array.append(abs(h))

            if abs(h) < self._tol:
                # Found a solution within erro tolerance
                print(f">> Muller: {curr_pt} in {i} iterations.")
                if sol_array is not None:
                    sol_array["muller"] = iter_array
                    if isinstance(curr_pt, complex):
                        sol_array["muller_im"] = iter_array_im
                    sol_array["error"] = err_array
                return curr_pt

            # Update for next iteration
            p0, p1, p2 = p1, p2, curr_pt
            h1 = p1 - p0
            h2 = p2 - p1
            delta1 = (expr(p1)-expr(p0)) / h1
            delta2 = (expr(p2)-expr(p1)) / h2
            d = (delta2-delta1) / (h2+h1)

        # No solution
        print(f">>> Muller: Maximum reached. Last evaluation: {curr_pt}")
        if sol_array is not None:
            sol_array["muller"] = iter_array
            if isinstance(curr_pt, complex):
                sol_array["muller_im"] = iter_array_im
            sol_array["error"] = err_array
        return None


    def NewtonRaphson(
            self,
            deriv: callable,
            init_pt: float,
            sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form f(x) = 0 given an initial approximation point and first-derivative
        function.

        Consider scipy.optimize.newton() for production code.

        Parameters
        ----------
        deriv : callable, required.
                A first derivative function of the mathematics function
                math_expr supplied to FindRoot constructor.
        init_pt : float, required.
                  A guess of initial approximation point that should be as
                  close as possible to actual solution.
        sol_array : dictionary, optional.
                    A "pass-by-reference" way of retriving interim solutions
                    for later analysis.

        Returns
        ----------
        result : approximate solution to within error tolerance for mathmatics
                 function provided, if exists. Otherwise, None is returned.
        '''
        # Pre-check parameter type
        if not callable(deriv):
            raise TypeError(" >> 'deriv' must be callable derivative function.")
        if not isinstance(init_pt, (int, float)):
            raise TypeError(">> 'init_pt' must be a numeric.")
        if not isinstance(sol_array, dict) and sol_array is not None:
            raise TypeError(">> 'sol_array' must be a dict.") 

        # Initialize
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func
        p0 = init_pt

        # Perform Newton-Raphson iterative search
        for i in range(1, self._max_iter):
            curr_pt = p0 - expr(p0)/deriv(p0) # Possible zero-division.
            abs_diff = abs(curr_pt-p0)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(abs_diff)

            if abs_diff < self._tol:
                # Found a solution within erro tolerance
                print(f">> Newton-Raphson: {curr_pt} in {i} iteration.")
                if sol_array is not None:
                    sol_array["newton_raphson"] = iter_array
                    sol_array["error"] = err_array
                return curr_pt

            # Update for next iteration
            p0 = curr_pt

        # No solution
        print(f">>> Newton: Maximum reached. Last evaluation: {curr_pt}")
        if sol_array is not None:
            sol_array["newton_raphson"] = iter_array
            sol_array["error"] = err_array
        return None


    def Secant(self, init_pt: tuple, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form f(x) = 0 given using two initial approximation point (which forms
        a secant line to the function. This is a modified Newton-Raphson Method
        in that derivative function is not required. However, Secant method
        will converge slower than Newton-Raphson Method.

        Consider scipy.optimize.newton() for production code. Instead of fprime,
        we can replace derivative with secant definition.

        Parameters
        ----------
        init_pt : 2-float, required.
                  Two initial points to construct "secant line" in place of
                  "tangent line" (derivative).
        sol_array : dictionary, optional.
                    A "pass-by-reference" way of retriving interim solutions
                    for later analysis.

        Returns
        ----------
        result : approximate solution to within error tolerance for mathmatics
                 function provided, if exists. Otherwise, None is returned.
        '''
        # Pre-check parameter type
        if not isinstance(init_pt, (tuple, list)):
            raise TypeError(">> 'init_pt' must be a 2-tuple of numerics.")
        if not isinstance(sol_array, dict) and sol_array is not None:
            raise TypeError(">> 'sol_array' must be a dict.") 

        # Initialize
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func
        p0 = init_pt[0]
        p1 = init_pt[1]
        q0 = expr(p0)
        q1 = expr(p1)

        # Perform Secant iterative search
        for i in range(1, self._max_iter):
            curr_pt = p1 - q1*(p1-p0) / (q1-q0) # Possible zero-division.
            abs_diff = abs(curr_pt-p1)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(abs_diff)

            if abs_diff < self._tol:
                # Found a solution within erro tolerance
                print(f">> Secant: {curr_pt} in {i} iterations.")
                if sol_array is not None:
                    sol_array["secant"] = iter_array
                    sol_array["error"] = err_array
                return curr_pt

            # Update for next iteration
            p0 = p1
            p1 = curr_pt
            q0 = q1
            q1 = expr(curr_pt)

        # No Solution 
        print(f">>> Secant: Maximum reached. Last evaluation: {curr_pt}")
        if sol_array is not None:
            sol_array["secant"] = iter_array
            sol_array["error"] = err_array
        return None


    def Steffensen(self, init_pt: float, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form g(p) = p or x - f(x) = 0 given an initial approximation point. 

        Steffensen Method is a modified Aitken Delta-Squared Method to ensure
        quadratic convergence, if conditions are met for the mathematics 
        function math_expr. Chiefly, if derivative g'(p) is not 1, and that 
        function g has continuous third derivative around a neighborhood of 
        solution p, then Steffensen method will guarantee quadratic order of 
        convergence.

        For production code, consider scipy.optimize.fixed_point() with method
        set to 'del2'

        Parameters
        ----------
        init_pt : float, required.
                  A guess of initial approximation point that should be as 
                  close as possible to actual solution.
        sol_array : dictionary, optional.
                    A "pass-by-reference" way of retriving interim solutions
                    for later analysis.

        Returns
        ----------
        result : approximate solution to within error tolerance for mathmatics
                 function provided, if exists. Otherwise, None is returned.
        '''
        # Pre-check parameter type
        if not isinstance(init_pt, (int, float)):
            raise TypeError(">> 'init_pt' must be a numeric.")
        if not isinstance(sol_array, dict) and sol_array is not None:
            raise TypeError(">> 'sol_array' must be a dict.") 

        # Initialize
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func
        p0 = init_pt

        # Perform Steffensen iterative search
        for i in range(1, self._max_iter):
            p1 = expr(p0)
            p2 = expr(p1)
            curr_pt = p0 - (p1-p0)**2 / (p2 - 2.0*p1 + p0) # Potential 0-div
            abs_diff = abs(curr_pt-p0)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                if sol_array is not None:
                    iter_array.append(curr_pt)
                    err_array.append(abs_diff)

            if abs_diff < self._tol:
                # Found a solution within erro tolerance
                print(f">> Steffensen: {curr_pt} in {i} iterations.")
                if sol_array is not None:
                    sol_array["steffensen"] = iter_array
                    sol_array["error"] = err_array
                return curr_pt

            # Update for next iteration
            p0 = curr_pt

        # No Solution
        print(f">>> Steffensen: Maximum reached. Last evaluation: {curr_pt}")
        if sol_array is not None:
            sol_array["steffensen"] = iter_array
            sol_array["error"] = err_array
        return None


    def Configure(
            self,
            tol: float = 1e-6,
            max_iter: int = 1000,
            sol_array_dtype: str = 'd'):
        '''Configuration setting for FindRoot object.

        Parameters
        ----------
        tol : float, optional.
              A stopping condition for acceptable error tolerance to our 
              numerical method.
        max_iter: int, optional.
                  Maximum number of iteration for which to refine our functional 
                  iteration will stop and assume no solution found. 
        sol_array_dtype: char, optional.
                         Array data type for sol_array.
        '''
        self._tol = tol
        self._max_iter = max_iter
        self._sol_array_dtype = sol_array_dtype


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # Only for testing
    import scipy.optimize as optimize

    print("\n\tROOT-FINDING EXAMPLES AND TESTING\n")

    # f1 has two real roots within [1,3], and two complex root at [-0.5, 0.5]
    #f1s = lambda x: x**4 - 3*x**3 + x*x + x + 1 # Simplified math
    f1 = lambda x: x*(x*(x*(x-3) + 1) + 1) + 1 # Same as f1s, less arithmetics
    f1d = lambda x: x*(x*(4*x - 9) + 2) + 1 # derivative of f1 at x

    # Array storage:
    store_bisection = dict()
    store_newton = dict()
    store_secant = dict()
    store_muller1 = dict()
    store_muller2 = dict()

    # Search for roots of the form f(x) = 0
    # By Fundamental Theorem of Algebra, we can expect f1 to have 4 roots.
    root1 = FindRoot(f1)
    root1.Configure(max_iter=50)
    root1.Bisection((1, 2), store_bisection) 
    root1.Bisection((2, 3))  
    root1.Bisection((-1, 1)) # Can't do complex root
    print()
    root1.NewtonRaphson(f1d, 1, store_newton)
    root1.NewtonRaphson(f1d, 2)
    root1.NewtonRaphson(f1d, -1) # Can't find complex root
    print()
    root1.Secant((1, 1.2), store_secant)
    root1.Secant((2, 2.2))
    root1.Secant((-2, -1)) # Can't find complex root
    print()
    root1.Muller((1, 1.5, 2), store_muller1)
    root1.Muller((2, 2.5, 3))
    root1.Muller((-1, 0, 1), store_muller2) # Can find complex root!

    # Search for roots of the form g(x) = x (a.k.a. fixed point)
    print("\n\nExpressing f1 in fixed-point form:") 

    # Express f1 in g(x)=x form. 
    # Be careful that domain of f1 may have changed and might not converge
    f2 = lambda x: (-(x+1) / (x*x - 3*x + 1))**0.5
    root2 = FindRoot(f2)
    root2.FixedPoint(1)
    print("\tVerify FixedPoint with SciPy result: ", end="")
    print(optimize.fixed_point(f2, 1, xtol=1e-6, method="iteration"))

    root2.Steffensen(1)
    print("\tVerify Steffensen with SciPy result: ", end="")
    print(optimize.fixed_point(f2, 1, xtol=1e-6, method="del2"))
    
    # Without changing form, FixedPoint and Steffensen won't give correct result
    print("\nIncorrect result when using fixed-point on f(x)=0 form:")
    root1.FixedPoint(1) 
    root1.Steffensen(0.5) # div-by-zero if init_pt=1

    # Display convergence results
    #plt.plot(store_bisection["bisection"]) 
    #plt.show()
