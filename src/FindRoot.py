#!env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A personal academic exercise for elementary numerical analysis to study and 
# explore well-known Root-Finder algorithms for continuous, univariate 
# real-valued functions. 
# 
# For production code, please consider using SciPy library.
#
# TODO:
#   1) Documentation and docstrings
#   2) Profile code to optimize
#   3) Build unit test cases. (include coverage test)
#   4) Think about how to adapt these methods to multivariate functions
#
# Alex Lim - mathvfx.github.io - 2019
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import array
import math

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
    def __init__(self, math_expr: callable):
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


    def _sgn(self, val: float) -> int:
        '''Return signum value of the value.
        For an efficient signum function, consider using numpy.sign()
        '''
        if val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0


    def Bisection(self, bracket: tuple, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form f(x) = 0 by continuously halving the given interval (bracketing) 
        until value is within error tolerance.


        Parameters
        ----------
        bracket : 2-float, required.
                  Interval (a,b) with a <= b for which a solution may be located. 
        sol_array : dictionary, optional.
                    A "pass-by-reference" way of retriving interim solutions
                    for later analysis.

        Returns
        ----------
        result : approximate solution to within error tolerance for mathmatics
                 function provided, if solution exists. Otherwise,
                 None is returned.
        '''
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func

        a = bracket[0]
        b = bracket[1]
        if a > b:
            raise ValueError(" >> bracket (a,b) must have a <= b")

        sgnFA = self._sgn(expr(a)) # Sign of expr evaluated at left interval
        sgnFB = self._sgn(expr(b)) # SIgn of expr evaluated at right interval

        # Initial condition check
        if sgnFA * sgnFB > 0:
            # FA and FB having same sign imply expr never crosses zero and thus
            # no solution within initial interval. This behavior is guaranteed
            # by Intermediate Value Theorem for continuous function.
            print(f" >> Function might not have solution on [{a}, {b}]. \
                    Try narrowing the bracket to where zeros may exist.")
            return None

        # Perform bisection search until max limit or within tolerance
        for i in range(1, self._max_iter):
            mid_pt = (b-a) / 2.0
            curr_pt = a + mid_pt
            sgnFP = self._sgn(expr(curr_pt))

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(mid_pt)

            if mid_pt < self._tol or sgnFP == 0:
                # Found solution
                print(f" >> Approximate solution: {curr_pt} Accuracy: {mid_pt}")
                sol_array["bisection"] = iter_array
                sol_array["error"] = err_array
                return curr_pt
            elif sgnFA * sgnFP > 0: 
                a = curr_pt 
            else:
                b = curr_pt

        print(f" >> Maximum reached. Either no solution or slow convergence. \
                Last value: {curr_pt}")
        sol_array["bisection"] = iter_array
        sol_array["error"] = err_array
        return None


    def FixedPoint(self, init_pt: float, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form g(p) = p or f(p) - p = 0 by functional iteration technique.


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
                 function provided, if solution exists. Otherwise,
                 None is returned.
        '''
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func

        p0 = init_pt
        for i in range(1, self._max_iter):
            curr_pt = expr(p0)
            absDiff = abs(curr_pt-p0)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(absDiff)

            if math.isinf(curr_pt):
                print(" >> WARNING: INF detected. No solution found.")
                sol_array["fixed_point"] = iter_array
                sol_array["error"] = err_array
                return None

            if absDiff < self._tol:
                # Found a solution within erro tolerance
                print(f" >> Approximate solution: {curr_pt} Accuracy: {absDiff}")
                sol_array["fixed_point"] = iter_array
                sol_array["error"] = err_array
                return curr_pt

            p0 = curr_pt

        print(f" >> Maximum reached. Last value evaluatd: {curr_pt}")
        sol_array["fixed_point"] = iter_array
        sol_array["error"] = err_array
        return None


    def Muller(self, init_pt: tuple, sol_array: dict = None) -> float:
        '''

        Algorithm: Burden and Faires, NUMERICAL ANALYSIS(10e), pg97
        '''
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func

        # Initialize data
        p0 = init_pt[0]
        p1 = init_pt[1]
        p2 = init_pt[2]
        
        h1 = p1 - p0 # Can be thought of as "differential"
        h2 = p2 - p1 # Can be thought of as "differential"
        delta1 = (expr(p1)-expr(p0)) / h1 # Thought of as secant
        delta2 = (expr(p2)-expr(p1)) / h2 # Thought of as secant
        d = (delta2-delta1) / (h2+h1)

        for i in range(1, self._max_iter):
            b = delta2 + d*h2
            FP2 = expr(p2)
            discrm = (b*b - 4.0*d*FP2)**0.5 # Check for complex val
            if abs(b-discrm) < abs(b+discrm):
                h = -2.0*FP2 / (b+discrm)
            else:
                h = -2.0*FP2 / (b-discrm)
            curr_pt = p2 + h
            print(f"{i:03d}: {curr_pt: }")
            #if sol_array is not None:
            #    # Store each interim result and its error for later analysis
            #    iter_array.append(curr_pt)
            #    err_array.append(abs(h))

            if abs(h) < self._tol:
                # Found a solution within erro tolerance
                print(f" >> Approximate solution: {curr_pt}")
                #sol_array["muller"] = iter_array
                #sol_array["error"] = err_array
                return curr_pt

            p0, p1, p2 = p1, p2, curr_pt
            h1 = p1 - p0
            h2 = p2 - p1
            delta1 = (expr(p1)-expr(p0)) / h1
            delta2 = (expr(p2)-expr(p1)) / h2
            d = (delta2-delta1) / (h2+h1)

        print(f" >> Maximum reached. Last value evaluatd: {curr_pt}")
        #sol_array["muller"] = iter_array
        #sol_array["error"] = err_array
        return None


    def NewtonRaphson(self, 
            deriv: callable, 
            init_pt: float, 
            sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form f(x) = 0 given an initial approximation point and first-derivative
        function.


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
                 function provided, if solution exists. Otherwise,
                 None is returned.
        '''
        if not callable(deriv):
            raise TypeError(" >> 'deriv' must be callable derivative function.")
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func

        p0 = init_pt
        for i in range(1, self._max_iter):
            curr_pt = p0 - expr(p0)/deriv(p0) # Possible zero-division. 
            absDiff = abs(curr_pt-p0)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(absDiff)

            if math.isinf(curr_pt):
                print(" >> WARNING: INF detected. No solution found.")
                sol_array["newton_raphson"] = iter_array
                sol_array["error"] = err_array
                return None

            if absDiff < self._tol:
                # Found a solution within erro tolerance
                print(f" >> Approximate solution: {curr_pt} Accuracy: {absDiff}")
                sol_array["newton_raphson"] = iter_array
                sol_array["error"] = err_array
                return curr_pt

            p0 = curr_pt 

        print(f" >> Maximum reached. Last value evaluatd: {curr_pt}")
        sol_array["newton_raphson"] = iter_array
        sol_array["error"] = err_array
        return None
        


    def Secant(self, init_pt: tuple, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form f(x) = 0 given using two initial approximation point (which forms 
        a secant line to the function. This is a modified Newton-Raphson Method 
        in that derivative function is not required. However, Secant method 
        will converge slower than Newton-Raphson Method.


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
                 function provided, if solution exists. Otherwise,
                 None is returned.
        '''
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func

        p0 = init_pt[0]
        p1 = init_pt[1]
        q0 = expr(p0)
        q1 = expr(p1)
        for i in range(1, self._max_iter):
            curr_pt = p1 - q1*(p1-p0)/(q1-q0) # Possible zero-division. 
            absDiff = abs(curr_pt-p1)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(absDiff)

            if math.isinf(curr_pt):
                print(" >> WARNING: INF detected. No solution found.")
                sol_array["secant"] = iter_array
                sol_array["error"] = err_array
                return None

            if absDiff < self._tol:
                # Found a solution within erro tolerance
                print(f" >> Approximate solution: {curr_pt} Accuracy: {absDiff}")
                sol_array["secant"] = iter_array
                sol_array["error"] = err_array
                return curr_pt

            p0 = p1
            p1 = curr_pt
            q0 = q1
            q1 = expr(curr_pt)

        print(f" >> Maximum reached. Last value evaluatd: {curr_pt}")
        sol_array["secant"] = iter_array
        sol_array["error"] = err_array
        return None


    def Steffensen(self, init_pt: float, sol_array: dict = None) -> float:
        '''Perform a solution search to univariate mathematics function of the
        form g(p) = p or x - f(x) = 0 given an initial approximation point. 
        Steffensen Method is a modified Aitken Delta-Squared Method to ensure
        quadratic convergence, if conditions are met for the mathematics 
        function math_expr.
        
        Chiefly, if derivative g'(p) is not 1, and that function g has 
        continuous third derivative around a neighborhood of the solution p,
        then Steffensen method will guarantee quadratic order of convergence.


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
                 function provided, if solution exists. Otherwise,
                 None is returned.
        '''
        iter_array = array.array(self._sol_array_dtype)
        err_array = array.array(self._sol_array_dtype)
        expr = self._math_func

        p0 = init_pt
        for i in range(1, self._max_iter):
            p1 = expr(p0)
            p2 = expr(p1)
            curr_pt = p0 - (p1-p0)**2 / (p2 - 2.0*p1 + p0) # Potential 0-div
            absDiff = abs(curr_pt-p0)

            if sol_array is not None:
                # Store each interim result and its error for later analysis
                iter_array.append(curr_pt)
                err_array.append(absDiff)

            if math.isinf(curr_pt):
                print(" >> WARNING: INF detected. No solution found.")
                sol_array["steffensen"] = iter_array
                sol_array["error"] = err_array
                return None

            if absDiff < self._tol:
                # Found a solution within erro tolerance
                print(f" >> Approximate solution: {curr_pt} Accuracy: {absDiff}")
                sol_array["steffensen"] = iter_array
                sol_array["error"] = err_array
                return curr_pt

            p0 = curr_pt

        print(f" >> Maximum reached. Last value evaluatd: {curr_pt}")
        sol_array["steffensen"] = iter_array
        sol_array["error"] = err_array
        return None


    def Configure(self, 
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
    def f1(x): return x*x*x + 4*x*x - 10.0
    def f2(x): return (10.0 / (x+4.0))**0.5
    def f1d(x): return 3.0*x*x + 8.0*x
    def f3(x): return x**4 - 3*x**3 +x**2 + x + 1
    sol1 = FindRoot(f3)
    #sol1.Configure(max_iter=25)
    intermediate = dict()
    #sol1.Bisection((1,2), sol_array=intermediate) 
    #sol1.FixedPoint(1.5, sol_array=intermediate)
    #sol1.NewtonRaphson(f1d, 1.5, sol_array=intermediate)
    #sol1.Secant((1.2, 1.6), sol_array=intermediate)
    #sol1.Steffensen(1.5, sol_array=intermediate)
    #print(intermediate)
    sol1.Muller((0.5, -0.5, 0))
