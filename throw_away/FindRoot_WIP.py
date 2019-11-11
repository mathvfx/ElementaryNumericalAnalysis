#!env python

def Bisection(
        mfunction, 
        interval:tuple, 
        ERR:float = 1e-4, 
        MAX:int = 1000, 
        verbose:bool = False) -> float:
    '''
    Bisection Method to find polynomial root f(x) = 0 for real number x.
    This is equivalent of Binary Search.

    
    Unlike O(log n) of Binary Search, this Bisection is O(n) operations and is 
    slow convergence to solution.

    Required condition:
      1) Function f must be continuous on its interval domain
    '''

    # Check that mfunction is a function object
    if not callable(mfunction):
        print("\t ERROR: 'mfunction' must be a math function.")
        raise TypeError

    a, b = interval
    FA = mfunction(a) # Function evaluated at the leftmost interval
    FB = mfunction(b) # Function evaluated at the rightmost interval

    # f(a) and f(b) have same sign on boundary interval imply no solution.
    if FA * FB > 0: 
        print(f"\t>> Function has no solution on [{a},{b}]")
        return None

    for i in range(1, MAX):
        midPt = (b - a)/2.0
        p = a + midPt
        FP = mfunction(p)

        if midPt < ERR or FP == 0:
            print(f"\t>> Approximated solution: {p}  Accuracy: {midPt}")
            return p 
        elif FA * FP > 0: # Possible underflow/overflow when products huge/tiny
            a = p  # move a interval rightward
            FA = FP
        else:
            b = p # move b interval leftward

        if verbose:
            print(f"{i:4d}) Current value: {p: }  Accuracy: {midPt}") # space is for sign alignment

    print(f"\t WARNING: Maximum reached. Unsuccessful search. Last-attempt value: {p}")

# Try implement Recursion version of Bisection as exercise? (Not for production use)


def FixedPoint( 
        mfunction, 
        initP:float,
        ERR:float = 1e-4,
        MAX:int = 1000,
        verbose:bool = False) -> float:
    '''
    Fixed-Point Iterative method can be thought of as a transformed root-finding from
    Bisection Method. While Fixed-Point Iteration can provide a significantly faster
    convergence, the downside is that rate of convergent is highly dependent on how
    the user transform the function from f(x) = 0 to g(x) = x - f(x) such that 
    g(p) = p.

    The rate of convergence is O(k^n) for 0 < k < 1. Length of the derivative
    g'(x) that is closer to 0 will converge faster than those closer to 1.
    '''

    if not callable(mfunction):
        print("\t ERROR: 'mfunction' must be a math function.")
        raise TypeError

    p0 = initP
    for i in range(1, MAX):
        p = mfunction( p0 )
        if abs(p - p0) < ERR:
            print(f"\t>> Approximated solution: {p}")
            return p 
        p0 = p

        if verbose:
            print(f"{i:4d}) Current value: {p: }") # space is for sign alignment

    print(f"\t WARNING: Maximum reached. Unsuccessful search. Last-attempt value: {p}")
    

# Try implement recursion version of FixedPoint with memoization as an exercise? (Not for production use)

    

def NewtonRaphson( 
        mfunction, # math function
        dfunction, # Derivative of mfunction
        initP:float,
        ERR:float = 1e-4,
        MAX:int = 1000,
        verbose:bool = False) -> float:
    '''
    Newton-Raphson Method can be thought of as an extension of Fixed-Point Method.
    However, Newton Method require that the given mathematics function be 
    twice-differentiable and that its first derivative does not vanish. Additionally,
    the initial approximation point should be "close" to the actual root of the 
    mathematics function. Otherwise, Newton Method may not converge to actual 
    solution.
    '''
    # Check mfunction is at least a function pointer
    if not callable(mfunction) or not callable(dfunction):
        print("\t ERROR: 'mfunction' must be a math function.")
        raise TypeError
        
    p0 = initP
    for i in range(1, MAX):
        p = p0 - mfunction(p0)/dfunction(p0)
        if abs(p - p0) < ERR:
            print(f"\t>> Approximated solution: {p}")
            return p

        if verbose:
            print(f"{i:4d}) Current value: p:{p: } and p0:{p0: }") # space is for sign alignment

        p0 = p

    print(f"\t WARNING: Maximum reached. Unsuccessful search. Last-attempt value: {p}")


def SecantMethod( 
        mfunction, # math function
        initP0:float,
        initP1:float,
        ERR:float = 1e-4,
        MAX:int = 1000,
        verbose:bool = False) -> float:
    '''
    Secant Method is essentially Newton-Raphson Method in that it's derivatives
    are defined as a secant instead. However, instead of requiring one initial
    approximation point, we now need two. 
    '''
    if not callable(mfunction):
        print("\t ERROR: 'mfunction' must be a math function.")
        raise TypeError

    p0 = initP0
    p1 = initP1
    for i in range(2, MAX):
        p = p1 - mfunction(p1)*(p1 - p0)/(mfunction(p1) - mfunction(p0)) #Bad move! Repeated function calls for mfunction(p0)! Consider memoize?
        if abs( p - p1 ) < ERR:
            print(f"\t>> Approximated solution: {p}")
            return p
        if verbose:
            print(f"{i:4d}) Current value: p:{p: } with p0:{p0: } and p1:{p1: }") # space is for sign alignment
        p0 = p1
        p1 = p

    print(f"\t WARNING: Maximum reached. Unsuccessful search. Last-attempt value: {p}")

def SecantMethod2( 
        mfunction, # math function
        initP0:float,
        initP1:float,
        ERR:float = 1e-4,
        MAX:int = 1000,
        verbose:bool = False) -> float:
    '''
    Secant Method is essentially Newton-Raphson Method in that it's derivatives
    are defined as a secant instead. However, instead of requiring one initial
    approximation point, we now need two. 
    '''
    if not callable(mfunction):
        print("\t ERROR: 'mfunction' must be a math function.")
        raise TypeError

    p0 = initP0
    p1 = initP1
    q0 = mfunction(p0)
    q1 = mfunction(p1)
    for i in range(2, MAX):
        p = p1 - q1*(p1 - p0)/(q1 - q0) 
        if abs( p - p1 ) < ERR:
            print(f"\t>> Approximated solution: {p}")
            return p
        if verbose:
            print(f"{i:4d}) Current value: p:{p: } with p0:{p0: } and p1:{p1: }") # space is for sign alignment
        q0 = q1
        q1 = mfunction(p)
        p0 = p1
        p1 = p

    print(f"\t WARNING: Maximum reached. Unsuccessful search. Last-attempt value: {p}")



def NewtonRaphson2( 
        mfunction, # math function
        dfunction, # Derivative of mfunction
        dfunction2, # 2nd Derivative of mfunction
        initP:float,
        ERR:float = 1e-4,
        MAX:int = 1000,
        verbose:bool = False) -> float:
    '''
    Newton-Raphson Method can be thought of as an extension of Fixed-Point Method.
    However, Newton Method require that the given mathematics function be 
    twice-differentiable and that its first derivative does not vanish. Additionally,
    the initial approximation point should be "close" to the actual root of the 
    mathematics function. Otherwise, Newton Method may not converge to actual 
    solution.
    '''
    # Check mfunction is at least a function pointer
    if not callable(mfunction) or not callable(dfunction):
        print("\t ERROR: 'mfunction' must be a math function.")
        raise TypeError
        
    p0 = initP
    for i in range(1, MAX):
        f = mfunction(p0)
        df1 = dfunction(p0)
        p = p0 - (f * df1)/(df1*df1 - f*dfunction2(p0))
        if abs(p - p0) < ERR:
            print(f"\t>> Approximated solution: {p}")
            return p

        if verbose:
            print(f"{i:4d}) Current value: p:{p: } and p0:{p0: }") # space is for sign alignment

        p0 = p

    print(f"\t WARNING: Maximum reached. Unsuccessful search. Last-attempt value: {p}")



if __name__ == "__main__":
    import scipy.optimize
    import math

    myFunc1 = lambda x: x**3 + 4*x**2 - 10 # interval(1,2)
    myFunc1d = lambda x: 3.0*x*x + 8.0*x  # interval(1,2)
    myFunc1d2 = lambda x: 6*x + 8.0  # interval(1,2)
    myFunc2 = lambda x: x*x*(x + 4) - 10
    myFunc3 = lambda x: math.exp(x) - 2.0 - math.cos(math.exp(x) - 2.0) # interval(0.5, 1.5). Consider Taylor Polynomial?
    
    verify_myFunc1 = scipy.optimize.root_scalar(myFunc1, bracket=(1, 2), xtol=5e-4, method="bisect")
    #print(f"ROOT TARGET\n {verify_myFunc1}\n\n" )

    verify_myFunc3 = scipy.optimize.root_scalar(myFunc3, bracket=(0.5, 1.5), xtol=1e-5, method="bisect")
    #print(f"ROOT TARGET\n {verify_myFunc3}\n\n" )

    #Bisection( myFunc1, (1, 2), ERR=1e-6, verbose=True )
    print()


    myFunc4 = lambda x: (10.0/(4.0 + x))**0.5 # Different form of myFunc1
    myFunc5 = lambda x: x - (x**3 + 4*x*x - 10.0)/(3*x*x + 8*x) # Even faster convergence than myFunc4
    FixedPoint(myFunc4, 1.5, ERR=1e-6, verbose=True )
    #FixedPoint(myFunc5, 1.5, ERR=1e-6, verbose=True )

    myFunc6 = lambda x: math.cos(x) - x
    myFunc6d = lambda x: -math.sin(x) - 1.0
    myFunc6d2 = lambda x: -math.cos(x)
    #print("Fixed point method: ")
    #FixedPoint(lambda x: math.cos(x), math.pi/4.0, ERR=1e-6) # Note that you don't need to do Cos(x)-x for FixedPoint
    #print("Newton method: ")
    #NewtonRaphson(myFunc6, myFunc6d, math.pi/4.0, ERR=1e-9, MAX=15, verbose=True)
    #print("Secant method: ")
    #SecantMethod2(myFunc6, 0.5, math.pi/4.0, ERR=1e-9, MAX=15, verbose=True)

    #NewtonRaphson(myFunc6, myFunc6d, math.pi/4.0, ERR=1e-9, verbose=True)
    #NewtonRaphson2(myFunc6, myFunc6d, myFunc6d2, math.pi/4.0, ERR=1e-9, verbose=True)

    

    #Demonstration that the farther p0 is from actual p, Newton Method may be
    #slow to converge, or might not even converge at all depending on function
    myFunc7 = lambda x: 4*x*x - math.exp(x) - math.exp(-x)
    myFunc7d = lambda x: 8*x - math.exp(x) + math.exp(-x)
    #print("p0=-200")
    #NewtonRaphson(myFunc7, myFunc7d, -200, ERR=1e-5, verbose=True)
    #print("p0=-10")
    #NewtonRaphson(myFunc7, myFunc7d, -10, ERR=1e-5, verbose=True)
    #print("p0=-5")
    #NewtonRaphson(myFunc7, myFunc7d, -5, ERR=1e-5, verbose=True)
    #print("p0=-3")
    #NewtonRaphson(myFunc7, myFunc7d, -3, ERR=1e-5, verbose=True)
    #print("p0=-1")
    #NewtonRaphson(myFunc7, myFunc7d, -1, ERR=1e-5, verbose=True)
    #print("p0=0")
    #NewtonRaphson(myFunc7, myFunc7d, 0, ERR=1e-5, verbose=True) #Make sure code can handle this case!
    #print("p0=1")
    #NewtonRaphson(myFunc7, myFunc7d, 1, ERR=1e-5, verbose=True)
    #print("p0=3")
    #NewtonRaphson(myFunc7, myFunc7d, 3, ERR=1e-5, verbose=True)
    #print("p0=5")
    #NewtonRaphson(myFunc7, myFunc7d, 5, ERR=1e-5, verbose=True)
    #print("p0=10")
    #NewtonRaphson(myFunc7, myFunc7d, 10, ERR=1e-5, verbose=True)
    #print("bracket with bisection method")
    #Bisection(myFunc7, interval=[0.5, 1], ERR=1e-5, MAX=100000)
    #Bisection(myFunc7, interval=[-5, -4], ERR=1e-5, MAX=100000)

    print("secant method")
    print("p0=-10, p1=-9")
    #SecantMethod(myFunc7, -10, 10, ERR=1e-5, verbose=True) #Check for division by zero case!
    #SecantMethod(myFunc7, -10, 3, ERR=1e-5, verbose=True) # Bracket actually contained 4 roots. Which ones we get is dependent on p0, p1


    myFunc8 = lambda x: x*x - 10.0*math.cos(x)
    myFunc8d = lambda x: 2*x + 10.0*math.sin(x)
    myFunc8d2 = lambda x: 2.0 + 10.0*math.cos(x)
    #print("\nnewton p0=-100")
    #NewtonRaphson(myFunc8, myFunc8d, -100, ERR=1e-9, verbose=True)
    #print("\nnewton p0=-50")
    #NewtonRaphson(myFunc8, myFunc8d, -50, ERR=1e-9, verbose=True)
    #print("\nnewton p0=-25")
    #NewtonRaphson(myFunc8, myFunc8d, -25, ERR=1e-9, verbose=True)
    #print("\nnewton p0=25")
    #NewtonRaphson(myFunc8, myFunc8d, 25, ERR=1e-9, verbose=True)
    #print("\nnewton p0=50")
    #NewtonRaphson(myFunc8, myFunc8d, 50, ERR=1e-9, verbose=True)
    #print("\nnewton p0=100")
    #NewtonRaphson(myFunc8, myFunc8d, 100, ERR=1e-9, verbose=True)


    #Bisection(myFunc8, [-5, 0], ERR=0.1, verbose=True)
    #NewtonRaphson(myFunc8, myFunc8d, -1.3, ERR=1e-9, verbose=True)
    

    #print("\n\n\n")
    myFunc9 = lambda x: 0.5 + 0.25*x*x -x*math.sin(x) - 0.5*math.cos(2*x)
    myFunc9d = lambda x: 0.8*x - math.sin(x) - x*math.cos(x) + math.sin(2*x)
    #NewtonRaphson(myFunc9, myFunc9d, math.pi/2.0, ERR=1e-5)
    #NewtonRaphson(myFunc9, myFunc9d, math.pi*5.0, ERR=1e-5)
    #NewtonRaphson(myFunc9, myFunc9d, math.pi*10.0, ERR=1e-5)

    myFunc10 = lambda x: math.exp(x) - x - 1.0
    myFunc10d = lambda x:math.exp(x) - 1.0
    myFunc10d2 = lambda x:math.exp(x) 
    NewtonRaphson( myFunc10, myFunc10d, 1, verbose=True)
    NewtonRaphson2( myFunc10, myFunc10d, myFunc10d2, 1, verbose=True)
