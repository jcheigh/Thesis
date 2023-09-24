
import numpy as np 
from sage.all import *
import random
from helper import legendre, sample_primes


def check(p, d, sin_approx=False):
    if not sin_approx:
        cons = 2 * np.sqrt(p) / pi 
        leg = [legendre(n, p).n() for n in range(1, floor(ln(p)**2)+1)]
        neg = [(-1) ** n for n in range(1, floor(ln(p)**2)+1)]
        sin_term1 = [sin(pi*n/p).n() for n in range(1, floor(ln(p)**2)+1)]
        sin_term2 = [sin(2*pi*n*d/p).n() for n in range(1, floor(ln(p)**2)+1)]
        prod = [a*b*c*d for a,b,c,d in zip(leg, neg, sin_term1, sin_term2)]
        total = list(zip(leg, neg, sin_term1, sin_term2))
        return total
    else:
        cons = 2 / np.sqrt(p) 
        leg = [legendre(n, p) for n in range(1, floor(ln(p)**2)+1)]
        neg = [(-1) ** n for n in range(1, floor(ln(p)**2)+1)]
        sin_term = [round(sin(2*pi*n*d/p).n(),4) for n in range(1, floor(ln(p)**2)+1)]
        prod = [round(a*b*c, 4) for a,b,c in zip(leg, neg, sin_term)]
        total = list(zip(leg, neg, sin_term, prod))     
        return total

def SymCheck(p, d, sin_approx=False):
    # should be ~ 0 if p equiv 3 mod 4
    if not sin_approx:
        cons = 2 * np.sqrt(p) / pi
        main = sum([(legendre(n, p) * (-1)**n * sin(pi*n/p) * sin(2*pi*n*d/p)) / n for n in range(1, floor(ln(p)**2)+1)])
    else:
        cons = 2 / np.sqrt(p)
        main = sum([(legendre(n, p) * (-1)**n * sin(2*pi*n*d/p)) for n in range(1, floor(ln(p)**2)+1)])
        
    total = (cons * main).n()
    print(total)
    return total


def AntiSymCheck(p, d):
    # should be ~ 0 if p equiv 1 mod 4 
    mid = (p - 1) // 2
    d = int(d)
    leg = [legendre(a, p) for a in range(mid - d + 1, mid + d + 1)]
    leg_sum = sum(leg)
    cons = (2 * np.sqrt(p) / pi)
    big_sum = sum(legendre(n, p) * cos(pi*n) * sin(pi*n/p)*cos((2*pi*n*d)/p) / n for n in range(1, floor(ln(p) ** 2) + 1))

    total = leg_sum - cons * big_sum

    return total.n()

if __name__ == "__main__":
    primes = sample_primes(50, 100000, 200000)
    for prime in primes:
        if prime % 4 == 3:
            d = 2 * prime / 5
            #lst = check(prime, 5, sin_approx = True)
            print(AntiSymCheck(prime, d))
            #for leg, neg, sin, prod in lst:
            #    print(f"Legendre: {leg}, Neg: {neg}, Sin: {sin}, Product: {prod}")
            #    print(f"Leg * Neg = {leg * neg}")
            #    print(f"Leg * Sin = {leg * sin}")       
            #    print(f"Neg * Sin = {neg * sin}")      
            #    print('=' * 15)
"""
Goal:

Empirically when p == 3 mod 4 then

Let #p = (p-1) / 2
sum(legendre(a, p) for a in range(#p - d + 1, #p + d + 1))
 
- (2 * np.sqrt(p)) / pi) * sum(
    legendre(n, p) * cos(pi * n) * sin((pi * n)/p) * sin((2pi * n * d) / p) / n)
    for n in range(1, floor(ln(p)**2)) + 1
    ) ~ 0 

So, 

(a) for what p does this phenomena occur
(b) is this formula actually right?
(c) what part of the expression is actually causing this 






"""