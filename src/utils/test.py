

from math_utils import sample_primes
from sage.all import sin, ln, floor, cos, e, pi

def error(p, t):
    x = sum([(-1)**(n+1)/n*(1-e**(2*pi*n*t*-1/p)) for n in range(floor(ln(p)**2) + 1, p*2+ 1)])
    return (x + sum([(-1)**(n+1)/n*(1-e**(2*pi*n*t*-1/p)) for n in range(-1 * (p*2), -floor(ln(p) ** 2))])).n()


print(error(10007, 10007//10))