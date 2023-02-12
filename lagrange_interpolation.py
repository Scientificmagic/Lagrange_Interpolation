import numpy


def interpolate(coordinates: list[tuple]) -> numpy.polynomial:
    # unpack coordinate pairs
    xs, ys = list(zip(*coordinates))

    # dupe list and remove diagonal
    # https://stackoverflow.com/a/46736275
    #             x1 x2 x3       x2 x3
    # x1 x2 x3 -> x1 x2 x3 -> x1    x3
    #             x1 x2 x3    x1 x2   
    xs2D = numpy.tile(xs, (len(xs), 1))
    m = xs2D.shape[0]
    strided = numpy.lib.stride_tricks.as_strided
    s0, s1 = xs2D.strides
    out = strided(xs2D.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)

    # c0 + c1*x + ... + x^n
    polynomial = numpy.apply_along_axis(numpy.polynomial.polynomial.polyfromroots, 1, out)
    # print(f'\tStarting polynomial:\n{polynomial}')

    # L(x) = 1 @ x, so polynomial needs to be scaled down (divide)
    # Lagrange polynomial later gets scaled up to proper Y value (multiply)
    # The scaling was lumped together with the coefficient for organization
    scalings = numpy.empty((len(xs), 0), int)
    for i, x in enumerate(xs):
        eval = numpy.polynomial.polynomial.polyval(x, polynomial[i])
        scalings = numpy.append(scalings, eval)
    scalings = numpy.divide(1, scalings)
    scalings = numpy.multiply(scalings, ys)
    # print(f'\tScalings div 1 mul y value:\n{scalings}')

    # Derive final lagrange for each (x,y)
    lagrange_polynomials = numpy.empty_like(polynomial)
    for i, scaling in enumerate(scalings):
        lagrange_polynomials[i] = numpy.multiply(scaling, polynomial[i])
    # print(f'\tLagrange Polynomial:\n{lagrange_polynomials}')

    # P(x) = y1*L1(x) + y2*L2(x) + ...
    # Summation of each individual lagrange polynomial
    final_poly = numpy.zeros_like(scalings)
    for exp in lagrange_polynomials:
        final_poly = numpy.polynomial.polynomial.polyadd(final_poly, exp)
    # print(f'\tFinal Polynomial:\n{final_poly}')
    return final_poly


if __name__ == "__main__":
    interpolate()
