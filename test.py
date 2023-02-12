import lagrange_interpolation
import numpy

c1 = [
    [1, 2],
    [3, 2],
    [4, -1]
]

c2 = [
    [0, 7],
    [1, 5],
    [2, 3],
    [3, 8],
    [4, 9],
    [5, 2]
]

c3 = [
    [0, 2],
    [1, 4],
    [2, 3],
    [3, 1]
]

p1 = lagrange_interpolation.interpolate(c1)
numpy.testing.assert_array_equal(
    p1,
    [-1, 4, -1]
)

p2 = lagrange_interpolation.interpolate(c2)
numpy.testing.assert_allclose(
    p2,
    [7, 9.8333, -22.1666, 12.9583, -2.8333, 0.2083],
    rtol=1e-3
)

p3 = lagrange_interpolation.interpolate(c3)
numpy.testing.assert_allclose(
    p3,
    [2, 4.1666, -2.5, 0.3333],
    rtol=1e-3
)