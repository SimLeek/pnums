import numpy as np

from pnums import PInt


def test_and_3d():
    a = PInt(0, 0, 0, bits=2)
    b = PInt(0, 0, 0, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(0, 0, 1, bits=2)
    b = PInt(0, 0, 0, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(0, 0, 1, bits=2)
    b = PInt(0, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(0, 1, 0, bits=2)
    b = PInt(0, 1, 0, bits=2)
    c = a & b
    assert c.asfloat() == (0, 1, 0)

    a = PInt(1, 0, 0, bits=2)
    b = PInt(1, 0, 0, bits=2)
    c = a & b
    assert c.asfloat() == (1, 0, 0)

    a = PInt(1, 1, 0, bits=2)
    b = PInt(1, 1, 0, bits=2)
    c = a & b
    assert c.asfloat() == (1, 1, 0)

    a = PInt(1, 0, 1, bits=2)
    b = PInt(1, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (1, 0, 1)

    a = PInt(0, 1, 1, bits=2)
    b = PInt(0, 1, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 1, 1)

    a = PInt(1, 1, 1, bits=2)
    b = PInt(1, 1, 1, bits=2)
    c = a & b
    assert c.asfloat() == (1, 1, 1)

    # 001
    # 010
    a = PInt(0, 0, 1, bits=2)
    b = PInt(0, 1, 0, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(0, 1, 0, bits=2)
    b = PInt(0, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    # 001
    # 100
    a = PInt(0, 0, 1, bits=2)
    b = PInt(0, 1, 0, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(0, 1, 0, bits=2)
    b = PInt(0, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    # 001
    # 110
    a = PInt(0, 0, 1, bits=2)
    b = PInt(1, 1, 0, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(1, 1, 0, bits=2)
    b = PInt(0, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    # 001
    # 101
    a = PInt(0, 0, 1, bits=2)
    b = PInt(1, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(1, 0, 1, bits=2)
    b = PInt(0, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    # 001
    # 011
    a = PInt(0, 0, 1, bits=2)
    b = PInt(0, 1, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(0, 1, 1, bits=2)
    b = PInt(0, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    # 001
    # 111
    a = PInt(0, 0, 1, bits=2)
    b = PInt(1, 1, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(1, 1, 1, bits=2)
    b = PInt(0, 0, 1, bits=2)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    # final
    a = PInt(10, 11, 12, bits=8)
    b = PInt(6, 13, 7, bits=8)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    np.testing.assert_array_almost_equal(
        [[[[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]],

          [[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 2., 0.]]],

         [[[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 2., 0., 0., 2.]],

          [[0., 0., 0., 0., 0., 2., 0., 0.],
           [2., 2., 2., 2., 0., 0., 0., 0.]]]],
        c.tensor,
    )

    a = PInt(10, 11, 12, bits=8, confidence=1)
    b = PInt(6, 13, 7, bits=8, confidence=.9)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    a = PInt(10, 11, 12, bits=8, confidence=.9)
    b = PInt(6, 13, 7, bits=8, confidence=.8)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    a = PInt(10, 11, 12, bits=8, confidence=.6)
    b = PInt(6, 13, 7, bits=8, confidence=.7)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    a = PInt(10, 11, 12, bits=8, confidence=.8)
    b = PInt(6, 13, 7, bits=8, confidence=.7)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    np.testing.assert_array_almost_equal(
        [[[[0.00183674, 0.00183674, 0.00183674, 0.00183674, 0.05142858,
            0.03, 0.00183673, 0.00183674],
           [0.00551021, 0.00551021, 0.00551021, 0.00551021, 0.05510204,
            0.03367347, 0.10469388, 0.00551021]],

          [[0.00551021, 0.00551021, 0.00551021, 0.00551021, 0.05510204,
            0.03367347, 0.06183673, 0.00551021],
           [0.01653061, 0.01653061, 0.01653061, 0.01653061, 0.06612246,
            0.04469386, 0.932449, 0.01653061]]],

         [[[0.00551021, 0.00551021, 0.00551021, 0.00551021, 0.05510204,
            0.03367347, 0.0055102, 0.06183673],
           [0.01653061, 0.01653061, 0.01653061, 0.01653061, 0.9391836,
            0.04469386, 0.11571428, 1.0316327]],

          [[0.01653061, 0.01653061, 0.01653061, 0.01653061, 0.06612246,
            1.0034693, 0.07285714, 0.07285716],
           [1.4320408, 1.4320408, 1.4320408, 1.4320408, 0.21183676,
            0.27612248, 0.20510204, 0.3042857]]]],
        c.tensor,
    )

    np.testing.assert_array_almost_equal(
        [[[[0.00122449, 0.00122449, 0.00122449, 0.00122449, 0.03428572,
            0.02, 0.00122449, 0.00122449],
           [0.00367347, 0.00367347, 0.00367347, 0.00367347, 0.03673469,
            0.02244898, 0.06979592, 0.00367347]],

          [[0.00367347, 0.00367347, 0.00367347, 0.00367347, 0.03673469,
            0.02244898, 0.04122449, 0.00367347],
           [0.01102041, 0.01102041, 0.01102041, 0.01102041, 0.04408164,
            0.02979591, 0.62163264, 0.01102041]]],

         [[[0.00367347, 0.00367347, 0.00367347, 0.00367347, 0.03673469,
            0.02244898, 0.00367347, 0.04122449],
           [0.01102041, 0.01102041, 0.01102041, 0.01102041, 0.6261224,
            0.02979591, 0.07714286, 0.6877551]],

          [[0.01102041, 0.01102041, 0.01102041, 0.01102041, 0.04408164,
            0.6689796, 0.04857143, 0.04857144],
           [0.95469385, 0.95469385, 0.95469385, 0.95469385, 0.1412245,
            0.18408166, 0.1367347, 0.20285714]]]],
        c.normalize(1.0).tensor,
    )


def test_and_3d_unsure():
    '''a = PInt(0, 0, 0, bits=2, confidence=.55)
    b = PInt(0, 0, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)'''

    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(0, 0, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(0, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(0, 1, 0, bits=2, confidence=.55)
    b = PInt(0, 1, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 1, 0)

    a = PInt(1, 0, 0, bits=2, confidence=.55)
    b = PInt(1, 0, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (1, 0, 0)

    a = PInt(1, 1, 0, bits=2, confidence=.55)
    b = PInt(1, 1, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (1, 1, 0)

    a = PInt(1, 0, 1, bits=2, confidence=.55)
    b = PInt(1, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (1, 0, 1)

    a = PInt(0, 1, 1, bits=2, confidence=.55)
    b = PInt(0, 1, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 1, 1)

    a = PInt(1, 1, 1, bits=2, confidence=.55)
    b = PInt(1, 1, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (1, 1, 1)

    # 001
    # 010
    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(0, 1, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(0, 1, 0, bits=2, confidence=.55)
    b = PInt(0, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    # 001
    # 100
    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(0, 1, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(0, 1, 0, bits=2, confidence=.55)
    b = PInt(0, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    # 001
    # 110
    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(1, 1, 0, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    a = PInt(1, 1, 0, bits=2, confidence=.55)
    b = PInt(0, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 0)

    # 001
    # 101
    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(1, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(1, 0, 1, bits=2, confidence=.55)
    b = PInt(0, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    # 001
    # 011
    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(0, 1, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(0, 1, 1, bits=2, confidence=.55)
    b = PInt(0, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    # 001
    # 111
    a = PInt(0, 0, 1, bits=2, confidence=.55)
    b = PInt(1, 1, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    a = PInt(1, 1, 1, bits=2, confidence=.55)
    b = PInt(0, 0, 1, bits=2, confidence=.55)
    c = a & b
    assert c.asfloat() == (0, 0, 1)

    # final
    a = PInt(10, 11, 12, bits=8, confidence=.55)
    b = PInt(6, 13, 7, bits=8, confidence=.55)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    np.testing.assert_array_almost_equal(
        [[[[0.00454592, 0.00454592, 0.00454592, 0.00454592, 0.03889285,
            0.03889286, 0.00454592, 0.00454592],
           [0.01363776, 0.01363776, 0.01363776, 0.01363776, 0.04798468,
            0.04798469, 0.08233164, 0.01363776]],

          [[0.01363776, 0.01363776, 0.01363776, 0.01363776, 0.04798468,
            0.04798469, 0.08233164, 0.01363776],
           [0.04091327, 0.04091327, 0.04091327, 0.04091327, 0.07526021,
            0.07526021, 0.43781123, 0.04091327]]],

         [[[0.01363776, 0.01363776, 0.01363776, 0.01363776, 0.04798468,
            0.04798469, 0.01363776, 0.08233164],
           [0.04091327, 0.04091327, 0.04091327, 0.04091327, 0.47215813,
            0.07526021, 0.10960714, 0.50650513]],

          [[0.04091327, 0.04091327, 0.04091327, 0.04091327, 0.07526021,
            0.47215816, 0.10960714, 0.10960714],
           [0.9318011, 0.9318011, 0.9318011, 0.9318011, 0.29447454,
            0.29447457, 0.2601276, 0.32882148]]]],
        c.tensor,
    )

    a = PInt(10, 11, 12, bits=8, confidence=1)
    b = PInt(6, 13, 7, bits=8, confidence=.9)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    a = PInt(10, 11, 12, bits=8, confidence=.9)
    b = PInt(6, 13, 7, bits=8, confidence=.8)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    a = PInt(10, 11, 12, bits=8, confidence=.6)
    b = PInt(6, 13, 7, bits=8, confidence=.7)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    a = PInt(10, 11, 12, bits=8, confidence=.8)
    b = PInt(6, 13, 7, bits=8, confidence=.7)
    c = a & b
    assert c.asfloat() == (2, 9, 4)

    np.testing.assert_array_almost_equal(
        [[[[0.00183674, 0.00183674, 0.00183674, 0.00183674, 0.05142858,
            0.03, 0.00183673, 0.00183674],
           [0.00551021, 0.00551021, 0.00551021, 0.00551021, 0.05510204,
            0.03367347, 0.10469388, 0.00551021]],

          [[0.00551021, 0.00551021, 0.00551021, 0.00551021, 0.05510204,
            0.03367347, 0.06183673, 0.00551021],
           [0.01653061, 0.01653061, 0.01653061, 0.01653061, 0.06612246,
            0.04469386, 0.932449, 0.01653061]]],

         [[[0.00551021, 0.00551021, 0.00551021, 0.00551021, 0.05510204,
            0.03367347, 0.0055102, 0.06183673],
           [0.01653061, 0.01653061, 0.01653061, 0.01653061, 0.9391836,
            0.04469386, 0.11571428, 1.0316327]],

          [[0.01653061, 0.01653061, 0.01653061, 0.01653061, 0.06612246,
            1.0034693, 0.07285714, 0.07285716],
           [1.4320408, 1.4320408, 1.4320408, 1.4320408, 0.21183676,
            0.27612248, 0.20510204, 0.3042857]]]],
        c.tensor,
    )

    np.testing.assert_array_almost_equal(
        [[[[0.00122449, 0.00122449, 0.00122449, 0.00122449, 0.03428572,
            0.02, 0.00122449, 0.00122449],
           [0.00367347, 0.00367347, 0.00367347, 0.00367347, 0.03673469,
            0.02244898, 0.06979592, 0.00367347]],

          [[0.00367347, 0.00367347, 0.00367347, 0.00367347, 0.03673469,
            0.02244898, 0.04122449, 0.00367347],
           [0.01102041, 0.01102041, 0.01102041, 0.01102041, 0.04408164,
            0.02979591, 0.62163264, 0.01102041]]],

         [[[0.00367347, 0.00367347, 0.00367347, 0.00367347, 0.03673469,
            0.02244898, 0.00367347, 0.04122449],
           [0.01102041, 0.01102041, 0.01102041, 0.01102041, 0.6261224,
            0.02979591, 0.07714286, 0.6877551]],

          [[0.01102041, 0.01102041, 0.01102041, 0.01102041, 0.04408164,
            0.6689796, 0.04857143, 0.04857144],
           [0.95469385, 0.95469385, 0.95469385, 0.95469385, 0.1412245,
            0.18408166, 0.1367347, 0.20285714]]]],
        c.normalize(1.0).tensor,
    )
