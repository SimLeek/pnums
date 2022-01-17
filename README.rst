PNums
=======
Probabilistic numbers.

This library simulates the coordinate system in the mammalian entorhinal cortex, which is used for recognizing and
mapping features to locations, or SLAM (Simultaneous Localization and Mapping).

The pnums here support coordinates in n-dimensions as well as mathematical operations on those numbers, which should
allow for sensor fusion. Some research suggests this system is repeated in the neocortex. [Hawkins2019]_

Example
-------

.. code-block:: python

    a = PInt(10, 11, bits=8, confidence=0.8)
    b = PInt(6, 13, bits=8, confidence=0.9)
    c = a + b

    assert c.asfloat() == (16, 24)

    np.testing.assert_array_almost_equal(
        [
            [
                [
                    0.15804069,
                    0.16226967,
                    0.20636845,
                    0.6811368,
                    0.31998023,
                    0.21316226,
                    0.16209187,
                    0.1548889,
                ],
                [
                    0.16720827,
                    0.17999499,
                    0.23011242,
                    0.2770214,
                    0.30978364,
                    0.30276603,
                    0.15728992,
                    0.1548889,
                ],
            ],
            [
                [
                    0.16867277,
                    0.18810391,
                    0.27501187,
                    0.5256308,
                    0.6511188,
                    0.44122377,
                    0.35417086,
                    0.1548889,
                ],
                [
                    1.2060783,
                    1.1696315,
                    0.9885073,
                    0.21621099,
                    0.41911733,
                    0.742848,
                    1.0264474,
                    1.2353333,
                ],
            ],
        ],
        c.tensor,
    )

    np.testing.assert_array_almost_equal(
        [
            [
                [
                    0.04648256,
                    0.04772637,
                    0.0606966,
                    0.20033436,
                    0.09411184,
                    0.06269478,
                    0.04767408,
                    0.04555556,
                ],
                [
                    0.0491789,
                    0.0529397,
                    0.06768012,
                    0.08147689,
                    0.09111284,
                    0.08904883,
                    0.04626174,
                    0.04555556,
                ],
            ],
            [
                [
                    0.04960964,
                    0.05532468,
                    0.08088584,
                    0.1545973,
                    0.19150554,
                    0.1297717,
                    0.10416789,
                    0.04555556,
                ],
                [
                    0.3547289,
                    0.34400925,
                    0.29073742,
                    0.06359147,
                    0.12326981,
                    0.2184847,
                    0.3018963,
                    0.3633333,
                ],
            ],
        ],
        c.normalize(0.5).tensor,
    )

    q = c.quantize()
    np.testing.assert_array_almost_equal(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
        ],
        q.tensor,
    )

Installation
------------

:code:`pip install pnums`

Citations
---------
.. [Hawkins2019]
    Hawkins, J., Lewis, M., Klukas, M., Purdy, S., &amp; Ahmad, S. (2019). A framework for intelligence and cortical function based on grid cells in the neocortex. Frontiers in Neural Circuits, 12. https://doi.org/10.3389/fncir.2018.00121

To Do
-----
* Needs to be tested for various use cases vs using a floating point number and an extra linear layer.

  * Known probabilities in training data
  * Combining outputs from multiple neural networks
  * Use as coordinates for transformer neural networks
  * Automatically gaining probability information from training data with no probability information

* More mathematical operations need to be defined

  * Currently only addition, subtraction, and a few logical operations are defined, but the rest of the operations can be defined out of the current logical operations.