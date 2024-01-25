__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import pytest
import numpy

import scine_utilities as scine


def test_atom():
    position = numpy.array([0.0, 1.0, 2.0])
    a = scine.Atom(scine.ElementType.H, position)
    assert numpy.array_equal(a.position, position)
    assert a.element == scine.ElementType.H

    a.element = scine.ElementType.F
    assert a.element == scine.ElementType.F

    new_position = numpy.array([1.0, 2.0, 3.0])
    assert not numpy.array_equal(a.position, new_position)
    a.position = new_position
    assert numpy.array_equal(a.position, new_position)
