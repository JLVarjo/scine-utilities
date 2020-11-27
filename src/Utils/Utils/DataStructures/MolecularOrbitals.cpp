/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */

#include "MolecularOrbitals.h"

namespace Scine {
namespace Utils {

void MolecularOrbitals::makeUnrestricted() {
  if (unrestricted_)
    return;
  // Set alpha by copy, set beta by move.
  matrix_.setAlphaMatrix(matrix_.restrictedMatrix());
  matrix_.setBetaMatrix(std::move(matrix_.restrictedMatrix()));
  unrestricted_ = true;
}

MolecularOrbitals MolecularOrbitals::toUnrestricted() const {
  MolecularOrbitals mo = *this;
  mo.makeUnrestricted();
  return mo;
}

void MolecularOrbitals::invalidate() {
  *this = MolecularOrbitals{};
}

MolecularOrbitals MolecularOrbitals::createEmptyUnrestrictedOrbitals() {
  return createFromUnrestrictedCoefficients(Eigen::MatrixXd{}, Eigen::MatrixXd{});
}

MolecularOrbitals MolecularOrbitals::createEmptyRestrictedOrbitals() {
  return createFromRestrictedCoefficients(Eigen::MatrixXd{});
}

} // namespace Utils
} // namespace Scine
