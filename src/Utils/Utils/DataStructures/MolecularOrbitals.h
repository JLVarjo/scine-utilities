/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */

#ifndef UTILS_MOLECULARORBITALS_H
#define UTILS_MOLECULARORBITALS_H

#include <Utils/DataStructures/SpinAdaptedMatrix.h>
#include <cassert>

namespace Scine {
namespace Utils {

/*!
 * Class for the coefficient (molecular orbital) matrix.
 * Contains all the eigenfunctions (molecular orbitals) produced when solving the (generalized) eigenvalue problem.
 * The contained matrices are therefore quadratic; their dimension is the number of basis functions.
 * TODO: Take SingleParticleEnergies inside this class?
 * \sa OccupiedMolecularOrbitals
 */
class MolecularOrbitals {
 public:
  using Matrix = SpinAdaptedMatrix::Matrix;

  static MolecularOrbitals createEmptyUnrestrictedOrbitals();
  static MolecularOrbitals createEmptyRestrictedOrbitals();
  template<typename T>
  static MolecularOrbitals createFromRestrictedCoefficients(T&& restricted);
  template<typename T>
  static MolecularOrbitals createFromUnrestrictedCoefficients(T&& alpha, T&& beta);

  /*! If the molecular orbitals are restricted, transforms them into unrestricted ones. */
  void makeUnrestricted();
  /*! Return a copy of the orbitals, transformed to the unrestricted variant if needed. */
  MolecularOrbitals toUnrestricted() const;

  /*! \return true if the MO's are valid and can be used, for instance, for the density matrix generation. */
  bool isValid() const;
  /*! Reinitialize the MO's, for instance if the molecular structure changes. */
  void invalidate();
  bool isRestricted() const;
  bool isUnrestricted() const;
  int numberOrbitals() const;

  Matrix& restrictedMatrix();
  const Matrix& restrictedMatrix() const;
  Matrix& alphaMatrix();
  const Matrix& alphaMatrix() const;
  Matrix& betaMatrix();
  const Matrix& betaMatrix() const;

 private:
  SpinAdaptedMatrix matrix_;
  bool valid_ = false;
  bool unrestricted_ = false;
};

template<typename T>
MolecularOrbitals MolecularOrbitals::createFromRestrictedCoefficients(T&& restricted) {
  MolecularOrbitals mo;
  mo.matrix_.setRestrictedMatrix(std::forward<T>(restricted));
  mo.valid_ = true;
  mo.unrestricted_ = false;
  return mo;
}

template<typename T>
MolecularOrbitals MolecularOrbitals::createFromUnrestrictedCoefficients(T&& alpha, T&& beta) {
  assert(alpha.cols() == beta.cols() && alpha.rows() == beta.rows() &&
         "Alpha and beta matrix for MolecularOrbitals construction do not have the same size");
  MolecularOrbitals mo;
  mo.matrix_.setAlphaMatrix(std::forward<T>(alpha));
  mo.matrix_.setBetaMatrix(std::forward<T>(beta));
  mo.valid_ = true;
  mo.unrestricted_ = true;
  return mo;
}

inline bool MolecularOrbitals::isValid() const {
  return valid_;
}

inline bool MolecularOrbitals::isUnrestricted() const {
  return unrestricted_;
}

inline bool MolecularOrbitals::isRestricted() const {
  return !isUnrestricted();
}

inline MolecularOrbitals::Matrix& MolecularOrbitals::restrictedMatrix() {
  return matrix_.restrictedMatrix();
}

inline MolecularOrbitals::Matrix& MolecularOrbitals::alphaMatrix() {
  return matrix_.alphaMatrix();
}

inline MolecularOrbitals::Matrix& MolecularOrbitals::betaMatrix() {
  return matrix_.betaMatrix();
}
inline const MolecularOrbitals::Matrix& MolecularOrbitals::restrictedMatrix() const {
  return matrix_.restrictedMatrix();
}

inline const MolecularOrbitals::Matrix& MolecularOrbitals::alphaMatrix() const {
  return matrix_.alphaMatrix();
}

inline const MolecularOrbitals::Matrix& MolecularOrbitals::betaMatrix() const {
  return matrix_.betaMatrix();
}

inline int MolecularOrbitals::numberOrbitals() const {
  if (isRestricted())
    return static_cast<int>(restrictedMatrix().cols());
  return static_cast<int>(alphaMatrix().cols());
}

} // namespace Utils
} // namespace Scine
#endif // UTILS_MOLECULARORBITALS_H