/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */

// Alain 18.12.2014: workaround after VS2013 & intel XE 2015
#ifndef MKL_BLAS
#  define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#include <Utils/Bonds/BondOrderCollection.h>
#include <Utils/DataStructures/AtomsOrbitalsIndexes.h>
#include <Utils/DataStructures/DensityMatrix.h>
#include <Utils/DataStructures/MolecularOrbitals.h>
#include <Utils/DataStructures/SingleParticleEnergies.h>
#include <Utils/DataStructures/SpinAdaptedMatrix.h>
#include <Utils/Math/IterativeDiagonalizer/DavidsonDiagonalizer.h>
#include <Utils/Scf/LcaoUtils/LcaoUtils.h>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace Scine {
namespace Utils {

namespace LcaoUtils {

void getNumberUnrestrictedElectrons(int& nAlpha, int& nBeta, int nElectrons, int spinMultiplicity) {
  assert(nElectrons >= 0);
  assert(spinMultiplicity >= 1);
  assert(spinMultiplicity <= nElectrons + 1 &&
         "Spin multiplicity is invalid with the number of electrons: too few electrons.");
  assert((spinMultiplicity + nElectrons) % 2 == 1 &&
         "Spin multiplicity is invalid with the number of electrons: they can't have the same parity.");
  // NB: system of two equations with two unknowns:
  // nAlpha + nBeta = nElectrons_
  // nAlpha - nBeta = spinMultiplicity - 1
  nAlpha = (nElectrons + spinMultiplicity - 1) / 2;
  nBeta = nElectrons - nAlpha;
}

void solveOccupiedRestrictedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix, MolecularOrbitals& coefficientMatrix,
                                              SingleParticleEnergies& singleParticleEnergies, int nElectrons, Core::Log& log) {
  if (fockMatrix.restrictedMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyRestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyRestrictedEnergies();
    return;
  }

  int nOccupied = nElectrons / 2;
  DavidsonDiagonalizer<Eigen::MatrixXd, DavidsonBalancedType::balanced> diagonalizer(
      nOccupied, nOccupied, fockMatrix.restrictedMatrix().cols(), log);
  diagonalizer.setMatrixToDiagonalize(fockMatrix.restrictedMatrix(), log);
  auto eigenPairs = diagonalizer.solve(log);
  eigenPairs.second.conservativeResize(Eigen::NoChange, fockMatrix.restrictedMatrix().cols());
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(std::move(eigenPairs.second));
  singleParticleEnergies.setRestricted(eigenPairs.first);
}

void solveOccupiedRestrictedGeneralizedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix,
                                                         const Eigen::MatrixXd& overlapMatrix,
                                                         MolecularOrbitals& coefficientMatrix,
                                                         SingleParticleEnergies& singleParticleEnergies, int nElectrons,
                                                         Core::Log& log) {
  if (fockMatrix.restrictedMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyRestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyRestrictedEnergies();
    return;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(overlapMatrix);
  const Eigen::MatrixXd invSqrtS = es.operatorInverseSqrt();
  const Eigen::MatrixXd modifiedFock =
      invSqrtS.selfadjointView<Eigen::Upper>() * fockMatrix.restrictedMatrix() * invSqrtS.selfadjointView<Eigen::Upper>();
  int nOccupied = nElectrons / 2;
  DavidsonDiagonalizer<Eigen::MatrixXd, DavidsonBalancedType::balanced> diagonalizer(
      nOccupied, nOccupied, fockMatrix.restrictedMatrix().cols(), log);
  diagonalizer.setMatrixToDiagonalize(modifiedFock, log);
  auto eigenPairs = diagonalizer.solve(log);
  Eigen::MatrixXd deOrthogonalizedCoefficients = invSqrtS.selfadjointView<Eigen::Upper>() * eigenPairs.second;
  deOrthogonalizedCoefficients.conservativeResize(Eigen::NoChange, fockMatrix.restrictedMatrix().cols());
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(deOrthogonalizedCoefficients);
  singleParticleEnergies.setRestricted(eigenPairs.first);
}

void solveOccupiedUnrestrictedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix, MolecularOrbitals& coefficientMatrix,
                                                SingleParticleEnergies& singleParticleEnergies, int nAlphaElectrons,
                                                int nBetaElectrons, Core::Log& log) {
  if (fockMatrix.alphaMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyUnrestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyUnrestrictedEnergies();
    return;
  }

  DavidsonDiagonalizer<Eigen::MatrixXd, DavidsonBalancedType::balanced> diagonalizerAlpha(
      nAlphaElectrons, nAlphaElectrons, fockMatrix.alphaMatrix().cols(), log);
  diagonalizerAlpha.setMatrixToDiagonalize(fockMatrix.alphaMatrix(), log);
  auto alphaEigenPairs = diagonalizerAlpha.solve(log);
  alphaEigenPairs.second.conservativeResize(Eigen::NoChange, fockMatrix.alphaMatrix().cols());
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(std::move(alphaEigenPairs.second));
  singleParticleEnergies.setRestricted(alphaEigenPairs.first);

  DavidsonDiagonalizer<Eigen::MatrixXd, DavidsonBalancedType::balanced> diagonalizerBeta(
      nBetaElectrons, nBetaElectrons, fockMatrix.betaMatrix().cols(), log);
  diagonalizerBeta.setMatrixToDiagonalize(fockMatrix.betaMatrix(), log);
  auto betaEigenPairs = diagonalizerBeta.solve(log);
  betaEigenPairs.second.conservativeResize(Eigen::NoChange, fockMatrix.betaMatrix().cols());
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(std::move(betaEigenPairs.second));
  singleParticleEnergies.setRestricted(betaEigenPairs.first);
}

void solveOccupiedUnrestrictedGeneralizedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix,
                                                           const Eigen::MatrixXd& overlapMatrix,
                                                           MolecularOrbitals& coefficientMatrix,
                                                           SingleParticleEnergies& singleParticleEnergies,
                                                           int nAlphaElectrons, int nBetaElectrons, Core::Log& log) {
  if (fockMatrix.alphaMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyUnrestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyUnrestrictedEnergies();
    return;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(overlapMatrix);
  const Eigen::MatrixXd invSqrtS = es.operatorInverseSqrt();
  const Eigen::MatrixXd modifiedAlphaFock =
      invSqrtS.selfadjointView<Eigen::Upper>() * fockMatrix.alphaMatrix() * invSqrtS.selfadjointView<Eigen::Upper>();
  const Eigen::MatrixXd modifiedBetaFock =
      invSqrtS.selfadjointView<Eigen::Upper>() * fockMatrix.betaMatrix() * invSqrtS.selfadjointView<Eigen::Upper>();

  DavidsonDiagonalizer<Eigen::MatrixXd, DavidsonBalancedType::balanced> diagonalizerAlpha(
      nAlphaElectrons, nAlphaElectrons, fockMatrix.alphaMatrix().cols(), log);
  auto alphaEigenPairs = diagonalizerAlpha.solve(log);
  Eigen::MatrixXd alphaDeorthogonalizedCoefficients = invSqrtS.selfadjointView<Eigen::Upper>() * alphaEigenPairs.second;
  alphaDeorthogonalizedCoefficients.conservativeResize(Eigen::NoChange, fockMatrix.alphaMatrix().cols());
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(std::move(alphaDeorthogonalizedCoefficients));
  singleParticleEnergies.setRestricted(alphaEigenPairs.first);

  DavidsonDiagonalizer<Eigen::MatrixXd, DavidsonBalancedType::balanced> diagonalizerBeta(
      nBetaElectrons, nBetaElectrons, fockMatrix.betaMatrix().cols(), log);
  auto betaEigenPairs = diagonalizerBeta.solve(log);
  Eigen::MatrixXd betaDeorthogonalizedCoefficients = invSqrtS.selfadjointView<Eigen::Upper>() * betaEigenPairs.second;
  betaDeorthogonalizedCoefficients.conservativeResize(Eigen::NoChange, fockMatrix.betaMatrix().cols());
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(std::move(betaDeorthogonalizedCoefficients));
  singleParticleEnergies.setRestricted(betaEigenPairs.first);
}

void solveRestrictedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix, MolecularOrbitals& coefficientMatrix,
                                      SingleParticleEnergies& singleParticleEnergies) {
  if (fockMatrix.restrictedMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyRestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyRestrictedEnergies();
    return;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
  es.compute(fockMatrix.restrictedMatrix());
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(es.eigenvectors());
  singleParticleEnergies.setRestricted(es.eigenvalues());
}

void solveRestrictedGeneralizedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix, const Eigen::MatrixXd& overlapMatrix,
                                                 MolecularOrbitals& coefficientMatrix,
                                                 SingleParticleEnergies& singleParticleEnergies) {
  if (fockMatrix.restrictedMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyRestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyRestrictedEnergies();
    return;
  }

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(fockMatrix.restrictedMatrix(), overlapMatrix);
  coefficientMatrix = MolecularOrbitals::createFromRestrictedCoefficients(ges.eigenvectors());
  singleParticleEnergies.setRestricted(ges.eigenvalues());
}

void solveUnrestrictedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix, MolecularOrbitals& coefficientMatrix,
                                        SingleParticleEnergies& singleParticleEnergies) {
  if (fockMatrix.alphaMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyUnrestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyUnrestrictedEnergies();
    return;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;

  es.compute(fockMatrix.alphaMatrix());
  Eigen::MatrixXd alphaCoefficients = es.eigenvectors();
  Eigen::VectorXd alpha = es.eigenvalues();

  es.compute(fockMatrix.betaMatrix());
  Eigen::MatrixXd betaCoefficients = es.eigenvectors();
  Eigen::VectorXd beta = es.eigenvalues();

  coefficientMatrix =
      MolecularOrbitals::createFromUnrestrictedCoefficients(std::move(alphaCoefficients), std::move(betaCoefficients));

  singleParticleEnergies.setUnrestricted(alpha, beta);
}

void solveUnrestrictedGeneralizedEigenvalueProblem(const SpinAdaptedMatrix& fockMatrix, const Eigen::MatrixXd& overlapMatrix,
                                                   MolecularOrbitals& coefficientMatrix,
                                                   SingleParticleEnergies& singleParticleEnergies) {
  if (fockMatrix.alphaMatrix().size() == 0) {
    coefficientMatrix = MolecularOrbitals::createEmptyUnrestrictedOrbitals();
    singleParticleEnergies = SingleParticleEnergies::createEmptyUnrestrictedEnergies();
    return;
  }

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;

  ges.compute(fockMatrix.alphaMatrix(), overlapMatrix);
  Eigen::MatrixXd alphaCoefficients = ges.eigenvectors();
  Eigen::VectorXd alpha = ges.eigenvalues();

  ges.compute(fockMatrix.betaMatrix(), overlapMatrix);
  Eigen::MatrixXd betaCoefficients = ges.eigenvectors();
  Eigen::VectorXd beta = ges.eigenvalues();

  coefficientMatrix =
      MolecularOrbitals::createFromUnrestrictedCoefficients(std::move(alphaCoefficients), std::move(betaCoefficients));

  singleParticleEnergies.setUnrestricted(alpha, beta);
}

void calculateRestrictedDensityMatrix(DensityMatrix& densityMatrix, const MolecularOrbitals& coefficientMatrix, int nElectrons) {
  assert(nElectrons >= 0);
  const auto& C = coefficientMatrix.restrictedMatrix();
  auto nAOs = C.cols();
  Eigen::MatrixXd P = 2 * C.block(0, 0, nAOs, nElectrons / 2) * C.block(0, 0, nAOs, nElectrons / 2).transpose();

  if ((nElectrons % 2) != 0) { // if odd number of electrons
    P += C.col(nElectrons / 2) * C.col(nElectrons / 2).transpose();
  }

  densityMatrix.setDensity(std::move(P), nElectrons);
}

void calculateUnrestrictedDensityMatrices(DensityMatrix& densityMatrix, const MolecularOrbitals& coefficientMatrix,
                                          int nElectrons, int spinMultiplicity) {
  assert(nElectrons >= 0);
  assert(spinMultiplicity >= 1);

  int nAlpha, nBeta;
  getNumberUnrestrictedElectrons(nAlpha, nBeta, nElectrons, spinMultiplicity);

  const auto& cA = coefficientMatrix.alphaMatrix();
  const auto& cB = coefficientMatrix.betaMatrix();
  auto nAOs = cA.cols();
  Eigen::MatrixXd alphaMatrix = cA.block(0, 0, nAOs, nAlpha) * cA.block(0, 0, nAOs, nAlpha).transpose();
  Eigen::MatrixXd betaMatrix = cB.block(0, 0, nAOs, nBeta) * cB.block(0, 0, nAOs, nBeta).transpose();
  densityMatrix.setDensity(std::move(alphaMatrix), std::move(betaMatrix), nAlpha, nBeta);
}

void calculateRestrictedEnergyWeightedDensityMatrix(Eigen::MatrixXd& energyWeightedDensityMatrix,
                                                    const MolecularOrbitals& coefficientMatrix,
                                                    const SingleParticleEnergies& singleParticleEnergies, int nElectrons) {
  assert(nElectrons >= 0);
  const auto& C = coefficientMatrix.restrictedMatrix();
  auto nAOs = C.cols();

  Eigen::MatrixXd CEn(nAOs, (nElectrons + 1) / 2); // Eigenvector matrix multiplied by energy eigenvalues

  for (int i = 0; i < (nElectrons + 1) / 2; i++)
    CEn.col(i) = C.col(i) * singleParticleEnergies.getRestrictedEnergies()[i];

  energyWeightedDensityMatrix =
      2 * C.block(0, 0, nAOs, (nElectrons) / 2) * CEn.block(0, 0, nAOs, (nElectrons) / 2).transpose();

  if ((nElectrons % 2) != 0) // if odd number of electrons
    energyWeightedDensityMatrix += C.col(nElectrons / 2) * CEn.col(nElectrons / 2).transpose();
}

void calculateUnrestrictedEnergyWeightedDensityMatrix(Eigen::MatrixXd& energyWeightedDensityMatrix,
                                                      const MolecularOrbitals& coefficientMatrix,
                                                      const SingleParticleEnergies& singleParticleEnergies,
                                                      int nElectrons, int spinMultiplicity) {
  assert(nElectrons >= 0);
  assert(spinMultiplicity >= 1);

  int nAlpha, nBeta;
  getNumberUnrestrictedElectrons(nAlpha, nBeta, nElectrons, spinMultiplicity);

  const auto& cA = coefficientMatrix.alphaMatrix();
  const auto& cB = coefficientMatrix.betaMatrix();
  auto nAOs = cA.cols();

  Eigen::MatrixXd CEnAlpha(nAOs, nAlpha); // Alpha Eigenvector matrix multiplied by energy eigenvalues
  Eigen::MatrixXd CEnBeta(nAOs, nBeta);   // Beta Eigenvector matrix multiplied by energy eigenvalues

  for (int i = 0; i < nAlpha; i++)
    CEnAlpha.col(i) = cA.col(i) * singleParticleEnergies.getAlphaEnergies()[i];
  for (int i = 0; i < nBeta; i++)
    CEnBeta.col(i) = cB.col(i) * singleParticleEnergies.getBetaEnergies()[i];

  energyWeightedDensityMatrix = cA.block(0, 0, nAOs, nAlpha) * CEnAlpha.block(0, 0, nAOs, nAlpha).transpose() +
                                cB.block(0, 0, nAOs, nBeta) * CEnBeta.block(0, 0, nAOs, nBeta).transpose();
}

void calculateBondOrderMatrix(Utils::BondOrderCollection& bondOrderMatrix, const DensityMatrix& densityMatrix,
                              const Eigen::MatrixXd& overlapMatrix, const AtomsOrbitalsIndexes& aoIndexes) {
  // Bond order analysis (Mayer bond order)
  Eigen::MatrixXd PS = densityMatrix.restrictedMatrix() * overlapMatrix; // TODO: WHAT IF OVERLAP MATRIX IS ONLY
                                                                         // LOWER-TRIANGULAR?
  for (int i = 1; i < aoIndexes.getNAtomicOrbitals(); i++)
    for (int j = 0; j < i; j++)
      PS(i, j) = PS(i, j) * PS(j, i);
  for (int a = 1; a < aoIndexes.getNAtoms(); a++) {
    for (int b = 0; b < a; b++) {
      double order = PS.block(aoIndexes.getFirstOrbitalIndex(a), aoIndexes.getFirstOrbitalIndex(b),
                              aoIndexes.getNOrbitals(a), aoIndexes.getNOrbitals(b))
                         .sum();
      bondOrderMatrix.setOrder(a, b, order);
    }
  }
}

void calculateOrthonormalBondOrderMatrix(Utils::BondOrderCollection& bondOrderMatrix,
                                         const DensityMatrix& densityMatrix, const AtomsOrbitalsIndexes& aoIndexes) {
  // Bond order analysis (Mayer bond order), no need for overlap since it is no generalized eigenvalue problem
  const auto& P = densityMatrix.restrictedMatrix();
  Eigen::MatrixXd P2 = P.cwiseProduct(P);

  for (int a = 1; a < aoIndexes.getNAtoms(); a++) {
    for (int b = 0; b < a; b++) {
      double order = P2.block(aoIndexes.getFirstOrbitalIndex(a), aoIndexes.getFirstOrbitalIndex(b),
                              aoIndexes.getNOrbitals(a), aoIndexes.getNOrbitals(b))
                         .sum();
      bondOrderMatrix.setOrder(a, b, order);
    }
  }
}

void calculateOrthonormalAtomicCharges(std::vector<double>& mullikenCharges, const std::vector<double>& coreCharges,
                                       const DensityMatrix& densityMatrix, const AtomsOrbitalsIndexes& aoIndexes) {
  // in the case of an orthonormal basis, the partial charge of an atom is given as
  // q_A = Z_A - \sum_{i in A} P_ii
  for (int a = 0; a < aoIndexes.getNAtoms(); a++) {
    mullikenCharges[a] = coreCharges[a];
    int nAOsA = aoIndexes.getNOrbitals(a);
    int index = aoIndexes.getFirstOrbitalIndex(a);

    double nElectronsOnAtom = densityMatrix.restrictedMatrix().block(index, index, nAOsA, nAOsA).trace();
    mullikenCharges[a] = coreCharges[a] - nElectronsOnAtom;
  }
}

void calculateMullikenAtomicCharges(std::vector<double>& mullikenCharges, const std::vector<double>& coreCharges,
                                    const DensityMatrix& densityMatrix, const Eigen::MatrixXd& overlapMatrix,
                                    const AtomsOrbitalsIndexes& aoIndexes) {
  // Calculation of the Mulliken charges as described in elstner1998
  Eigen::MatrixXd D = densityMatrix.restrictedMatrix().cwiseProduct(overlapMatrix); // population matrix

  for (int a = 0; a < aoIndexes.getNAtoms(); a++) {
    mullikenCharges[a] = coreCharges[a];
    int nAOsA = aoIndexes.getNOrbitals(a);
    int index = aoIndexes.getFirstOrbitalIndex(a);

    for (int mu = 0; mu < nAOsA; mu++) {
      for (int nu = 0; nu < aoIndexes.getNAtomicOrbitals(); nu++)
        mullikenCharges[a] -= D(index + mu, nu);
    }
  }
}

} // namespace LcaoUtils
} // namespace Utils
} // namespace Scine
