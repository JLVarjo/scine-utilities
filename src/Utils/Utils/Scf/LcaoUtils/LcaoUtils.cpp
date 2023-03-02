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
#include <Utils/Math/IterativeDiagonalizer/IndirectPreconditionerEvaluator.h>
#include <Utils/Math/IterativeDiagonalizer/IndirectSigmaVectorEvaluator.h>
#include <Utils/Scf/LcaoUtils/LcaoUtils.h>
#include <Utils/Constants.h>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>

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

  for (int i = 0; i < (nElectrons + 1) / 2; i++) {
    CEn.col(i) = C.col(i) * singleParticleEnergies.getRestrictedEnergies()[i];
  }

  energyWeightedDensityMatrix =
      2 * C.block(0, 0, nAOs, (nElectrons) / 2) * CEn.block(0, 0, nAOs, (nElectrons) / 2).transpose();

  // if odd number of electrons
  if ((nElectrons % 2) != 0) {
    energyWeightedDensityMatrix += C.col(nElectrons / 2) * CEn.col(nElectrons / 2).transpose();
  }
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

  for (int i = 0; i < nAlpha; i++) {
    CEnAlpha.col(i) = cA.col(i) * singleParticleEnergies.getAlphaEnergies()[i];
  }
  for (int i = 0; i < nBeta; i++) {
    CEnBeta.col(i) = cB.col(i) * singleParticleEnergies.getBetaEnergies()[i];
  }

  energyWeightedDensityMatrix = cA.block(0, 0, nAOs, nAlpha) * CEnAlpha.block(0, 0, nAOs, nAlpha).transpose() +
                                cB.block(0, 0, nAOs, nBeta) * CEnBeta.block(0, 0, nAOs, nBeta).transpose();
}

void calculateBondOrderMatrix(Utils::BondOrderCollection& bondOrderMatrix, const DensityMatrix& densityMatrix,
                              const Eigen::MatrixXd& overlapMatrix, const AtomsOrbitalsIndexes& aoIndexes) {
  bondOrderMatrix.resize(aoIndexes.getNAtoms());
  bondOrderMatrix.setZero();
  // Bond order analysis (Mayer bond order)
  Eigen::MatrixXd PS = densityMatrix.restrictedMatrix() * overlapMatrix; // TODO: WHAT IF OVERLAP MATRIX IS ONLY
                                                                         // LOWER-TRIANGULAR?
  for (int i = 1; i < aoIndexes.getNAtomicOrbitals(); i++) {
    for (int j = 0; j < i; j++) {
      PS(i, j) = PS(i, j) * PS(j, i);
    }
  }
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
  bondOrderMatrix.resize(aoIndexes.getNAtoms());
  bondOrderMatrix.setZero();
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
      for (int nu = 0; nu < aoIndexes.getNAtomicOrbitals(); nu++) {
        mullikenCharges[a] -= D(index + mu, nu);
      }
    }
  }
}

const size_t* getIrot() {

    /*
    "IROT IS A MAPPING LIST. FOR EACH ELEMENT OF AROT 5 NUMBERS ARE
    NEEDED. THESE ARE, IN ORDER, FIRST AND SECOND SUBSCRIPTS OF AROT,
    AND FIRST,SECOND, AND THIRD SUBSCRIPTS OF C, THUS THE FIRST
    LINE OF IROT DEFINES AROT(1,1)=C(1,3,3)"
    */

    static const size_t irot[] = {0, 0, 0, 2, 2, 1, 1, 1, 3, 2, 2, 1, 1, 1, 2, 3, 1, 1, 2, 2, 1, 2, 1, 3, 1, 2, 2, 1, 1, 1,
                              3, 2, 1, 2, 1, 1, 3, 1, 3, 3, 2, 3, 1, 1, 3, 3, 3, 1, 2, 3, 4, 4, 2, 0, 4, 5, 4, 2, 3, 2,
                              6, 4, 2, 2, 2, 7, 4, 2, 1, 2, 8, 4, 2, 4, 2, 4, 5, 2, 0, 1, 5, 5, 2, 3, 1, 6, 6, 2, 2, 1,
                              7, 5, 2, 1, 1, 8, 5, 2, 4, 1, 4, 6, 2, 0, 3, 5, 6, 2, 3, 3, 6, 6, 2, 2, 3, 7, 6, 2, 1, 3,
                              8, 6, 2, 4, 3, 4, 7, 2, 0, 0, 5, 7, 2, 3, 0, 6, 7, 2, 2, 0, 7, 7, 2, 1, 0, 8, 7, 2, 4, 0,
                              4, 8, 2, 0, 4, 5, 8, 2, 3, 4, 6, 8, 2, 2, 4, 7, 8, 2, 1, 4, 8, 8, 2, 4, 4};
    return irot;
}

const size_t* getIsp() {
    static const size_t isp[] = {0, 1, 2, 2, 3, 4, 4, 5, 5};
    return isp;
}


void coe(double x2, double y2, double z2, int nij, double& r, Eigen::VectorXd& c) {

    // "COE Utility: Within the general overlap routine COE calculates the angular coefficients for the s, p and d real atomic orbitals
    // given the axis and returns the rotation matrix. "

    const double rt34 = 0.86602540378444;
    const double rt13 = 0.57735026918963;
    double ca, cb, sa, sb;
    double xy = std::pow(x2, 2) + std::pow(y2, 2);
    r = std::sqrt(xy + std::pow(z2, 2));
    xy = std::sqrt(xy);
    if (xy>=1e-10) {
        ca = x2 / xy;
        cb = z2 / r;
        sa = y2 / xy;
        sb = xy / r;
    } else {
        if (z2 <= 0.0) {
            if (z2 != 0.0) {
                ca = -1.0;
                cb = -1.0;
                sa = 0.0; 
                sb = 0.0;
            }
            else{
                ca = 0.0;
                cb = 0.0;
                sa = 0.0;
                sb = 0.0;
            }
        }
        else {
          ca = 1.0; 
          cb = 1.0; 
          sa = 0.0; 
          sb = 0.0; 
        }
    }
    c.setZero();

    double c2a, c2b, s2a, s2b;
    c(37-1) = 1.0;
    if (nij >= 2) {
        c(56-1) = ca * cb;
        c(41-1) = ca * sb;
        c(26-1) = -sa;
        c(53-1) = -sb;
        c(38-1) = cb;
        c(23-1) = 0.0;
        c(50-1) = sa * cb;
        c(35-1) = sa * sb;
        c(20-1) = ca;
        if (nij >= 5) {
          c2a = 2 * ca * ca - 1.0;
          c2b = 2 * cb * cb - 1.0;
          s2a = 2 * sa * ca;
          s2b = 2 * sb * cb;
          c(75-1) = c2a * cb * cb + 0.5 * c2a * sb * sb;
          c(60-1) = 0.5 * c2a * s2b;
          c(45-1) = rt34 * c2a * sb * sb;
          c(30-1) = -s2a * sb;
          c(15-1) = -s2a * cb;
          c(72-1) = -0.5 * ca * s2b;
          c(57-1) = ca * c2b;
          c(42-1) = rt34 * ca * s2b;
          c(27-1) = -sa * cb;
          c(12-1) = sa * sb;
          c(69-1) = rt13 * sb * sb * 1.5;
          c(54-1) = -rt34 * s2b;
          c(39-1) = cb * cb - 0.5 * sb * sb;
          c(66-1) = -0.5 * sa * s2b;
          c(51-1) = sa * c2b;
          c(36-1) = rt34 * sa * s2b;
          c(21-1) = ca * cb;
          c(6-1) = -ca * sb;
          c(63-1) = s2a * cb * cb + 0.5 * s2a * sb * sb;
          c(48-1) = 0.5 * s2a * s2b;
          c(33-1) = rt34 * s2a * sb * sb;
          c(18-1) = c2a * sb;
          c(3-1) = c2a * cb;
        }
    }
}

void calculateSigmaPiDensityMatrix(Eigen::MatrixXd& sigmaPiDensityMatrix, const DensityMatrix& densityMatrix,
                                     const PositionCollection& positions,
                                     const AtomsOrbitalsIndexes& aoIndexes) {

    // PORTED FROM MOPAC 7.1 (PUBLIC DOMAIN) SUBROUTINE "DENROT":
    // "DENROT PRINTS THE DENSITY MATRIX AS(S - SIGMA, P - SIGMA, P - PI) RATHER THAN(S, PX, PY, PZ)."

    // RESULTING BOND ORBITAL POPULATIONS (AS AVAILABLE) ARE IN ORDER
    // 'S-SIGMA', 'P-SIGMA', 'P-PI', 'P-PI', 'D-SIGMA', 'D-PI ', 'D-PI ', 'D-DELL', 'D-DELL'
    
    // work matrices:
    Eigen::MatrixXd pab = Eigen::MatrixXd::Zero(9, 9);
    Eigen::MatrixXd arot = Eigen::MatrixXd::Zero(9, 9);
    Eigen::MatrixXd vect = Eigen::MatrixXd::Zero(9, 9);

    // statics:
    auto irot = getIrot();
    auto isp = getIsp();

    // for coe output:
    double r;
    Eigen::VectorXd c = Eigen::VectorXd::Zero(75);

    sigmaPiDensityMatrix.setZero();

    for (size_t i = 0; i < aoIndexes.getNAtoms(); i++) {
      
        const size_t ifirst = aoIndexes.getFirstOrbitalIndex(i);
        const size_t inorbs = aoIndexes.getNOrbitals(i);
        if (inorbs == 0) continue;

        // Note: in the Mopac 7.1 code ipq and jpq were limited as follows:
        // 
        // const int ipq = std::min(std::max((int)inorbs - 2, 1), 3);
        // 
        // which effectively skipped the rotations of any d-orbitals.
        // Can't see reason for this really as it seems to work.
        // So let's just activate it and see what happens...
        const int ipq = inorbs - 2;
        
        for (size_t j=0; j<i; j++) {

            const size_t jfirst = aoIndexes.getFirstOrbitalIndex(j);
            const size_t jnorbs = aoIndexes.getNOrbitals(j);
            if (jnorbs == 0) continue;

            //const int jpq = std::min(std::max(((int)jnorbs - 2), 1), 3); // see above
            const int jpq = jnorbs - 2;

            pab.setZero();
            pab.block(0, 0, inorbs, jnorbs) = densityMatrix.restrictedMatrix().block(ifirst, jfirst, inorbs, jnorbs);
            
            /*std::cout << "PAB FOR i,j = " << i << "," << j << std::endl;
            for (size_t di = 0; di < 9; di++) {
                for (size_t dj = 0; dj < 9; dj++) {
                    std::cout << pab.coeff(di, dj) << " ";
                }
                std::cout << std::endl;
            }*/

            const double delx = (positions(i, 0) - positions(j, 0)) / Constants::bohr_per_angstrom;
            const double dely = (positions(i, 1) - positions(j, 1)) / Constants::bohr_per_angstrom;
            const double delz = (positions(i, 2) - positions(j, 2)) / Constants::bohr_per_angstrom;
            coe(delx, dely, delz, std::max(ipq, jpq), r, c);
            
            arot.setZero();
            for (size_t i1 = 0; i1 < 35; i1++) {
                const size_t ac = irot[i1 * 5 + 0];
                const size_t ar = irot[i1 * 5 + 1];
                const size_t ci1 = irot[i1 * 5 + 2];
                const size_t ci2 = irot[i1 * 5 + 3];
                const size_t ci3 = irot[i1 * 5 + 4];
                const size_t cidx = ci1 + ci2 * 3 + ci3 * 3 * 5;
                arot.coeffRef(ac, ar) = c(cidx);
            }

            /*
            std::cout << "AROT FOR i,j = " << i << "," << j << std::endl;
            for (size_t di = 0; di < 9; di++) {
                for (size_t dj = 0; dj < 9; dj++) {
                    std::cout << arot.coeff(di, dj) << " ";
                }
                std::cout << std::endl;
            }
            */

            vect.setZero();

            const size_t maxnorbs = std::max(inorbs, jnorbs);
            for (size_t i1 = 0; i1 < inorbs; i1++) {
                for (size_t j1 = 0; j1 < jnorbs; j1++) {
                    double sum = 0;
                    for (size_t l1 = 0; l1 < maxnorbs; l1++) {
                            for (size_t l2 = 0; l2 < maxnorbs; l2++) {
                            sum = sum + arot.coeff(l1, i1) * pab.coeff(l1, l2) * arot.coeff(l2, j1);
                        }
                    }
                    vect.coeffRef(isp[i1], isp[j1]) = vect.coeff(isp[i1], isp[j1]) + std::pow(sum, 2);
                }
            }
            
            /*
            std::cout<< "VECT FOR i,j = " << i << "," << j << std::endl;
            for (size_t di=0; di<9; di++){
                for (size_t dj = 0; dj < 9; dj++) {
                    std::cout << vect.coeff(di, dj) << " ";
                }
                std::cout << std::endl;           
            }
            */

            sigmaPiDensityMatrix.block(jfirst, ifirst, jnorbs, inorbs) = vect.block(0, 0, inorbs, jnorbs).transpose();

        }
    }

    // PUT ATOMIC ORBITAL VALENCIES ONTO THE DIAGONAL
    const size_t dsize = densityMatrix.restrictedMatrix().innerSize();
    for (size_t di = 0; di < dsize; di++) {
        const double sumr = sigmaPiDensityMatrix.block(di, di + 1, 1, dsize - di - 1).sum();
        const double sumc = sigmaPiDensityMatrix.block(0, di, di, 1).sum();
        // std::cout << di + 1 << " " << di << " " << dsize - di - 1 << " sums: " << sumr << " & " << sumc << std::endl;
        sigmaPiDensityMatrix.coeffRef(di, di) = sumr + sumc;
    }

    /*
    std::cout << "-- RESULT MATRIX --"<< std::endl;
    for (size_t di = 0; di < dsize; di++) {
        for (size_t dj = 0; dj < dsize; dj++) {
            std::cout << sigmaPiDensityMatrix.coeff(di, dj) << " ";
        }
        std::cout << std::endl;
    }
    */

    return;
}

} // namespace LcaoUtils
} // namespace Utils
} // namespace Scine
