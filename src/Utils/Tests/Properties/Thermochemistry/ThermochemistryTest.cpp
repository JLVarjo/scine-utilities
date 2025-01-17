/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */

#include <Utils/GeometricDerivatives/NormalModeAnalysis.h>
#include <Utils/Geometry.h>
#include <Utils/Properties/Thermochemistry/ThermochemistryCalculator.h>
#include <Utils/Typenames.h>
#include <gmock/gmock.h>
#include <memory>

using namespace testing;

namespace Scine {
namespace Utils {

/*
 * Reference calculation performed with MOPAC2016.
 * PM6 PRECISE
 *
 *
 * C     0.000000  1   0.000000  1   0.000000 1
 * O     0.000000  1   0.000000  1   1.212200 1
 * H     0.937197  1   0.000000  1  -0.584262 1
 * H    -0.937197  1   0.000000  1  -0.584262 1
 *
 * PM6 RHF THERMO OLDGEO PRECISE
 *
 * and
 *
 * PM6 PRECISE
 *
 *
 * H     0  1   0.000000  0  0 0
 * F     1  1   0.000000  0  0 0
 *
 * PM6 RHF THERMO OLDGEO PRECISE
 */

class AThermochemistryTest : public Test {
 public:
  NormalModesContainer formaldehydeNormalModes;
  NormalModesContainer HFNormalModes;
  Geometry::Properties::PrincipalMomentsOfInertia formaldehydePMI;
  Geometry::Properties::PrincipalMomentsOfInertia hfPMI;
  std::unique_ptr<ThermochemistryCalculator> arbitraryTCCalculator;
  ElementTypeCollection formaldehydeElements;
  ElementTypeCollection hfElements;
  PositionCollection hfPositions = Eigen::MatrixX3d::Zero(2, 3);
  int formaldehydeMultiplicity = 1;
  int hfMultiplicity = 1;
  double arbitraryEnergy = 1.0;
  Eigen::MatrixXd hfHessian = Eigen::MatrixXd::Zero(6, 6);

 protected:
  void SetUp() final {
    formaldehydeElements = {ElementType::C, ElementType::O, ElementType::H, ElementType::H};
    NormalMode m1(1101.75, DisplacementCollection::Random(4, 3));
    formaldehydeNormalModes.add(m1);
    NormalMode m2(1157.94, DisplacementCollection::Random(4, 3));
    formaldehydeNormalModes.add(m2);
    NormalMode m3(1349.03, DisplacementCollection::Random(4, 3));
    formaldehydeNormalModes.add(m3);
    NormalMode m4(1791.24, DisplacementCollection::Random(4, 3));
    formaldehydeNormalModes.add(m4);
    NormalMode m5(2614.79, DisplacementCollection::Random(4, 3));
    formaldehydeNormalModes.add(m5);
    NormalMode m6(2664.54, DisplacementCollection::Random(4, 3));
    formaldehydeNormalModes.add(m6);
    Eigen::Vector3d eigenValues(2.8969, 21.7672, 24.6640);
    // Convert from 1e-40 g cm^2 to amu*bohr^2
    eigenValues *= 1e-47 * Constants::u_per_kg * std::pow(Constants::bohr_per_meter, 2);
    formaldehydePMI.eigenvalues = eigenValues;
    formaldehydePMI.eigenvectors = Eigen::Matrix3d::Random();

    hfElements = {ElementType::H, ElementType::F};
    // clang-format off
    hfPositions << 0.0000000000000,    0.0000000000000,    0.0000000000000,
                   0.9655884052935,    0.0000000000000,   -0.0000001000000;
    // clang-format on
    hfPositions *= Constants::bohr_per_angstrom;
    NormalMode m1HF(3968.7, DisplacementCollection::Random(1, 3));
    HFNormalModes.add(m1HF);
    Eigen::Vector3d eigenValuesHF(0.00000000, 1.4818, 1.4818);
    // Convert from 1e-40 g cm^2 to amu*bohr^2
    eigenValuesHF *= 1e-47 * Constants::u_per_kg * std::pow(Constants::bohr_per_meter, 2);
    hfPMI.eigenvalues = eigenValuesHF;
    hfPMI.eigenvectors = Eigen::Matrix3d::Random();
    // clang-format off
    // ref matrix lower triangular in MILLIDYNES/ANGSTROM/SQRT(MASS(I)*MASS(J))
    std::vector<double> refMatrixTriangular = {
       8.8120622747042,  0.0000000050605,   0.0047603389882,  -0.0000009019966,  -0.0000000000000,   0.0047603389883,
      -2.0296809143334, -0.0000000011656,   0.0000002077567,   0.4674960849783,  -0.0000000011656,  -0.0010964357188,
       0.0000000000000,  0.0000000002685,   0.0002525390079,   0.0000002077567,   0.0000000000000,  -0.0010964357188,
      -0.0000000478526, -0.0000000000000,   0.0002525390079};
    // clang-format on
    int count = 0;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j <= i; ++j) {
        hfHessian(i, j) = refMatrixTriangular[count];
        hfHessian(j, i) = refMatrixTriangular[count];
        count++;
      }
    }
    // Back-scale the mass-weighted coordinates to Cartesian coordinates.
    auto masses = Geometry::Properties::getMasses(hfElements);
    for (unsigned long i = 0; i < masses.size(); ++i) {
      hfHessian.middleRows(3 * i, 3) *= std::sqrt(masses[i]);
      hfHessian.middleCols(3 * i, 3) *= std::sqrt(masses[i]);
    }
    // unit conversion, original is milliDyn / angstrom
    hfHessian *= 1e-8;                                      // N / Angstrom = kg m / (s^2 angstrom)
    hfHessian *= Constants::meter_per_angstrom;             // J / angstrom^2
    hfHessian *= Constants::hartree_per_joule;              // hartree / angstrom^2
    hfHessian *= std::pow(Constants::angstrom_per_bohr, 2); // hartree / bohr^2
  }
};

TEST_F(AThermochemistryTest, CanConstructCalculator) {
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(
      formaldehydeNormalModes, formaldehydePMI, formaldehydeElements, formaldehydeMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator =
      std::make_unique<ThermochemistryCalculator>(hfHessian, hfElements, hfPositions, hfMultiplicity, arbitraryEnergy);
  AtomCollection atoms = AtomCollection(hfElements, hfPositions);
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(hfHessian, atoms, hfMultiplicity, arbitraryEnergy);
}

TEST_F(AThermochemistryTest, DifferentConstructionsAreEqual) {
  arbitraryTCCalculator =
      std::make_unique<ThermochemistryCalculator>(HFNormalModes, hfPMI, hfElements, hfMultiplicity, arbitraryEnergy);
  // C2v symmetry
  arbitraryTCCalculator->setMolecularSymmetryNumber(2);
  ThermochemicalComponentsContainer container1 = arbitraryTCCalculator->calculate();
  arbitraryTCCalculator =
      std::make_unique<ThermochemistryCalculator>(hfHessian, hfElements, hfPositions, hfMultiplicity, arbitraryEnergy);
  // C2v symmetry
  arbitraryTCCalculator->setMolecularSymmetryNumber(2);
  ThermochemicalComponentsContainer container2 = arbitraryTCCalculator->calculate();
  AtomCollection atoms = AtomCollection(hfElements, hfPositions);
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(hfHessian, atoms, hfMultiplicity, arbitraryEnergy);
  // C2v symmetry
  arbitraryTCCalculator->setMolecularSymmetryNumber(2);
  ThermochemicalComponentsContainer container3 = arbitraryTCCalculator->calculate();
  ASSERT_TRUE(container1.isApprox(container2, 1e-6));
  ASSERT_TRUE(container1.isApprox(container3, 1e-6));
  ASSERT_TRUE(container2.isApprox(container3, 1e-12));
}

TEST_F(AThermochemistryTest, CorrectlyCalculatesZPVE) {
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(
      formaldehydeNormalModes, formaldehydePMI, formaldehydeElements, formaldehydeMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator->setZPVEInclusion(ZPVEInclusion::alreadyIncluded);
  arbitraryTCCalculator->setTemperature(298.15);
  auto container = arbitraryTCCalculator->calculate();
  ASSERT_THAT(container.overall.symmetryNumber, Eq(1));
  ASSERT_THAT(container.vibrationalComponent.zeroPointVibrationalEnergy * Constants::kCalPerMol_per_hartree,
              DoubleNear(15.267, 1e-3));
}

TEST_F(AThermochemistryTest, CorrectlyCalculatesZPVEForLinearMolecule) {
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(HFNormalModes, formaldehydePMI, formaldehydeElements,
                                                                      formaldehydeMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator->setZPVEInclusion(ZPVEInclusion::alreadyIncluded);
  arbitraryTCCalculator->setTemperature(298.15);

  auto container = arbitraryTCCalculator->calculate();
  ASSERT_THAT(container.vibrationalComponent.zeroPointVibrationalEnergy * Constants::kCalPerMol_per_hartree,
              DoubleNear(5.674, 1e-3));
}

TEST_F(AThermochemistryTest, CanCalculateVibrationalComponent) {
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(
      formaldehydeNormalModes, formaldehydePMI, formaldehydeElements, formaldehydeMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator->setZPVEInclusion(ZPVEInclusion::alreadyIncluded);
  arbitraryTCCalculator->setTemperature(298.00);
  auto container = arbitraryTCCalculator->calculate();

  EXPECT_THAT(container.vibrationalComponent.heatCapacityP * Constants::kCalPerMol_per_hartree,
              DoubleNear(0.6650 / 1000, 1e-6));
  EXPECT_THAT(container.vibrationalComponent.entropy * Constants::kCalPerMol_per_hartree, DoubleNear(0.1365 / 1000, 1e-6));
  EXPECT_THAT(container.vibrationalComponent.enthalpy * Constants::kCalPerMol_per_hartree, DoubleNear(0.0345759, 1e-6));
}

TEST_F(AThermochemistryTest, CanCalculateRotationalComponentHF) {
  arbitraryTCCalculator =
      std::make_unique<ThermochemistryCalculator>(HFNormalModes, hfPMI, hfElements, hfMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator->setZPVEInclusion(ZPVEInclusion::alreadyIncluded);
  arbitraryTCCalculator->setTemperature(298.00);
  auto container = arbitraryTCCalculator->calculate();
  ASSERT_THAT(container.rotationalComponent.enthalpy * Constants::kCalPerMol_per_hartree, DoubleNear(592.1875 / 1000, 1e-5));
  ASSERT_THAT(container.rotationalComponent.heatCapacityP * Constants::kCalPerMol_per_hartree, DoubleNear(1.9872 / 1000, 1e-5));
  ASSERT_THAT(container.rotationalComponent.entropy * Constants::kCalPerMol_per_hartree, DoubleNear(6.7458 / 1000, 1e-5));
}

TEST_F(AThermochemistryTest, CanCalculateRotationalComponentOfAsymmetricMolecule) {
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(
      formaldehydeNormalModes, formaldehydePMI, formaldehydeElements, formaldehydeMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator->setZPVEInclusion(ZPVEInclusion::alreadyIncluded);
  arbitraryTCCalculator->setTemperature(298.00);
  // C2v symmetry
  arbitraryTCCalculator->setMolecularSymmetryNumber(2);
  auto container = arbitraryTCCalculator->calculate();
  ASSERT_THAT(container.overall.symmetryNumber, Eq(2));
  ASSERT_THAT(container.rotationalComponent.enthalpy * Constants::kCalPerMol_per_hartree, DoubleNear(888.2813 / 1000, 1e-5));
  ASSERT_THAT(container.rotationalComponent.heatCapacityP * Constants::kCalPerMol_per_hartree, DoubleNear(2.9808 / 1000, 1e-5));
  ASSERT_THAT(container.rotationalComponent.entropy * Constants::kCalPerMol_per_hartree, DoubleNear(16.0088 / 1000, 1e-5));
}

TEST_F(AThermochemistryTest, CanCalculateTranslationalComponentOfAsymmetricMolecule) {
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(
      formaldehydeNormalModes, formaldehydePMI, formaldehydeElements, formaldehydeMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator->setZPVEInclusion(ZPVEInclusion::alreadyIncluded);
  arbitraryTCCalculator->setTemperature(298.00);
  // C2v symmetry
  arbitraryTCCalculator->setMolecularSymmetryNumber(2);
  auto container = arbitraryTCCalculator->calculate();
  ASSERT_THAT(container.translationalComponent.enthalpy * Constants::kCalPerMol_per_hartree,
              DoubleNear(1480.4688 / 1000, 1e-5));
  ASSERT_THAT(container.translationalComponent.heatCapacityP * Constants::kCalPerMol_per_hartree,
              DoubleNear(4.9680 / 1000, 1e-5));
  ASSERT_THAT(container.translationalComponent.entropy * Constants::kCalPerMol_per_hartree, DoubleNear(36.1295 / 1000, 1e-5));
}

TEST_F(AThermochemistryTest, CanCalculateOverallComponent) {
  arbitraryTCCalculator = std::make_unique<ThermochemistryCalculator>(
      formaldehydeNormalModes, formaldehydePMI, formaldehydeElements, formaldehydeMultiplicity, arbitraryEnergy);
  arbitraryTCCalculator->setZPVEInclusion(ZPVEInclusion::alreadyIncluded);
  arbitraryTCCalculator->setTemperature(298.00);
  // C2v symmetry
  arbitraryTCCalculator->setMolecularSymmetryNumber(2);
  auto container = arbitraryTCCalculator->calculate();
  ASSERT_THAT(container.overall.enthalpy * Constants::kCalPerMol_per_hartree,
              DoubleNear(2403.3261 / 1000 + arbitraryEnergy * Constants::kCalPerMol_per_hartree, 3e-5));
  ASSERT_THAT(container.overall.heatCapacityP * Constants::kCalPerMol_per_hartree, DoubleNear(8.6138 / 1000, 3e-5));
  ASSERT_THAT(container.overall.entropy * Constants::kCalPerMol_per_hartree, DoubleNear(52.2772 / 1000, 3e-5));
}
} // namespace Utils
} // namespace Scine
