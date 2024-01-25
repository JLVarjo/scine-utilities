/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.\n
 *            See LICENSE.txt for details.
 */

#include "gmock/gmock.h"
#include <Utils/DataStructures/SlaterToGaussian.h>

using namespace testing;
namespace Scine {
namespace Utils {
namespace Tests {

class ASlaterToGaussianTest : public Test {
 public:
  const unsigned int arbitraryN{2}, arbitraryL{4};
  const double arbitraryExponent{1.93834};
};

TEST_F(ASlaterToGaussianTest, ReturnsCorrectNumberOfParameters) {
  auto values = SlaterToGaussian::get(4, arbitraryN, arbitraryL);
  ASSERT_THAT(values.size(), Eq(4u));
}

TEST_F(ASlaterToGaussianTest, ScalesAlphaWithSquaredExponent) {
  auto values = SlaterToGaussian::get(4, arbitraryN, arbitraryL);
  auto scaledValues = SlaterToGaussian::get(4, arbitraryN, arbitraryL, arbitraryExponent);

  ASSERT_THAT(scaledValues[2].first, DoubleEq(values[2].first * arbitraryExponent * arbitraryExponent));
}

TEST_F(ASlaterToGaussianTest, ReturnsCorrectValuesForSTO_1G_1s) {
  auto values = SlaterToGaussian::get(1, 1, 0);
  ASSERT_THAT(values[0].first, DoubleEq(2.709496091e-1)); // 2.709498091e-1
  ASSERT_THAT(values[0].second, DoubleEq(1.0));
}

TEST_F(ASlaterToGaussianTest, ReturnsCorrectValuesForSTO_4G_4f) {
  auto values = SlaterToGaussian::get(4, 4, 3);
  ASSERT_THAT(values[0].first, DoubleEq(5.691670217e-1));
  ASSERT_THAT(values[0].second, DoubleEq(5.902730589e-2));
  ASSERT_THAT(values[1].first, DoubleEq(2.074585819e-1));
  ASSERT_THAT(values[1].second, DoubleEq(3.191828952e-1));
  ASSERT_THAT(values[2].first, DoubleEq(9.298346885e-2));
  ASSERT_THAT(values[2].second, DoubleEq(5.639423893e-1));
  ASSERT_THAT(values[3].first, DoubleEq(4.473508853e-2));
  ASSERT_THAT(values[3].second, DoubleEq(2.284796537e-1));
}

TEST_F(ASlaterToGaussianTest, ReturnsResultAsGTOExpansionInstance) {
  auto gtos = SlaterToGaussian::getGTOExpansion(4, 4, 3);
  auto values = SlaterToGaussian::get(4, 4, 3);
  ASSERT_THAT(gtos.gtfs.at(0).coefficient, Eq(values[0].second));
  ASSERT_THAT(gtos.gtfs.at(1).coefficient, Eq(values[1].second));
  ASSERT_THAT(gtos.gtfs.at(2).coefficient, Eq(values[2].second));
  ASSERT_THAT(gtos.gtfs.at(3).coefficient, Eq(values[3].second));
  ASSERT_THAT(gtos.gtfs.at(0).exponent, Eq(values[0].first));
  ASSERT_THAT(gtos.gtfs.at(1).exponent, Eq(values[1].first));
  ASSERT_THAT(gtos.gtfs.at(2).exponent, Eq(values[2].first));
  ASSERT_THAT(gtos.gtfs.at(3).exponent, Eq(values[3].first));
}

} // namespace Tests
} // namespace Utils
} // namespace Scine
