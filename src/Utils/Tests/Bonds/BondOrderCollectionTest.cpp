/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include "Utils/Bonds/BondOrderCollection.h"
#include <gmock/gmock.h>
#include <Eigen/SparseCore>

using namespace testing;

namespace Scine::Utils::Tests {

TEST(BondOrderCollectionTest, Empty) {
  int size = 3;
  BondOrderCollection b;
  b.resize(3);
  ASSERT_EQ(b.getSystemSize(), size);
  ASSERT_EQ(b.getOrder(0, 0), 0);
  ASSERT_EQ(b.getOrder(0, 1), 0);
  b.setOrder(0, 0, 0);
  ASSERT_TRUE(b.empty());
}

TEST(BondOrderCollectionTest, RangeCheckWorks) {
  int size = 3;
  BondOrderCollection b;
  b.resize(3);
  ASSERT_EQ(b.getSystemSize(), size);
  ASSERT_THROW(b.getOrder(4, 0), std::runtime_error);
  ASSERT_THROW(b.getOrder(2, 10), std::runtime_error);
  ASSERT_THROW(b.getOrder(-10, 0), std::runtime_error);
  ASSERT_THROW(b.getOrder(2, -2), std::runtime_error);
  ASSERT_THROW(b.setOrder(4, 0, 1.0), std::runtime_error);
  ASSERT_THROW(b.setOrder(2, 10, 1.0), std::runtime_error);
  ASSERT_THROW(b.setOrder(-10, 0, 1.0), std::runtime_error);
  ASSERT_THROW(b.setOrder(2, -2, 1.0), std::runtime_error);
}

TEST(BondOrderCollectionTest, SetZero) {
  int size = 3;
  BondOrderCollection b;
  b.resize(size);
  b.setOrder(0, 1, 3.0);
  ASSERT_EQ(b.getOrder(0, 1), 3.0);
  ASSERT_EQ(b.getOrder(1, 0), 3.0);
  b.setZero();
  ASSERT_EQ(b.getOrder(0, 1), 0);
  ASSERT_EQ(b.getOrder(1, 0), 0);
}

TEST(BondOrderCollectionTest, SetMatrixAndCompare) {
  BondOrderCollection b1;
  BondOrderCollection b2;
  BondOrderCollection b3;
  Eigen::SparseMatrix<double> same(3, 3);
  Eigen::SparseMatrix<double> diff(3, 3);

  same.insert(0, 1) = 5.0;
  same.insert(1, 0) = 5.0;
  same.insert(2, 2) = 2.2;

  diff.insert(1, 1) = 4.0;
  diff.insert(2, 1) = 3.0;
  diff.insert(1, 2) = 3.0;

  b1.setMatrix(same);
  b2.setMatrix(same);
  b3.setMatrix(diff);

  ASSERT_TRUE(b1 == b2);
  ASSERT_TRUE(b1 != b3);
  ASSERT_FALSE(b1 == b3);
  ASSERT_FALSE(b1 != b2);
}

TEST(BondOrderCollectionTest, SizeOne) {
  BondOrderCollection b;
  b.resize(1);
  b.setOrder(0, 0, 1);
  ASSERT_EQ(b.getOrder(0, 0), 1);
}

TEST(BondOrderCollectionTest, SizeTwo) {
  BondOrderCollection b;
  b.resize(2);
  b.setOrder(0, 1, 1);
  ASSERT_EQ(b.getOrder(0, 1), 1);
  ASSERT_EQ(b.getOrder(1, 0), 1);
}

TEST(BondOrderCollectionTest, AbsoluteValues) {
  BondOrderCollection b;
  b.resize(3);
  b.setOrder(0, 1, -1);
  b.setOrder(0, 2, 1);
  ASSERT_EQ(b.getOrder(0, 1), -1);
  ASSERT_EQ(b.getOrder(0, 2), 1);
  b.setToAbsoluteValues();
  ASSERT_EQ(b.getOrder(0, 1), 1);
  ASSERT_EQ(b.getOrder(0, 2), 1);
}

TEST(BondOrderCollectionTest, RemoveAtoms) {
  Eigen::MatrixXd bondOrders = Eigen::MatrixXd::Zero(6, 6);
  // 6 atoms in 2 molecules with 3 atoms each. For instance, two water molecules.
  // For the sake of the unit test, I also added a bond between the central atoms of the molecules.
  bondOrders << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0;
  BondOrderCollection bondOrderCollection(6);
  bondOrderCollection.setMatrix(bondOrders.sparseView());
  ASSERT_EQ(bondOrderCollection.getSystemSize(), 6);
  ASSERT_EQ(bondOrderCollection.getOrder(1, 0), 1);
  // Remove one of the molecules.
  bondOrderCollection.removeAtomsByIndices({0, 1, 2});
  ASSERT_EQ(bondOrderCollection.getSystemSize(), 3);
  ASSERT_EQ(bondOrderCollection.getOrder(1, 0), 1);
  ASSERT_EQ(bondOrderCollection.getOrder(2, 0), 0);
  ASSERT_EQ(bondOrderCollection.getOrder(2, 1), 1);
}

} // namespace Scine::Utils::Tests
