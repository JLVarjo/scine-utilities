/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#ifndef UTILS_EXTERNALQC_ORCAHESSIANOUTPUTPARSER_H
#define UTILS_EXTERNALQC_ORCAHESSIANOUTPUTPARSER_H

#include <Utils/Typenames.h>
#include <Eigen/Core>
#include <string>

namespace Scine {
namespace Utils {
namespace ExternalQC {

/**
 * @class OrcaHessianOutputParser OrcaHessianOutputParser.h
 * @brief This class parses information out of the ORCA hessian output file.
 */
class OrcaHessianOutputParser {
 public:
  /**
   * @brief Constructor.
   * @param outputFileName Name of the Hessian output file (*.hess)
   */
  explicit OrcaHessianOutputParser(const std::string& outputFileName);
  /**
   * @brief Parse the Hessian matrix from the output file.
   * @return The Hessian matrix.
   */
  [[deprecated("Prefer static variant with filename arg")]] HessianMatrix getHessian() const;

  static HessianMatrix getHessian(const std::string& outputFileName);

 private:
  static HessianMatrix extractHessian(const std::string& content);
  // Helper function for parsing the Hessian matrix
  static std::string extractContent(const std::string& filename);
  static void readUntilHessianKeyword(std::istream& in);
  static int getNumberAtomsFromHessianOutput(std::istream& in);
  static void ignoreFirstBlockLine(std::istream& in);
  static void readOneBlock(std::istream& in, Eigen::MatrixXd& m, int atomCount, int firstBlockColumnIndex);

  std::string content_;
};

} // namespace ExternalQC
} // namespace Utils
} // namespace Scine

#endif // UTILS_EXTERNALQC_ORCAHESSIANOUTPUTPARSER_H
