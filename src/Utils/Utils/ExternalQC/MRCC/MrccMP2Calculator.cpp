/**
 * @file
 * @brief Default module that is always loaded by Core
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include "MrccMP2Calculator.h"

namespace Scine {
namespace Utils {
namespace ExternalQC {

std::string MrccMP2Calculator::getMethodFamily() const {
  return "MP2";
}

} // namespace ExternalQC
} // namespace Utils
} // namespace Scine
