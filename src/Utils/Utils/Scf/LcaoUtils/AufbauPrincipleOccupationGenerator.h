/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.\n
 *            See LICENSE.txt for details.
 */

#ifndef UTILS_AUFBAUPRINCIPLEOCCUPATIONGENERATOR_H
#define UTILS_AUFBAUPRINCIPLEOCCUPATIONGENERATOR_H

#include <Utils/Scf/LcaoUtils/ElectronicOccupationGenerator.h>

namespace Scine {
namespace Utils {

namespace LcaoUtils {

class AufbauPrincipleOccupationGenerator : public ElectronicOccupationGenerator {
 public:
 private:
  ElectronicOccupation generateOccupationImpl() override;
};

} // namespace LcaoUtils

} // namespace Utils
} // namespace Scine
#endif // UTILS_AUFBAUPRINCIPLEOCCUPATIONGENERATOR_H
