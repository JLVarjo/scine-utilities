/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include "TurbomoleNumberStringConverter.h"
#include <assert.h>

namespace Scine {
namespace Utils {
namespace ExternalQC {
namespace TurbomoleOrbitalPerturbation {

double TurbomoleNumberStringConverter::toDouble(const std::string& numberString) {
  assert(numberString.length() == 20);
  std::string s = numberString;
  s[16] = 'E';
  return std::stod(s);
}

std::string TurbomoleNumberStringConverter::toString(double number) {
  // Take care of extremal cases
  if (number == 0.0)
    return "0.00000000000000D+00";
  if (number < 1e-100 && number > -1e-100)
    return "0.00000000000000D+00";
  if (number >= 1e99)
    return "0.99999999999999D+99";
  if (number <= -1e99)
    return "-.99999999999999D+99";

  return getPrefix(number) + getNumberPart(number);
}

std::string TurbomoleNumberStringConverter::getPrefix(double number) {
  if (number > 0)
    return "0.";
  else
    return "-.";
}

std::string TurbomoleNumberStringConverter::getNumberPart(double number) {
  if (number < 0)
    number *= -1;

  int exponent = 14;
  while (number < 1e13) {
    number *= 10;
    --exponent;
  }
  auto n = static_cast<long long>(number + 0.5);
  return std::to_string(n) + getExponentPart(exponent);
}

std::string TurbomoleNumberStringConverter::getExponentPart(int exponent) {
  std::string s = "D";
  if (exponent < 0) {
    s += '-';
    exponent *= -1;
  }
  else
    s += '+';

  if (exponent < 10)
    s += '0';
  s += std::to_string(exponent);
  return s;
}

} // namespace TurbomoleOrbitalPerturbation
} // namespace ExternalQC
} // namespace Utils
} // namespace Scine
