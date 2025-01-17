/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include "TurbomoleInputFileCreator.h"
#include "TurbomoleCalculatorSettings.h"
#include "TurbomoleHelper.h"
#include <Utils/CalculatorBasics.h>
#include <Utils/Geometry/AtomCollection.h>
#include <Utils/Geometry/ElementInfo.h>
#include <Utils/IO/NativeFilenames.h>
#include <Utils/Scf/LcaoUtils/SpinMode.h>
#include <math.h>
#include <boost/process.hpp>
#include <fstream>
#include <iomanip>

namespace bp = boost::process;

namespace Scine {
namespace Utils {
namespace ExternalQC {

TurbomoleInputFileCreator::TurbomoleInputFileCreator(std::string& calculationDirectory,
                                                     std::string& turbomoleExecutableBase, TurbomoleFiles& files)
  : calculationDirectory_(calculationDirectory), turbomoleExecutableBase_(turbomoleExecutableBase), files_(files) {
}

void TurbomoleInputFileCreator::createInputFiles(const AtomCollection& atoms, const Settings& settings) {
  writeCoordFile(atoms);
  prepareDefineSession(settings, atoms);
  runDefine();
  checkAndUpdateControlFile(settings);
}

void TurbomoleInputFileCreator::writeCoordFile(const AtomCollection& atoms) {
  std::ofstream coordStream;
  coordStream.open(files_.coordFile);
  coordStream << "$coord\n";
  for (const auto& atom : atoms) {
    std::string elementName = ElementInfo::symbol(atom.getElementType());
    std::for_each(elementName.begin(), elementName.end(), [](char& c) { c = ::tolower(c); });
    coordStream << atom.getPosition() << " " << elementName << std::endl;
  }
  coordStream << "$end";
  coordStream.close();
}

void TurbomoleInputFileCreator::prepareDefineSession(const Settings& settings, const AtomCollection& atoms) {
  // Check this before because define yields very strange output in case that charge and multiplicity don't match.
  CalculationRoutines::checkValidityOfChargeAndMultiplicity(settings.getInt(Utils::SettingsNames::molecularCharge),
                                                            settings.getInt(Utils::SettingsNames::spinMultiplicity), atoms);

  std::ofstream out;
  // out.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  out.open(files_.defineInputFile);

  out << "\n"
      << "\n"
      << "a coord"
      << "\n"
      << "*\nno\n";

  auto basisSet = settings.getString(Utils::SettingsNames::basisSet);
  TurbomoleHelper helper(calculationDirectory_, turbomoleExecutableBase_);
  helper.mapBasisSetToTurbomoleStringRepresentation(basisSet);
  out << "\nb all " << basisSet << "\n\n\n*\neht\n\n";

  out << settings.getInt(Scine::Utils::SettingsNames::molecularCharge) << "\n";

  auto spinMode = SpinModeInterpreter::getSpinModeFromString(settings.getString(Utils::SettingsNames::spinMode));
  auto multiplicity = settings.getInt(Scine::Utils::SettingsNames::spinMultiplicity);
  // If spin mode is not set or set to "restricted", let turbomole handle spin mode automatically (default is RHF)
  if (spinMode == SpinMode::Any) {
    out << "\n\n\n";
  }
  else if (spinMode == SpinMode::Restricted) {
    if (multiplicity != 1) {
      throw std::logic_error("Specified restricted spin for multiplicity larger than 1.");
    }
    out << "\n\n\n";
  }
  else if (spinMode == SpinMode::Unrestricted) {
    int numElectrons = multiplicity - 1;
    if (numElectrons == 0) {
      out << "no\ns\n*\n\n";
    }
    else {
      out << "no\nu " << numElectrons << "\n*\n\n";
    }
  }
  else if (spinMode == SpinMode::RestrictedOpenShell) {
    throw std::logic_error("Spin mode not implemented in Turbomole!");
  }
  else {
    throw std::logic_error("Specified unknown spin mode " + SpinModeInterpreter::getStringFromSpinMode(spinMode) +
                           " in settings."); // this should have been handled by settings
  }

  out << "ri\non\n\n";
  // Method
  auto methodInput = Scine::Utils::CalculationRoutines::splitIntoMethodAndDispersion(
      settings.getString(Scine::Utils::SettingsNames::method));
  helper.mapDftFunctionalToTurbomoleStringRepresentation(methodInput.first);
  out << "dft\non\nfunc " << methodInput.first << "\n\n";
  // Dispersion Correction
  if (!methodInput.second.empty()) {
    // make sure that dispersion is uppercase only
    std::for_each(methodInput.second.begin(), methodInput.second.end(), [](char& c) { c = ::toupper(c); });
    auto it = std::find(availableDispersionParams_.begin(), availableDispersionParams_.end(), methodInput.second);
    if (it - availableDispersionParams_.begin() == 0) {
      out << "dsp\non\n\n";
    }
    else if (it - availableDispersionParams_.begin() == 1) {
      out << "dsp\nbj\n\n";
    }
    else if (it - availableDispersionParams_.begin() == 2) {
      out << "dsp\nd4\n\n";
    }
    else {
      throw std::runtime_error("Invalid dispersion correction!");
    }
  }
  auto maxScfIterations = settings.getInt(Utils::SettingsNames::maxScfIterations);

  out << "scf\niter\n" << std::to_string(maxScfIterations) << "\n";
  out << "\n*";
  out.close();
}

void TurbomoleInputFileCreator::runDefine() {
  TurbomoleHelper helper(calculationDirectory_, turbomoleExecutableBase_);
  // empty the control file in case it is present already
  helper.emptyFile(files_.controlFile);
  helper.execute("define", files_.defineInputFile);
}

void TurbomoleInputFileCreator::checkAndUpdateControlFile(const Settings& settings) {
  bool basisSetIsCorrect = false;
  bool functionalIsCorrect = false;

  double scfConvCriterion = settings.getDouble(Utils::SettingsNames::selfConsistenceCriterion);
  bool scfDamping = settings.getBool(Utils::SettingsNames::scfDamping);
  auto scfOrbitalShift = settings.getDouble(SettingsNames::scfOrbitalShift);
  auto solvent = settings.getString(Utils::SettingsNames::solvent);

  if (!solvent.empty() && solvent != "none") {
    addSolvation(settings);
  }

  auto method =
      Scine::Utils::CalculationRoutines::splitIntoMethodAndDispersion(settings.getString(Scine::Utils::SettingsNames::method))
          .first;
  // make sure that functional is lowercase only
  std::for_each(method.begin(), method.end(), [](char& c) { c = ::tolower(c); });

  std::ifstream in;
  std::ofstream out;
  std::string controlBackupFile = NativeFilenames::combinePathSegments(calculationDirectory_, "control.bak");
  in.open(files_.controlFile);
  out.open(controlBackupFile);
  std::string line;

  auto basisSet = settings.getString(Utils::SettingsNames::basisSet);
  TurbomoleHelper helper(calculationDirectory_, turbomoleExecutableBase_);
  helper.mapBasisSetToTurbomoleStringRepresentation(basisSet);
  helper.mapDftFunctionalToTurbomoleStringRepresentation(method);

  while (std::getline(in, line)) {
    if ((line.find("$scfdamp") != std::string::npos) && scfDamping) {
      double dampingValue = settings.getDouble(SettingsNames::scfDampingValue);
      out << "$scfdamp   start=" << std::fixed << std::setprecision(3) << dampingValue << "  step=0.05  min=0.10\n";
    }
    else if ((line.find("scforbitalshift") != std::string::npos)) {
      out << "$scforbitalshift closedshell=" << scfOrbitalShift << "\n";
    }
    else if ((line.find("$drvopt") != std::string::npos) && (!settings.getString(SettingsNames::pointChargesFile).empty()))
      out << "$drvopt\n  point charges\n";
    else if (line.find("$end") != std::string::npos) {
      out << "$pop loewdin wiberg\n";
      auto pointChargesFile = settings.getString(SettingsNames::pointChargesFile);
      if (!pointChargesFile.empty()) {
        // open
        out << "$point_charges\n";
        std::ifstream pc;
        pc.open(pointChargesFile);
        std::string line;
        while (std::getline(pc, line)) {
          out << line << "\n";
        }
        pc.close();
        out << "$point_charge_gradients file=pc_gradient\n";
      }
      out << "$end";
    }
    else if (line.find("$scfconv") != std::string::npos) {
      out << "$scfconv " << int(round(-log10(scfConvCriterion))) << "\n";
    }
    else {
      out << line << "\n";
    }

    if (line.find(basisSet) != std::string::npos) {
      basisSetIsCorrect = true;
    }
    else if (line.find(method) != std::string::npos) {
      functionalIsCorrect = true;
    }

    auto spinMode = SpinModeInterpreter::getSpinModeFromString(settings.getString(Utils::SettingsNames::spinMode));
    if (line.find("$uhfmo") != std::string::npos && spinMode == SpinMode::Restricted) {
      throw std::runtime_error(
          "The requested restricted calculation was converted to an unrestricted calculation. Turbomole automatically "
          "assigned an UHF spin mode for half-open shell "
          "cases. If you still want to enforce singlet multiplicity, please choose UHF singlet instead.");
    }
  }

  in.close();
  out.close();
  std::rename(controlBackupFile.c_str(), files_.controlFile.c_str());

  if (!basisSetIsCorrect) {
    throw std::runtime_error("Your specified basis set " + basisSet + " is invalid. Check the spelling");
  }
  if (!functionalIsCorrect) {
    throw std::runtime_error("Your specified DFT functional " + method + " is invalid. Check the spelling");
  }
}

void TurbomoleInputFileCreator::addSolvation(const Settings& settings) {
  auto solvent = settings.getString(Utils::SettingsNames::solvent);

  std::for_each(solvent.begin(), solvent.end(), [](char& c) { c = ::tolower(c); });

  std::ofstream out;
  out.open(files_.solvationInputFile);

  std::unordered_map<std::string, std::pair<double, double>>::iterator it = availableSolventModels_.find(solvent);
  if (it != availableSolventModels_.end()) {
    out << it->second.first << "\n\n\n\n\n\n\n"
        << it->second.second << "\n\n\n\n"
        << "r all b"
        << "\n"
        << "*"
        << "\n\n\n";
  }
  else {
    throw std::runtime_error("The solvent '" + solvent + "' is currently not supported.");
  }
  out.close();

  // run cosmoprep
  auto directory = bp::start_dir(calculationDirectory_);
  bp::ipstream stdout, stderr;
  std::string outputFile = NativeFilenames::combinePathSegments(calculationDirectory_, "COSMO.out");
  std::string executable = NativeFilenames::combinePathSegments(turbomoleExecutableBase_, "cosmoprep");
  TurbomoleHelper helper(calculationDirectory_, turbomoleExecutableBase_);
  helper.execute("cosmoprep", files_.solvationInputFile);
}

} // namespace ExternalQC
} // namespace Utils
} // namespace Scine
