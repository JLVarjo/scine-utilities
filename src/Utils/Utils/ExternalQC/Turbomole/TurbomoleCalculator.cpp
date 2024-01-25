/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include "TurbomoleCalculator.h"
#include "Utils/ExternalQC/Exceptions.h"
#include "Utils/ExternalQC/ExternalProgram.h"
#include "Utils/ExternalQC/Turbomole/OrbitalSteering/TurbomoleOrbitalSteerer.h"
#include "Utils/ExternalQC/Turbomole/TurbomoleCalculatorSettings.h"
#include "Utils/ExternalQC/Turbomole/TurbomoleHelper.h"
#include "Utils/ExternalQC/Turbomole/TurbomoleInputFileCreator.h"
#include "Utils/ExternalQC/Turbomole/TurbomoleMainOutputParser.h"
#include "Utils/ExternalQC/Turbomole/TurbomoleState.h"
#include "Utils/IO/FilesystemHelpers.h"
#include "Utils/IO/NativeFilenames.h"
#include "Utils/MSVCCompatibility.h"
#include "Utils/Properties/Thermochemistry/ThermochemistryCalculator.h"
#include "Utils/Scf/LcaoUtils/SpinMode.h"
#include "Utils/Solvation/ImplicitSolvation.h"
#include <stdlib.h>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/process.hpp>
#include "../Utils/Utils/MSVCCompatibility.h"

namespace bp = boost::process;
namespace bfs = boost::filesystem;

namespace Scine {
namespace Utils {
namespace ExternalQC {

using namespace TurbomoleOrbitalPerturbation;

std::string TurbomoleCalculator::name() const {
  return "TURBOMOLE";
}

bool TurbomoleCalculator::supportsMethodFamily(const std::string& methodFamily) const {
  if (std::getenv("TURBODIR")) {
    return std::find(availableMethodFamilies_.begin(), availableMethodFamilies_.end(), methodFamily) !=
           availableMethodFamilies_.end();
  }

  return false;
}

TurbomoleCalculator::TurbomoleCalculator() {
  requiredProperties_ = Utils::Property::Energy;
  this->settings_ = std::make_unique<TurbomoleCalculatorSettings>();
  applySettings();
}

TurbomoleCalculator::TurbomoleCalculator(const TurbomoleCalculator& rhs) : CloneInterface(rhs) {
  this->requiredProperties_ = rhs.requiredProperties_;
  auto valueCollection = dynamic_cast<const Utils::UniversalSettings::ValueCollection&>(rhs.settings());
  this->settings_ =
      std::make_unique<Utils::Settings>(Utils::Settings(valueCollection, rhs.settings().getDescriptorCollection()));
  this->setLog(rhs.getLog());
  applySettings();
  atoms_ = rhs.atoms_;
  calculationDirectory_ = NativeFilenames::createRandomDirectoryName(baseWorkingDirectory_);
  results_ = rhs.results();
  this->turbomoleBinaryDir_ = rhs.turbomoleBinaryDir_;
  this->turbomoleSmpBinaryDir_ = rhs.turbomoleSmpBinaryDir_;
  this->turbomoleScriptsDir_ = rhs.turbomoleScriptsDir_;
  this->binaryHasBeenChecked_ = rhs.binaryHasBeenChecked_;
}

void TurbomoleCalculator::initializeProgram() {
  const char* turboRootEnv = std::getenv("TURBODIR");
  const char* paraArchEnv = std::getenv("PARA_ARCH");
  if (paraArchEnv) {
    UnsetEnv_("PARA_ARCH");
  }
  std::string archScript = NativeFilenames::combinePathSegments(turboRootEnv, "scripts", "sysname");
  if (bfs::exists(archScript)) {
    bp::ipstream out;
    bp::system(archScript, bp::std_out > out);
    std::string s((std::istreambuf_iterator<char>(out)), std::istreambuf_iterator<char>());
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    std::string architecture = s;
    std::string smpArchitecture = architecture + "_smp";
    turbomoleBinaryDir_ = NativeFilenames::combinePathSegments(turboRootEnv, "bin", architecture);
    turbomoleSmpBinaryDir_ = NativeFilenames::combinePathSegments(turboRootEnv, "bin", smpArchitecture);
    turbomoleScriptsDir_ = NativeFilenames::combinePathSegments(turboRootEnv, "scripts");
    std::string newPath = turbomoleScriptsDir_ + ":" + std::getenv("PATH");
    setenv("PATH", newPath.c_str(), 1);
    binaryHasBeenChecked_ = true;
  }
  else
    throw std::runtime_error("TURBODIR was set incorrectly!");

  // Set parallel variables if required
  auto numProcs = settings_->getInt(Utils::SettingsNames::externalProgramNProcs);
  if (numProcs > 1) {
    // conversion to const char* required to set the corresponding env variable
    std::stringstream temp_str;
    temp_str << numProcs;
    const std::string numProcsChar = temp_str.str();
    SetEnv_("PARA_ARCH", "SMP");
    SetEnv_("PARNODES", numProcsChar);
  }
}

void TurbomoleCalculator::applySettings() {
  if (settings_->valid()) {
    if (settings_->getDouble(Utils::SettingsNames::electronicTemperature) > 0.0) {
      throw Core::InitializationException(
          "Turbomole calculations with an electronic temperature above 0.0 K are not supported.");
    }
    if (requiredProperties_.containsSubSet(Property::ExcitedStates)) {
      if (settings_->getInt(SettingsNames::numExcitedStates) == 0) {
        throw std::logic_error("Excited states requested but number of excited states to calculate is zero.");
      }
      if (requiredProperties_.containsSubSet(Property::BondOrderMatrix) ||
          requiredProperties_.containsSubSet(Property::AtomicCharges) || requiredProperties_.containsSubSet(Property::Hessian) ||
          requiredProperties_.containsSubSet(Property::Thermochemistry) ||
          requiredProperties_.containsSubSet(Property::PointChargesGradients)) {
        throw std::runtime_error(
            "Currently, Hessians, point charges gradients, thermochemical properties, atomic charges, and bond orders "
            "cannot be calculated for excited states with the Turbomole calculator.");
      }
    }
    baseWorkingDirectory_ = settings_->getString(SettingsNames::baseWorkingDirectory);
    // throws error for wrong input and updates 'any' entries */
    // information if solvation is performed is deduced from settings from InputFileCreator and therefore discarded here
    Solvation::ImplicitSolvation::solvationNeededAndPossible(availableSolvationModels_, *settings_);
    if (!(settings_->getBool(SettingsNames::enforceScfCriterion)) &&
        (requiredProperties_.containsSubSet(Property::Gradients) || requiredProperties_.containsSubSet(Property::Hessian)) &&
        settings_->getDouble(Utils::SettingsNames::selfConsistenceCriterion) > 1e-8) {
      settings_->modifyDouble(Utils::SettingsNames::selfConsistenceCriterion, 1e-8);
      this->getLog().warning << "Warning: Energy accuracy was increased to 1e-8 to ensure valid gradients/hessian "
                                "as recommended by TURBOMOLE developers."
                             << Core::Log::nl;
    }
  }
  else {
    settings_->throwIncorrectSettings();
  }
}

void TurbomoleCalculator::setStructure(const Utils::AtomCollection& structure) {
  applySettings();
  atoms_ = structure;
  calculationDirectory_ = NativeFilenames::createRandomDirectoryName(baseWorkingDirectory_);
  results_ = Results{};
}

std::unique_ptr<Utils::AtomCollection> TurbomoleCalculator::getStructure() const {
  return std::make_unique<Utils::AtomCollection>(atoms_);
}

void TurbomoleCalculator::modifyPositions(Utils::PositionCollection newPositions) {
  atoms_.setPositions(std::move(newPositions));
  results_ = Results{};
}

const Utils::PositionCollection& TurbomoleCalculator::getPositions() const {
  return atoms_.getPositions();
}

void TurbomoleCalculator::setRequiredProperties(const PropertyList& requiredProperties) {
  requiredProperties_ = requiredProperties;
}

PropertyList TurbomoleCalculator::getRequiredProperties() const {
  return requiredProperties_;
}

PropertyList TurbomoleCalculator::possibleProperties() const {
  return Property::Energy | Property::Gradients | Property::Hessian | Property::BondOrderMatrix |
         Property::PointChargesGradients | Property::Thermochemistry | Property::AtomicCharges |
         Property::ExcitedStates | Property::SuccessfulCalculation;
}

const Results& TurbomoleCalculator::calculate(std::string description) {
  applySettings();

  try {
    return calculateImpl(description);
  }
  catch (std::runtime_error& e) {
    throw Core::UnsuccessfulCalculationException(e.what()); // boost::current_exception_diagnostic_information());
  }
}

const Results& TurbomoleCalculator::calculateImpl(std::string description) {
  initializeProgram();
  ExternalProgram externalProgram;
  externalProgram.setWorkingDirectory(calculationDirectory_);
  externalProgram.createWorkingDirectory();

  TurbomoleHelper helper(calculationDirectory_, turbomoleExecutableBase_);

  setCorrectTurbomoleFileNames(files_, calculationDirectory_);

  // empty artefacts from the previous SCF (important for parsing)
  helper.emptyFile(files_.gradientFile);
  helper.emptyFile(files_.energyFile);

  int numProcs = settings_->getInt(Utils::SettingsNames::externalProgramNProcs);

  if (numProcs > 1)
    turbomoleExecutableBase_ = turbomoleSmpBinaryDir_;
  else
    turbomoleExecutableBase_ = turbomoleBinaryDir_;

  auto inputCreator = TurbomoleInputFileCreator(calculationDirectory_, turbomoleExecutableBase_, files_);
  inputCreator.createInputFiles(atoms_, *settings_);

  // Check whether the given binary is valid
  if (!binaryIsValid()) {
    throw std::runtime_error("No or incorrect information about the binary path for Turbomole was given.");
  }

  std::string executableName = "ridft";
  if (!settings_->getBool(SettingsNames::enableRi)) {
    executableName = "dscf";
    files_.outputFile = files_.dscfFile;
  }

  helper.execute(executableName, true);

  TurbomoleMainOutputParser parser(files_);
  parser.checkForErrors(getLog());

  // steer the orbitals after an SCF calculation if required
  if (settings_->getBool(SettingsNames::steerOrbitals)) {
    auto spinMode = SpinModeInterpreter::getSpinModeFromString(settings_->getString(Utils::SettingsNames::spinMode));
    auto nElec = settings_->getInt(Utils::SettingsNames::spinMultiplicity) - 1;
    // check if calculation really is unrestricted
    if ((spinMode == SpinMode::Unrestricted) || (nElec % 2 != 0)) {
      FilesystemHelpers::copyFile(files_.outputFile, files_.outputBakFile);
      TurbomoleOrbitalSteerer steerer(calculationDirectory_);
      steerer.steerOrbitals();
      helper.execute(executableName, true);
    }
    else
      throw std::runtime_error("Orbital steering can only be performed for unrestricted calculations. ");
  }

  if (requiredProperties_.containsSubSet(Property::ExcitedStates)) {
    helper.execute("escf", true);
  }

  if (requiredProperties_.containsSubSet(Property::Gradients) ||
      requiredProperties_.containsSubSet(Property::PointChargesGradients)) {
    if (requiredProperties_.containsSubSet(Property::ExcitedStates)) {
      helper.execute("egrad", true);
    }
    else {
      if (settings_->getBool(SettingsNames::enableRi)) {
        helper.execute("rdgrad", true);
      }
      else {
        helper.execute("grad", true);
      }
    }
  }
  if (requiredProperties_.containsSubSet(Property::Hessian) || requiredProperties_.containsSubSet(Property::Thermochemistry)) {
    if (settings_->getString(SettingsNames::hessianCalculationType) == "analytical") {
      helper.execute("aoforce", true);
    }
    else {
      // -central as suggested by Turbomole manual
      std::string numforceCommand = "../../scripts/NumForce -central ";
      if (settings_->getBool(SettingsNames::enforceNumforce)) {
        numforceCommand += "-c ";
      }
      if (settings_->getBool(SettingsNames::enableRi)) {
        // one gradient call before NumForce
        helper.execute("rdgrad", true);
        // -ri for Ri
        helper.execute(numforceCommand + "-ri ", true, "numforce.out");
      }
      else {
        // one gradient call before NumForce
        helper.execute("grad", true);
        helper.execute(numforceCommand, true, "numforce.out");
      }
      // Check, if Turbomole's gradient check stopped numforce
      std::ifstream out;
      out.open(NativeFilenames::combinePathSegments(calculationDirectory_, "numforce.out"));
      bool failedNumforceGradientCheck = helper.jobWasSuccessful(out, "no 2nd derivatives");
      if (failedNumforceGradientCheck) {
        throw std::runtime_error("Turbomole's gradient norm check failed -- "
                                 "gradient norm on given structure too large. ");
      }
    }
  }

  // Turbomole writes some large .tmp files (even for successful calculation) that are not deleted automatically
  deleteTemporaryFiles();

  results_.set<Property::Description>(std::move(description));
  if (requiredProperties_.containsSubSet(Property::Energy)) {
    if (requiredProperties_.containsSubSet(Property::ExcitedStates)) {
      int highestExcitedState = settings_->getInt(SettingsNames::numExcitedStates);
      results_.set<Property::Energy>(parser.getExcitedStateEnergy(highestExcitedState));
    }
    else {
      results_.set<Property::Energy>(parser.getEnergy());
    }
  }

  if (requiredProperties_.containsSubSet(Property::Gradients)) {
    results_.set<Property::Gradients>(parser.getGradients());
  }
  if (requiredProperties_.containsSubSet(Property::Hessian)) {
    results_.set<Property::Hessian>(parser.getHessian());
  }
  if (requiredProperties_.containsSubSet(Property::BondOrderMatrix)) {
    results_.set<Property::BondOrderMatrix>(parser.getBondOrders());
  }
  if (requiredProperties_.containsSubSet(Property::AtomicCharges)) {
    results_.set<Property::AtomicCharges>(parser.getLoewdinCharges());
  }
  if (requiredProperties_.containsSubSet(Property::PointChargesGradients)) {
    results_.set<Property::PointChargesGradients>(parser.getPointChargesGradients());
  }
  if (requiredProperties_.containsSubSet(Property::Thermochemistry)) {
    auto thermoCalc = ThermochemistryCalculator(results_.get<Property::Hessian>(), atoms_,
                                                settings_->getInt(Utils::SettingsNames::spinMultiplicity),
                                                results_.get<Property::Energy>());
    thermoCalc.setMolecularSymmetryNumber(parser.getSymmetryNumber());
    thermoCalc.setTemperature(settings_->getDouble(Utils::SettingsNames::temperature));
    thermoCalc.setPressure(settings_->getDouble(Utils::SettingsNames::pressure));
    auto thermochemistry = thermoCalc.calculate();
    results_.set<Property::Thermochemistry>(thermochemistry);
  }
  results_.set<Property::SuccessfulCalculation>(true);
  results_.set<Property::ProgramName>("turbomole");

  return results_;
}

const Utils::Settings& TurbomoleCalculator::settings() const {
  return *settings_;
}

Utils::Settings& TurbomoleCalculator::settings() {
  return *settings_;
}

void TurbomoleCalculator::copyBackupFiles(const std::string& from, const std::string& to) const {
  std::string mosFromFile = NativeFilenames::combinePathSegments(from, "mos");
  std::string alphaFromFile = NativeFilenames::combinePathSegments(from, "alpha");
  std::string betaFromFile = NativeFilenames::combinePathSegments(from, "beta");

  std::string mosToFile = NativeFilenames::combinePathSegments(to, "mos");
  std::string alphaToFile = NativeFilenames::combinePathSegments(to, "alpha");
  std::string betaToFile = NativeFilenames::combinePathSegments(to, "beta");

  try {
    if (bfs::exists(mosFromFile)) {
      FilesystemHelpers::copyFile(mosFromFile, mosToFile);
    }
    else if (bfs::exists(alphaFromFile) && bfs::exists(betaFromFile)) {
      FilesystemHelpers::copyFile(alphaFromFile, alphaToFile);
      FilesystemHelpers::copyFile(betaFromFile, betaToFile);
    }
  }
  catch (std::runtime_error& e) {
    throw TurbomoleStateSavingException();
  }
}

std::shared_ptr<Core::State> TurbomoleCalculator::getState() const {
  auto ret = std::make_shared<TurbomoleState>(this->getCalculationDirectory());
  this->copyBackupFiles(this->getCalculationDirectory(), ret->stateIdentifier);
  return ret;
}

void TurbomoleCalculator::loadState(std::shared_ptr<Core::State> state) {
  auto turbomoleState = std::dynamic_pointer_cast<TurbomoleState>(state);
  this->copyBackupFiles(turbomoleState->stateIdentifier, this->getCalculationDirectory());
}

const Utils::Results& TurbomoleCalculator::results() const {
  return results_;
}

Utils::Results& TurbomoleCalculator::results() {
  return results_;
}

std::string TurbomoleCalculator::getCalculationDirectory() const {
  return calculationDirectory_;
}

std::string TurbomoleCalculator::getTurbomoleExecutableBase() const {
  return turbomoleExecutableBase_;
}

bool TurbomoleCalculator::binaryIsValid() {
  if (binaryHasBeenChecked_) {
    return true;
  }
  return false;
}

void TurbomoleCalculator::deleteTemporaryFiles() {
  bfs::path p(calculationDirectory_);
  if (bfs::exists(p) && bfs::is_directory(p)) {
    bfs::directory_iterator end;
    for (bfs::directory_iterator it(p); it != end; ++it) {
      try {
        if (bfs::is_regular_file(it->status()) && (it->path().extension() == ".tmp")) {
          bfs::remove(it->path());
        }
      }
      catch (const std::exception& e) {
      }
    }
  }
}

// commented out until new interface is there
void TurbomoleCalculator::setOrbitals(const MolecularOrbitals& mos) {
  mos_ = mos;
}

} // namespace ExternalQC
} // namespace Utils
} // namespace Scine
