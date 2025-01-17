/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#ifndef UTILS_ORCACALCULATORSETTINGS_H
#define UTILS_ORCACALCULATORSETTINGS_H

#include <Utils/ExternalQC/SettingsNames.h>
#include <Utils/IO/FilesystemHelpers.h>
#include <Utils/Scf/LcaoUtils/SpinMode.h>
#include <Utils/Settings.h>
#include <Utils/UniversalSettings/SettingsNames.h>

namespace Scine {
namespace Utils {
namespace ExternalQC {

/**
 * @class OrcaCalculatorSettings OrcaCalculatorSettings.h
 * @brief Settings for ORCA calculations.
 */
class OrcaCalculatorSettings : public Scine::Utils::Settings {
 public:
  // These functions populate certain settings
  void addMolecularCharge(UniversalSettings::DescriptorCollection& settings);
  void addSpinMultiplicity(UniversalSettings::DescriptorCollection& settings);
  void addSelfConsistenceCriterion(UniversalSettings::DescriptorCollection& settings);
  void addMaxScfIterations(UniversalSettings::DescriptorCollection& settings);
  void addMethod(UniversalSettings::DescriptorCollection& settings);
  void addBasisSet(UniversalSettings::DescriptorCollection& settings);
  void addSpinMode(UniversalSettings::DescriptorCollection& settings);
  void addNumProcs(UniversalSettings::DescriptorCollection& settings);
  void addOrcaFilenameBase(UniversalSettings::DescriptorCollection& settings);
  void addBaseWorkingDirectory(UniversalSettings::DescriptorCollection& settings);
  void addMemoryForOrca(UniversalSettings::DescriptorCollection& settings);
  void addDeleteTemporaryFilesOption(UniversalSettings::DescriptorCollection& settings);
  void addPointChargesFile(UniversalSettings::DescriptorCollection& settings);
  void addTemperature(UniversalSettings::DescriptorCollection& settings);
  void addSolvent(UniversalSettings::DescriptorCollection& settings);
  void addSolvation(UniversalSettings::DescriptorCollection& settings);
  void addElectronicTemperature(UniversalSettings::DescriptorCollection& settings);
  void addScfDamping(UniversalSettings::DescriptorCollection& settings);
  void addHessianCalculationType(UniversalSettings::DescriptorCollection& settings);
  void addSpecialOption(UniversalSettings::DescriptorCollection& settings);

  /**
   * @brief Constructor that populates the OrcaCalculatorSettings.
   */
  OrcaCalculatorSettings() : Settings("OrcaCalculatorSettings") {
    addMolecularCharge(_fields);
    addSpinMultiplicity(_fields);
    addSelfConsistenceCriterion(_fields);
    addMaxScfIterations(_fields);
    addMethod(_fields);
    addBasisSet(_fields);
    addSpinMode(_fields);
    addNumProcs(_fields);
    addOrcaFilenameBase(_fields);
    addBaseWorkingDirectory(_fields);
    addMemoryForOrca(_fields);
    addDeleteTemporaryFilesOption(_fields);
    addPointChargesFile(_fields);
    addTemperature(_fields);
    addSolvent(_fields);
    addSolvation(_fields);
    addScfDamping(_fields);
    addHessianCalculationType(_fields);
    addElectronicTemperature(_fields);
    addSpecialOption(_fields);
    resetToDefaults();
  };
};

inline void OrcaCalculatorSettings::addMolecularCharge(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::IntDescriptor molecularCharge("Sets the molecular charge to use in the calculation.");
  molecularCharge.setMinimum(-10);
  molecularCharge.setMaximum(10);
  molecularCharge.setDefaultValue(0);
  settings.push_back(Scine::Utils::SettingsNames::molecularCharge, std::move(molecularCharge));
}

inline void OrcaCalculatorSettings::addSpinMultiplicity(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::IntDescriptor spinMultiplicity("Sets the desired spin multiplicity to use in the "
                                                           "calculation.");
  spinMultiplicity.setMinimum(1);
  spinMultiplicity.setMaximum(10);
  spinMultiplicity.setDefaultValue(1);
  settings.push_back(Scine::Utils::SettingsNames::spinMultiplicity, std::move(spinMultiplicity));
}

inline void OrcaCalculatorSettings::addSelfConsistenceCriterion(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::DoubleDescriptor selfConsistenceCriterion("Sets the desired convergence criterion.");
  selfConsistenceCriterion.setMinimum(0);
  selfConsistenceCriterion.setDefaultValue(1e-7);
  settings.push_back(Scine::Utils::SettingsNames::selfConsistenceCriterion, std::move(selfConsistenceCriterion));
}

inline void OrcaCalculatorSettings::addMethod(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor method("The method used in the ORCA calculation.");
  method.setDefaultValue("PBE");
  settings.push_back(Utils::SettingsNames::method, std::move(method));
}

inline void OrcaCalculatorSettings::addBasisSet(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor basisSet("The basis set used in the ORCA calculation.");
  basisSet.setDefaultValue("def2-SVP");
  settings.push_back(Utils::SettingsNames::basisSet, std::move(basisSet));
}

inline void OrcaCalculatorSettings::addSpinMode(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::OptionListDescriptor spinMode("The spin mode such as 'restricted' or 'unrestricted'.");
  spinMode.addOption(SpinModeInterpreter::getStringFromSpinMode(SpinMode::Any));
  spinMode.addOption(SpinModeInterpreter::getStringFromSpinMode(SpinMode::Restricted));
  spinMode.addOption(SpinModeInterpreter::getStringFromSpinMode(SpinMode::RestrictedOpenShell));
  spinMode.addOption(SpinModeInterpreter::getStringFromSpinMode(SpinMode::Unrestricted));
  spinMode.setDefaultOption(SpinModeInterpreter::getStringFromSpinMode(SpinMode::Any));
  settings.push_back(Utils::SettingsNames::spinMode, std::move(spinMode));
}

inline void OrcaCalculatorSettings::addNumProcs(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::IntDescriptor orcaNumProcs("Number of processes for the ORCA calculation.");
  orcaNumProcs.setDefaultValue(1);
  orcaNumProcs.setMinimum(1);
  settings.push_back(Utils::SettingsNames::externalProgramNProcs, std::move(orcaNumProcs));
}

inline void OrcaCalculatorSettings::addOrcaFilenameBase(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor orcaFilenameBase("Base of the file name of the ORCA calculations.");
  orcaFilenameBase.setDefaultValue("orca_calc");
  settings.push_back(SettingsNames::orcaFilenameBase, std::move(orcaFilenameBase));
}

inline void OrcaCalculatorSettings::addBaseWorkingDirectory(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor baseWorkingDirectory("Base directory for the ORCA calculations.");
  baseWorkingDirectory.setDefaultValue(FilesystemHelpers::currentDirectory());
  settings.push_back(SettingsNames::baseWorkingDirectory, std::move(baseWorkingDirectory));
}

inline void OrcaCalculatorSettings::addMaxScfIterations(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::IntDescriptor maxScfIterations("Maximum number of SCF iterations.");
  maxScfIterations.setMinimum(1);
  maxScfIterations.setDefaultValue(100);
  settings.push_back(Scine::Utils::SettingsNames::maxScfIterations, std::move(maxScfIterations));
}

inline void OrcaCalculatorSettings::addMemoryForOrca(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::IntDescriptor orcaMemory("Memory that can be used by the ORCA calculation.");
  orcaMemory.setDefaultValue(1024);
  settings.push_back(Utils::SettingsNames::externalProgramMemory, std::move(orcaMemory));
}

inline void OrcaCalculatorSettings::addDeleteTemporaryFilesOption(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::BoolDescriptor deleteTemporaryFiles(
      "Delete all files with the .tmp extension after an ORCA calculation has failed.");
  deleteTemporaryFiles.setDefaultValue(true);
  settings.push_back(SettingsNames::deleteTemporaryFiles, std::move(deleteTemporaryFiles));
}

inline void OrcaCalculatorSettings::addPointChargesFile(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor pointChargesFile("Sets the file name for an ORCA point charges file.");
  pointChargesFile.setDefaultValue("");
  settings.push_back(SettingsNames::pointChargesFile, std::move(pointChargesFile));
}

inline void OrcaCalculatorSettings::addTemperature(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::DoubleDescriptor temperature("Sets the temperature for the thermochemical calculation.");
  temperature.setDefaultValue(298.15);
  settings.push_back(Utils::SettingsNames::temperature, std::move(temperature));
}

inline void OrcaCalculatorSettings::addSolvent(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor solventOption(
      "Sets the implicit solvent using the CPCM model to be applied in the ORCA calculation.");
  solventOption.setDefaultValue("");
  settings.push_back(Utils::SettingsNames::solvent, std::move(solventOption));
}

inline void OrcaCalculatorSettings::addSolvation(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor solvationOption(
      "Sets the implicit solvation model in the ORCA calculation. Currently, only CPCM is available.");
  solvationOption.setDefaultValue("");
  settings.push_back(Utils::SettingsNames::solvation, std::move(solvationOption));
}

inline void OrcaCalculatorSettings::addScfDamping(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::BoolDescriptor scfDamping("Switch SCF damping on/off.");
  scfDamping.setDefaultValue(false);
  settings.push_back(Utils::SettingsNames::scfDamping, std::move(scfDamping));
}

inline void OrcaCalculatorSettings::addHessianCalculationType(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::OptionListDescriptor hessianType("The method for calculating the Hessian.");
  hessianType.addOption("analytical");
  hessianType.addOption("numerical");
  hessianType.setDefaultOption("analytical");
  settings.push_back(ExternalQC::SettingsNames::hessianCalculationType, std::move(hessianType));
}

inline void OrcaCalculatorSettings::addElectronicTemperature(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::DoubleDescriptor electronicTemperature(
      "Sets the electronic temperature for SCF calculations.");
  electronicTemperature.setMinimum(0.0);
  electronicTemperature.setDefaultValue(0.0);
  settings.push_back(Utils::SettingsNames::electronicTemperature, std::move(electronicTemperature));
}

inline void OrcaCalculatorSettings::addSpecialOption(UniversalSettings::DescriptorCollection& settings) {
  Utils::UniversalSettings::StringDescriptor specialOption(
      "Allows to add a custom string to the ORCA input line; recommended for experts only.");
  specialOption.setDefaultValue("");
  settings.push_back(ExternalQC::SettingsNames::specialOption, std::move(specialOption));
}

} // namespace ExternalQC
} // namespace Utils
} // namespace Scine

#endif // UTILS_ORCACALCULATORSETTINGS_H
