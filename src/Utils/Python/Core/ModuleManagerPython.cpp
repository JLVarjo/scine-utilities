/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include <Core/Interfaces/Calculator.h>
#include <Core/Interfaces/CalculatorWithReference.h>
#include <Core/ModuleManager.h>
#include <Utils/Pybind.h>

using namespace Scine::Core;

void module_manager_load(ModuleManager& manager, const std::string& filename) {
  manager.load(filename);
}

using GetVariantType = boost::variant<std::shared_ptr<Calculator>, std::shared_ptr<CalculatorWithReference>>;

using GetReturnType = boost::optional<GetVariantType>;

GetReturnType module_manager_get(ModuleManager& manager, const std::string& interface, const std::string& model) {
  if (manager.has(interface, model)) {
    if (interface == Calculator::interface) {
      return GetVariantType{manager.get<Calculator>(model)};
    }
    if (interface == CalculatorWithReference::interface) {
      return GetVariantType{manager.get<CalculatorWithReference>(model)};
    }
  }

  return boost::none;
}

std::vector<std::string> module_manager_models(ModuleManager& manager, const std::string& interface) {
  return manager.getLoadedModels(interface);
}

bool module_manager_has(ModuleManager& manager, const std::string& interface, const std::string& model) {
  return manager.has(interface, model);
}

void init_module_manager(pybind11::module& m) {
  pybind11::class_<ModuleManager, std::unique_ptr<ModuleManager, pybind11::nodelete>> module_manager(m,
                                                                                                     "ModuleManager");

  module_manager.doc() = R"(
    Manager for all dynamically loaded SCINE modules

    SCINE Modules are shared libraries that offer models of the interfaces
    defined in SCINE Core, such as the :class:`core.Calculator`. Generally,
    loading a python module that wraps a particular SCINE project directly
    loads these shared libraries and makes the models it provides available
    through the query interface present here.

    This class is a singleton, accessible via a default ``__init__`` method:

    >>> m = core.ModuleManager()
  )";

  module_manager.def(
      pybind11::init([]() { return std::unique_ptr<ModuleManager, pybind11::nodelete>(&ModuleManager::getInstance()); }));

  module_manager.def("load", &module_manager_load, pybind11::arg("filename"),
                     R"(
      Load a module file

      SCINE Module files have the suffix ``.module.so`` to clearly disambiguate
      them from general-purpose shared libraries.
    )");

  module_manager.def_property_readonly("modules", &ModuleManager::getLoadedModuleNames, "List of loaded module names");

  module_manager.def_property_readonly("interfaces", &ModuleManager::getLoadedInterfaces,
                                       "List of interfaces for which at least one model is loaded");

  module_manager.def("models", &module_manager_models, pybind11::arg("interface"),
                     R"(
      List of available models of an interface

      Collects all classes modeling a particular interface. If no further
      SCINE python modules such as Sparrow are loaded, the list of models for
      the calculator is empty. Some external quantum chemical software is
      auto-detected and made available via the :class:`core.Calculator`
      interface, so your mileage may vary.

      :param interface: String name of the interface to check for
      :returns: List of string names of models of the interface argument. You
        can use these as arguments to the ``get`` function.

      >>> m = core.ModuleManager()
      >>> m.models(core.Calculator.INTERFACE)
      ['TEST', 'LENNARDJONES']
    )");

  module_manager.def("has", &module_manager_has, pybind11::arg("interface"), pybind11::arg("model"),
                     R"(
      Check whether a particular model of an interface is available

      :param interface: String name of the interface to check for
      :param model: String name of the model of the interface to check for
      :returns: Whether the model is available

      >>> m = core.ModuleManager()
      >>> m.has(core.Calculator.INTERFACE, "PM6") # If Sparrow is not loaded
      False
    )");

  module_manager.def("module_loaded", &ModuleManager::moduleLoaded, pybind11::arg("module"),
                     R"(
      Check whether a particular module is loaded

      :param module: Name of the module to check for
      :returns: Whether the module is loaded

      >>> m = core.ModuleManager()
      >>> m.module_loaded("Sparrow")
      False
    )");

  module_manager.def("get", &module_manager_get, pybind11::arg("interface"), pybind11::arg("model"),
                     R"(
      Get an instance of an interface model

      :param interface: String name of the interface to check for
      :param model: String name of the model of the interface to check for
      :returns: The model if available, ``None`` otherwise

      >>> m = core.ModuleManager()
      >>> m.get(core.Calculator.INTERFACE, "PM6") # if Sparrow is not loaded
    )");
}
