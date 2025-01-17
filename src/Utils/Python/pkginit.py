__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import importlib.util
import os
import sys
from pathlib import Path
from distutils import ccompiler, sysconfig


# Step 1: Load the python bindings
# Look for the binary file
expected_suffix = sysconfig.get_config_var("EXT_SUFFIX")
expected_name = __name__ + expected_suffix
this_file_dir = Path(__file__).parent.absolute()
python_module_path = os.path.join(this_file_dir, expected_name)
if not os.path.exists(python_module_path):
    raise ImportError("Could not find {}".format(expected_name))

# This is more or less copy-paste from importlib's docs, makes the pybind11
# module .so wild-importable (e.g. from `scine_utilities.IO import *`)
spec = importlib.util.spec_from_file_location(__name__, python_module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)

# Step 2: Load the accompanying C++ module
manager = module.core.ModuleManager()
# scine_utilities provides both Gaussian and Orca, maybe more later. Testing
# for one should be enough to ensure the the module is present.
if not manager.module_loaded('Gaussian'):
    shlib_suffix = ccompiler.new_compiler().shared_lib_extension
    module_filename = "utilsos.module" + shlib_suffix
    # Look within the python module directory (module is here in the case of
    # python packages) and the lib folder the site packages are in
    current_path = os.path.dirname(os.path.realpath(__file__))
    lib_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
    test_paths = [current_path, lib_path]

    def exists_and_could_load(path):
        full_path = os.path.join(path, module_filename)
        if os.path.exists(full_path):
            try:
                manager.load(full_path)
            except RuntimeError as err:
                print("Could not load {}: {}".format(full_path, err))
                return False

            return True

        return False

    if not any(map(exists_and_could_load, test_paths)):
        raise ImportError('{} could not be located.'.format(module_filename))
