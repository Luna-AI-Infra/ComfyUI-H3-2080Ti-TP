"""Build the separate experimental Kitchen+Sage extension, not the normal Kitchen path."""
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def main():
    package = '_h3_kitchen_sage_build'
    mod = types.ModuleType(package)
    mod.__path__ = [str(ROOT / 'multistream')]
    sys.modules[package] = mod
    source = ROOT / 'multistream/tp_kitchen_sage.py'
    spec = importlib.util.spec_from_file_location(package + '.tp_kitchen_sage', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.extension(build=True)
    print('Experimental Kitchen+Sage extension built')


if __name__ == '__main__':
    main()
