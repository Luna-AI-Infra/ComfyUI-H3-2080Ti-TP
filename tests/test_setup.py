"""CPU-only regressions for installation failures; no model downloads or CUDA."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupTests(unittest.TestCase):
    def test_prebuilt_rejects_tampered_archive(self):
        installer = script('install_prebuilt')
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'bad.tar.gz'
            archive.write_bytes(b'not the pinned release')
            with patch.object(installer.platform, 'system', return_value='Linux'):
                with patch.object(installer.platform, 'machine', return_value='x86_64'):
                    with patch.object(sys, 'argv', ['installer', '--archive', str(archive)]):
                        with self.assertRaisesRegex(RuntimeError, 'Archive SHA256 mismatch'):
                            installer.main()

    def test_default_launch_uses_speed_preset(self):
        launcher = script('launch_comfyui')
        config = json.loads((ROOT / 'configs/performance.json').read_text())
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace())))
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'main.py').write_text('# fixture', encoding='utf-8')
            before = Path.cwd()
            def check(*args, **kwargs):
                self.assertEqual(sys.argv, ['main.py'] + config['comfyui_args'] + ['--port', '8189'])
                for key, value in config['environment'].items():
                    self.assertEqual(os.environ[key], value)
                self.assertNotIn('ram_weights', config)
            try:
                with patch.dict(sys.modules, {'torch': torch}), patch.dict(os.environ):
                    with patch.object(sys, 'path', list(sys.path)):
                        with patch.object(sys, 'argv', ['launcher', '--comfyui', folder, '--', '--port', '8189']):
                            with patch.object(launcher.runpy, 'run_path', side_effect=check):
                                launcher.main()
            finally:
                os.chdir(before)

    def test_missing_task_weights_fail(self):
        verifier = script('verify_models')
        for task in ('fl2va', 'ref2va'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as folder:
                with patch.object(sys, 'argv', ['verify_models', '--comfyui', folder, '--task', task]):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        with self.assertRaises(SystemExit) as error:
                            verifier.main()
                    self.assertEqual(error.exception.code, 1)
                    self.assertEqual(output.getvalue().count('MISSING'), 5)
                    other = 'ref2va' if task == 'fl2va' else 'fl2va'
                    self.assertNotIn(other, output.getvalue())

    def test_ram_install_sees_only_comfyui_arguments(self):
        launcher = script('launch_comfyui')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'main.py').write_text('# fixture', encoding='utf-8')
            config = root / 'config.json'
            config.write_text(json.dumps({'ram_weights': {'paths': []},
                                          'comfyui_args': ['--preview-method', 'none']}))
            expected = ['main.py', '--preview-method', 'none', '--port', '8189']
            def check_argv(*args, **kwargs):
                self.assertEqual(sys.argv, expected)
            options = types.ModuleType('comfy.options')
            options.enable_args_parsing = check_argv
            comfy = types.ModuleType('comfy')
            comfy.options = options
            ram = types.ModuleType('ram_weights')
            ram.preload = check_argv
            ram.install = check_argv
            resource = types.ModuleType('resource')
            resource.RLIMIT_MEMLOCK = 0
            resource.RLIM_INFINITY = -1
            resource.getrlimit = lambda _: (-1, -1)
            torch = types.SimpleNamespace(backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace())))
            modules = {'torch': torch, 'comfy': comfy, 'comfy.options': options,
                       'ram_weights': ram, 'resource': resource}
            before = Path.cwd()
            try:
                with patch.dict(sys.modules, modules), patch.object(sys, 'path', list(sys.path)):
                    with patch.object(sys, 'argv', ['launcher', '--comfyui', folder,
                                                   '--config', str(config), '--', '--port', '8189']):
                        with patch.object(launcher.runpy, 'run_path', side_effect=check_argv):
                            launcher.main()
            finally:
                os.chdir(before)


if __name__ == '__main__':
    unittest.main()
