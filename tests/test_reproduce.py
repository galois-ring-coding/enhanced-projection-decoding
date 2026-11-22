import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import reproduce


class ReproductionTests(unittest.TestCase):
    def test_current_theorem_certificates_match_archived_calculations(self):
        report = reproduce.model_examples()
        recorded = json.loads((reproduce.ROOT / 'reference' / 'lee' /
                               'summary.json').read_text(encoding='utf-8'))['certificates']
        lee = report['lee']
        self.assertAlmostEqual(lee['ep_upper_exponent'], recorded['analytic_super_upper'])
        self.assertAlmostEqual(lee['projection_lower_exponent'], recorded['analytic_dcc_lower'])
        self.assertLess(lee['ep_upper_exponent'], lee['projection_lower_exponent'])
        self.assertLess(lee['distance_entropy'], lee['distance_threshold'])
        self.assertTrue(all(lee['integer_certificates'].values()))
        self.assertAlmostEqual(report['rank']['epsilon'], .34)
        self.assertAlmostEqual(report['rank']['ep_exponent_per_log2p'], .01)
        self.assertAlmostEqual(report['rank']['projection_exponent_per_log2p'], .03)

    def test_manifest_detects_modified_missing_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('reproduce.py', 'README.md', 'requirements.txt', 'requirements-plots.txt',
                         '.gitattributes', '.gitignore'):
                (root / name).write_text('original\n', encoding='utf-8')
            (root / 'src').mkdir()
            source = root / 'src' / 'example.py'
            source.write_text('original\n', encoding='utf-8')
            with patch.object(reproduce, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                reproduce.manifest(write=True)
                source.write_text('modified\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                    reproduce.manifest()
                source.write_text('original\n', encoding='utf-8')
                source.unlink()
                with self.assertRaisesRegex(ValueError, 'file list differs'):
                    reproduce.manifest()
                source.write_text('original\n', encoding='utf-8')
                (root / 'src' / 'unlisted.py').write_text('extra\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'file list differs'):
                    reproduce.manifest()

    def test_output_guard_prevents_reference_overwrite(self):
        result = subprocess.run(
            [sys.executable, str(reproduce.ROOT / 'reproduce.py'), 'models',
             '--output', str(reproduce.ROOT / 'reference' / 'lee')],
            capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Outputs must not overwrite reference data', result.stderr)


if __name__ == '__main__':
    unittest.main()
