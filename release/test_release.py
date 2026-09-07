import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from aggregate import aggregate, TRANSFERS
from download import download, verify


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.grid = dict(datasets=["a", "b"], models=["m"], seeds=[1, 2])
        self.rows = [
            dict(dataset=dataset, model="m", transfer=transfer, seed=seed,
                 score=20 if dataset == "a" else 80, status="measured", source="fixture")
            for dataset in ["a", "b"]
            for transfer in TRANSFERS
            for seed in [1, 2]
        ]

    def test_complete_balanced_mean(self):
        result = aggregate(self.rows, **self.grid)
        self.assertEqual(result["summaries"][0]["overall"]["mean"], 50)
        self.assertEqual(result["summaries"][0]["overall"]["seed_std"], 0)
        self.assertTrue(result["complete"])

    def test_overall_weights_three_settings_equally(self):
        setting_scores = {
            "RGB->RGB": 10,
            "S2->S2": 10,
            "RGB->SAR": 30,
            "S2->SAR": 30,
            "RGB->N'S1S2": 30,
            "RGB->RGBN": 80,
            "S2->S2+SAR": 80,
        }
        rows = [
            dict(dataset=dataset, model="m", transfer=transfer, seed=seed,
                 score=score, status="measured", source="fixture")
            for dataset in self.grid["datasets"]
            for transfer, score in setting_scores.items()
            for seed in self.grid["seeds"]
        ]
        summary = aggregate(rows, **self.grid)["summaries"][0]
        self.assertEqual(summary["settings"]["in_distribution"]["mean"], 10)
        self.assertEqual(summary["settings"]["no_overlap"]["mean"], 30)
        self.assertEqual(summary["settings"]["superset"]["mean"], 80)
        self.assertEqual(summary["overall"]["mean"], 40)

    def test_missing_seed_blocks_rankings(self):
        result = aggregate(self.rows[:-1], **self.grid)
        self.assertFalse(result["complete"])
        self.assertEqual(result["summaries"], [])
        self.assertEqual(result["missing"][0]["missing_seeds"], [2])

    def test_imputation_duplicate_and_nonfinite_rejected(self):
        for modification in ({"status": "imputed"}, {"score": float("nan")}, {"source": ""}):
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                aggregate([dict(self.rows[0], **modification)] + self.rows[1:], **self.grid)
        with self.assertRaises(ValueError):
            aggregate(self.rows + [self.rows[0]], **self.grid)

    def test_backbone_manifest_and_chivit_license(self):
        root = Path(__file__).parent
        backbones = json.loads((root / "backbones.json").read_text())
        models = backbones["models"]
        self.assertEqual(len(models), 14)
        self.assertEqual(len({item["model"] for item in models}), 14)
        self.assertTrue(all(item.get("loader") for item in models))
        chivit = json.loads((root / "chivit.json").read_text())
        self.assertEqual(chivit["license"], "Apache-2.0")
        self.assertTrue((root / "CHIVIT_LICENSE").is_file())

    def test_corrupt_asset_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            path.write_bytes(b"valid")
            checksum = {"type": "SHA-256", "value": hashlib.sha256(b"valid").hexdigest()}
            verify(path, 5, checksum)
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                verify(path, 5, checksum)
            with self.assertRaises(ValueError):
                download({"name": "../escape", "bytes": 5, "restricted": False,
                          "checksum": checksum}, Path(directory))


if __name__ == "__main__":
    unittest.main()
