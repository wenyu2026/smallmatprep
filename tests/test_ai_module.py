import json
import unittest
from unittest.mock import patch

import pandas as pd

from smallmatprep.ai import config_generator, diagnose_reporter


class TestAIEnhancements(unittest.TestCase):
    def test_generate_config_returns_dict_when_chat_returns_dict(self):
        expected = {
            "target": "conductivity",
            "impute": {"method": "knn", "columns": ["temperature"]},
        }

        dummy_client = unittest.mock.Mock()
        dummy_client.chat.return_value = expected

        with patch.object(config_generator, "LLMClient", return_value=dummy_client):
            result = config_generator.generate_config(
                description="Predict conductivity from electrolyte data.",
                api_key="test-key",
            )

        self.assertEqual(result, expected)

    def test_generate_config_parses_json_string(self):
        response_text = json.dumps(
            {
                "target": "conductivity",
                "categorical_columns": ["solvent"],
            }
        )

        dummy_client = unittest.mock.Mock()
        dummy_client.chat.return_value = response_text

        with patch.object(config_generator, "LLMClient", return_value=dummy_client):
            result = config_generator.generate_config(
                description="Predict conductivity from electrolyte data.",
                api_key="test-key",
            )

        self.assertEqual(result, {"target": "conductivity", "categorical_columns": ["solvent"]})

    def test_diagnose_returns_string_from_llm(self):
        df = pd.DataFrame(
            {
                "temperature": [20.0, 25.0, None],
                "conductivity": [1.2, 1.4, 1.3],
            }
        )

        dummy_client = unittest.mock.Mock()
        dummy_client.chat.return_value = "Good dataset."

        with patch.object(diagnose_reporter, "LLMClient", return_value=dummy_client):
            report = diagnose_reporter.diagnose(
                df,
                target="conductivity",
                api_key="test-key",
            )

        self.assertIsInstance(report, str)
        self.assertEqual(report, "Good dataset.")

    def test_prompt_templates_exist(self):
        self.assertTrue(config_generator._load_prompt("config_generator"))
        self.assertTrue(diagnose_reporter._load_prompt("diagnose_reporter"))

    def test_generate_config_with_columns_injects_column_names(self):
        """Ensure column names are included in the user prompt."""
        expected = {
            "target": "target",
            "id_column": "sample_id",
        }

        dummy_client = unittest.mock.Mock()
        dummy_client.chat.return_value = expected

        with patch.object(config_generator, "LLMClient", return_value=dummy_client) as mock_cls:
            result = config_generator.generate_config_with_columns(
                description="Predict target.",
                api_key="test-key",
                columns=["sample_id", "target", "temp", "solvent_A"],
            )
            # Check that the prompt included the column names
            call_args = mock_cls.return_value.chat.call_args
            user_prompt = call_args[0][1]  # second positional arg is user_prompt
            self.assertIn("temp", user_prompt)
            self.assertIn("solvent_A", user_prompt)

        self.assertEqual(result, expected)

    def test_generate_config_with_columns_from_dataframe(self):
        """Ensure passing a DataFrame works the same as passing a list."""
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})

        dummy_client = unittest.mock.Mock()
        dummy_client.chat.return_value = {"target": "col_b"}

        with patch.object(config_generator, "LLMClient", return_value=dummy_client):
            result = config_generator.generate_config_with_columns(
                description="Predict col_b",
                api_key="test-key",
                columns=df,
            )
        self.assertEqual(result, {"target": "col_b"})

    def test_save_config_creates_file(self):
        """save_config should write a valid JSON file and return the path."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "my_config.json"
            config = {"target": "y", "model": "Ridge"}
            saved = config_generator.save_config(config, str(filepath))
            self.assertEqual(saved, filepath)
            loaded = json.loads(filepath.read_text(encoding="utf-8"))
            self.assertEqual(loaded, config)


if __name__ == "__main__":
    unittest.main()
