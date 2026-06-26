import json
import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.project_config import load_project_config


class ProjectConfigTests(unittest.TestCase):
    def test_load_project_config_reads_project_metadata_and_remote_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "project.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "unitree_rl_mjlab",
                        "local_cache_root": str(Path(tmp) / "cache"),
                        "parser_profile": "rsl_rl_tensorboard",
                        "preferred_metrics": [
                            "Train/mean_reward",
                            "Episode_Termination/fell_over",
                        ],
                        "log_patterns": ["logs/rsl_rl/*/*"],
                        "tag_schema": ["good-flat", "bad-stairs"],
                        "remote_sources": [
                            {
                                "name": "x-server",
                                "host": "example.com",
                                "user": "eai",
                                "port": 12188,
                                "remote_log_root": "/home/eai/workspace/logs/rsl_rl",
                                "method": "rsync",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_project_config(config_path)

        self.assertEqual(config.name, "unitree_rl_mjlab")
        self.assertEqual(config.parser_profile, "rsl_rl_tensorboard")
        self.assertEqual(config.preferred_metrics, ("Train/mean_reward", "Episode_Termination/fell_over"))
        self.assertEqual(config.log_patterns, ("logs/rsl_rl/*/*",))
        self.assertEqual(config.tag_schema, ("good-flat", "bad-stairs"))
        self.assertEqual(config.remote_sources[0].name, "x-server")
        self.assertEqual(config.remote_sources[0].project, "unitree_rl_mjlab")
        self.assertEqual(config.remote_sources[0].port, 12188)


if __name__ == "__main__":
    unittest.main()
