import unittest

from rl_exp_dashboard.config_diff import diff_configs, flatten_config


class ConfigDiffTests(unittest.TestCase):
    def test_flatten_config_uses_dotted_paths_for_nested_values(self):
        config = {
            "env": {
                "rewards": {
                    "action_rate_l2": {"weight": -0.001},
                    "enabled": True,
                },
                "terrain_cols": [0, 1, 2],
            }
        }

        flattened = flatten_config(config)

        self.assertEqual(flattened["env.rewards.action_rate_l2.weight"], -0.001)
        self.assertEqual(flattened["env.rewards.enabled"], True)
        self.assertEqual(flattened["env.terrain_cols.0"], 0)
        self.assertEqual(flattened["env.terrain_cols.2"], 2)

    def test_diff_configs_reports_added_removed_changed_and_type_changes(self):
        before = {
            "agent": {"entropy_coef": 0.01},
            "env": {
                "rewards": {
                    "alive": {"weight": 1.0},
                    "action_rate_l2": {"weight": -0.0001},
                },
                "old_key": "remove-me",
                "num_envs": 2048,
            },
        }
        after = {
            "agent": {"entropy_coef": 0.005},
            "env": {
                "rewards": {
                    "alive": {"weight": 1.0},
                    "action_rate_l2": {"weight": -0.001},
                    "velocity_direction": {"weight": 2.0},
                },
                "num_envs": "2048",
            },
        }

        diffs = diff_configs(before, after)
        by_path = {item.path: item for item in diffs}

        self.assertEqual(by_path["agent.entropy_coef"].kind, "changed")
        self.assertEqual(by_path["agent.entropy_coef"].before, 0.01)
        self.assertEqual(by_path["agent.entropy_coef"].after, 0.005)
        self.assertEqual(by_path["env.rewards.action_rate_l2.weight"].kind, "changed")
        self.assertEqual(by_path["env.rewards.velocity_direction.weight"].kind, "added")
        self.assertEqual(by_path["env.old_key"].kind, "removed")
        self.assertEqual(by_path["env.num_envs"].kind, "type_changed")
        self.assertNotIn("env.rewards.alive.weight", by_path)


if __name__ == "__main__":
    unittest.main()
