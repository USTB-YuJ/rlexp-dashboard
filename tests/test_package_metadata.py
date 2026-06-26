from pathlib import Path
import unittest


class PackageMetadataTests(unittest.TestCase):
    def test_pyproject_uses_readme_as_package_landing_page(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('readme = "README.md"', pyproject)

    def test_readme_documents_open_source_local_deployment_flow(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("pip install rl-exp-dashboard", readme)
        self.assertIn("rl-exp-dashboard serve --workspace ~/rl-exp-dashboard", readme)
        self.assertIn("rl-exp-dashboard sync", readme)
        self.assertIn("rl-exp-dashboard index", readme)
        self.assertIn("docker run -p 7860:7860", readme)

    def test_plain_install_includes_dashboard_server_runtime(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        project_section = pyproject.split("[project.optional-dependencies]", maxsplit=1)[0]

        self.assertIn('"fastapi>=0.110"', project_section)
        self.assertIn('"uvicorn>=0.27"', project_section)
