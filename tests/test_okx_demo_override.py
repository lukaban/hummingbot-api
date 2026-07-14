import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
DERIVATIVE = ROOT / "overrides" / "okx_perpetual_derivative.py"
UTILS = ROOT / "overrides" / "okx_perpetual_utils.py"
COMPOSE = ROOT / "docker-compose.yml"
DOCKER_SERVICE = ROOT / "services" / "docker_service.py"


class OkxDemoOverrideTest(unittest.TestCase):
    def test_demo_domain_is_registered_and_simulated_header_is_forced(self):
        utils = UTILS.read_text()
        derivative = DERIVATIVE.read_text()
        compose = COMPOSE.read_text()
        docker_service = DOCKER_SERVICE.read_text()

        self.assertIn('OTHER_DOMAINS = ["okx_perpetual_demo"]', utils)
        self.assertIn('"x-simulated-trading": "1"', derivative)
        self.assertIn("self._domain == CONSTANTS.DEMO_DOMAIN", derivative)
        self.assertIn('for rule in response["data"] if rule.get("ctVal")', derivative)
        self.assertIn("./overrides/okx_perpetual_utils.py:", compose)
        self.assertIn("okx_perpetual_utils.py:ro", compose)
        self.assertIn("/home/hummingbot/hummingbot/connector/derivative/okx_perpetual/okx_perpetual_derivative.py", docker_service)
        self.assertIn("/home/hummingbot/hummingbot/connector/derivative/okx_perpetual/okx_perpetual_utils.py", docker_service)
        self.assertIn('network_mode="hummingbot-api_emqx-bridge"', docker_service)
        self.assertIn("mqtt_section['mqtt_host'] = 'emqx'", docker_service)
        self.assertIn('self.client.containers.get("okx-agent-service")', docker_service)
        self.assertIn('environment["HUMMINGBOT_AGENT_TOKEN_FILE"]', docker_service)
        self.assertIn("'/run/secrets/hummingbot_agent_token'", docker_service)
        ast.parse(utils)
        ast.parse(derivative)


if __name__ == "__main__":
    unittest.main()
