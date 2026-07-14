import ast
import unittest
from pathlib import Path


OVERRIDE = Path(__file__).parents[1] / "overrides" / "okx_perpetual_derivative.py"


class OkxIsolatedOverrideTest(unittest.TestCase):
    def test_place_order_uses_isolated_margin(self) -> None:
        tree = ast.parse(OVERRIDE.read_text())
        place_order = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_place_order"
        )
        td_modes = [
            value.value
            for node in ast.walk(place_order)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
            and key.value == "tdMode"
            and isinstance(value, ast.Constant)
        ]

        self.assertEqual(td_modes, ["isolated"])


if __name__ == "__main__":
    unittest.main()
