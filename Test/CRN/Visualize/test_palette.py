import re
import unittest
from dataclasses import fields

from synkit.CRN.Visualize.palette import ColorPalette, get_palette, palette_names

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class TestPaletteRegistry(unittest.TestCase):
    def test_names_are_sorted_and_non_empty(self):
        names = palette_names()
        self.assertTrue(names)
        self.assertEqual(names, sorted(names))

    def test_every_registered_palette_resolves(self):
        for name in palette_names():
            with self.subTest(palette=name):
                self.assertIsInstance(get_palette(name), ColorPalette)

    def test_default_palette_exists(self):
        self.assertIsInstance(get_palette(), ColorPalette)

    def test_unknown_palette_raises_with_available_names(self):
        with self.assertRaises(ValueError) as ctx:
            get_palette("chartreuse_explosion")
        self.assertIn("Available", str(ctx.exception))

    def test_every_slot_is_a_hex_color(self):
        slots = [f.name for f in fields(ColorPalette)]
        for name in palette_names():
            palette = get_palette(name)
            for slot in slots:
                with self.subTest(palette=name, slot=slot):
                    self.assertRegex(getattr(palette, slot), HEX_RE)


class TestPaletteOverrides(unittest.TestCase):
    def test_get_palette_applies_overrides(self):
        palette = get_palette(palette_names()[0], background="#000000")
        self.assertEqual(palette.background, "#000000")

    def test_with_overrides_returns_a_new_palette(self):
        base = get_palette(palette_names()[0])
        derived = base.with_overrides(background="#123456")
        self.assertEqual(derived.background, "#123456")
        self.assertNotEqual(base.background, "#123456")

    def test_palette_is_frozen(self):
        palette = get_palette(palette_names()[0])
        with self.assertRaises(Exception):
            palette.background = "#000000"  # type: ignore[misc]

    def test_unknown_override_raises(self):
        with self.assertRaises(TypeError):
            get_palette(palette_names()[0], not_a_slot="#000000")


if __name__ == "__main__":
    unittest.main()
