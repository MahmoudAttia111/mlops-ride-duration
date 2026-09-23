# tests/test_features.py
from prodml.features import clean_arabic

def test_removes_diacritics():
    assert clean_arabic("الْفُنْدُق") == "الفندق"

def test_normalizes_alef():
    assert "ا" in clean_arabic("إمتاز أحمد آسر")

def test_handles_empty_string():
    assert clean_arabic("") == ""