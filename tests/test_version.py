"""Test version and package metadata."""

import pytest


@pytest.mark.unit
def test_version_exists() -> None:
    """Test that version is accessible."""
    from omicselector2 import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


@pytest.mark.unit
def test_version_format() -> None:
    """Test that version follows semantic versioning format."""
    from omicselector2 import __version__

    # Should be in format: major.minor.patch or major.minor.patch-suffix
    parts = __version__.split(".")
    assert len(parts) >= 2, "Version should have at least major.minor"


@pytest.mark.unit
def test_metadata_exists() -> None:
    """Test that package metadata is accessible."""
    from omicselector2 import __author__, __email__, __license__

    assert __author__ is not None
    assert __email__ is not None
    assert __license__ is not None


@pytest.mark.unit
def test_package_docstring() -> None:
    """Test that main package has docstring."""
    import omicselector2

    assert omicselector2.__doc__ is not None
    assert len(omicselector2.__doc__) > 50
    assert "OmicSelector2" in omicselector2.__doc__
