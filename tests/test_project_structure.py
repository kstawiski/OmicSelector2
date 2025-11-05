"""Test that project structure is correctly set up."""

import pathlib

import pytest

# Get project root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.parent


@pytest.mark.unit
def test_src_directory_exists() -> None:
    """Test that src directory exists."""
    src_dir = PROJECT_ROOT / "src"
    assert src_dir.exists(), "src/ directory should exist"
    assert src_dir.is_dir(), "src/ should be a directory"


@pytest.mark.unit
def test_tests_directory_exists() -> None:
    """Test that tests directory exists."""
    tests_dir = PROJECT_ROOT / "tests"
    assert tests_dir.exists(), "tests/ directory should exist"
    assert tests_dir.is_dir(), "tests/ should be a directory"


@pytest.mark.unit
def test_docs_directory_exists() -> None:
    """Test that docs directory exists."""
    docs_dir = PROJECT_ROOT / "docs"
    assert docs_dir.exists(), "docs/ directory should exist"
    assert docs_dir.is_dir(), "docs/ should be a directory"


@pytest.mark.unit
def test_docker_directory_exists() -> None:
    """Test that docker directory exists."""
    docker_dir = PROJECT_ROOT / "docker"
    assert docker_dir.exists(), "docker/ directory should exist"
    assert docker_dir.is_dir(), "docker/ should be a directory"


@pytest.mark.unit
def test_scripts_directory_exists() -> None:
    """Test that scripts directory exists."""
    scripts_dir = PROJECT_ROOT / "scripts"
    assert scripts_dir.exists(), "scripts/ directory should exist"
    assert scripts_dir.is_dir(), "scripts/ should be a directory"


@pytest.mark.unit
def test_pyproject_toml_exists() -> None:
    """Test that pyproject.toml exists."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml should exist"
    assert pyproject.is_file(), "pyproject.toml should be a file"


@pytest.mark.unit
def test_pytest_ini_exists() -> None:
    """Test that pytest.ini exists."""
    pytest_ini = PROJECT_ROOT / "pytest.ini"
    assert pytest_ini.exists(), "pytest.ini should exist"
    assert pytest_ini.is_file(), "pytest.ini should be a file"


@pytest.mark.unit
def test_gitignore_exists() -> None:
    """Test that .gitignore exists."""
    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore should exist"
    assert gitignore.is_file(), ".gitignore should be a file"


@pytest.mark.unit
def test_readme_exists() -> None:
    """Test that README.md exists."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "README.md should exist"
    assert readme.is_file(), "README.md should be a file"


@pytest.mark.unit
def test_claude_md_exists() -> None:
    """Test that CLAUDE.md exists."""
    claude_md = PROJECT_ROOT / "CLAUDE.md"
    assert claude_md.exists(), "CLAUDE.md should exist"
    assert claude_md.is_file(), "CLAUDE.md should be a file"


@pytest.mark.unit
def test_env_example_exists() -> None:
    """Test that .env.example exists."""
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.exists(), ".env.example should exist"
    assert env_example.is_file(), ".env.example should be a file"


@pytest.mark.unit
def test_package_structure() -> None:
    """Test that main package structure is correct."""
    package_dir = PROJECT_ROOT / "src" / "omicselector2"
    assert package_dir.exists(), "src/omicselector2/ should exist"

    # Check for key subdirectories
    expected_dirs = [
        "api",
        "data",
        "features",
        "models",
        "training",
        "inference",
        "visualization",
        "tasks",
        "utils",
    ]

    for dir_name in expected_dirs:
        dir_path = package_dir / dir_name
        assert dir_path.exists(), f"{dir_name}/ should exist in package"
        assert dir_path.is_dir(), f"{dir_name}/ should be a directory"


@pytest.mark.unit
def test_test_structure() -> None:
    """Test that test directory structure is correct."""
    tests_dir = PROJECT_ROOT / "tests"

    # Check for key subdirectories
    expected_dirs = ["unit", "integration", "fixtures"]

    for dir_name in expected_dirs:
        dir_path = tests_dir / dir_name
        assert dir_path.exists(), f"tests/{dir_name}/ should exist"
        assert dir_path.is_dir(), f"tests/{dir_name}/ should be a directory"
