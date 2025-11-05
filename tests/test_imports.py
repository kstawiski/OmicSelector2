"""Test that all modules can be imported without errors."""

import pytest


@pytest.mark.unit
def test_import_main_package() -> None:
    """Test importing main package."""
    import omicselector2

    assert omicselector2 is not None


@pytest.mark.unit
def test_import_api_module() -> None:
    """Test importing API module."""
    import omicselector2.api

    assert omicselector2.api is not None


@pytest.mark.unit
def test_import_data_module() -> None:
    """Test importing data module."""
    import omicselector2.data

    assert omicselector2.data is not None


@pytest.mark.unit
def test_import_features_module() -> None:
    """Test importing features module."""
    import omicselector2.features

    assert omicselector2.features is not None


@pytest.mark.unit
def test_import_features_classical() -> None:
    """Test importing classical features module."""
    import omicselector2.features.classical

    assert omicselector2.features.classical is not None


@pytest.mark.unit
def test_import_features_graph_based() -> None:
    """Test importing graph-based features module."""
    import omicselector2.features.graph_based

    assert omicselector2.features.graph_based is not None


@pytest.mark.unit
def test_import_features_deep_learning() -> None:
    """Test importing deep learning features module."""
    import omicselector2.features.deep_learning

    assert omicselector2.features.deep_learning is not None


@pytest.mark.unit
def test_import_features_single_cell() -> None:
    """Test importing single-cell features module."""
    import omicselector2.features.single_cell

    assert omicselector2.features.single_cell is not None


@pytest.mark.unit
def test_import_models_module() -> None:
    """Test importing models module."""
    import omicselector2.models

    assert omicselector2.models is not None


@pytest.mark.unit
def test_import_models_classical() -> None:
    """Test importing classical models module."""
    import omicselector2.models.classical

    assert omicselector2.models.classical is not None


@pytest.mark.unit
def test_import_models_neural() -> None:
    """Test importing neural models module."""
    import omicselector2.models.neural

    assert omicselector2.models.neural is not None


@pytest.mark.unit
def test_import_models_graph() -> None:
    """Test importing graph models module."""
    import omicselector2.models.graph

    assert omicselector2.models.graph is not None


@pytest.mark.unit
def test_import_models_multi_omics() -> None:
    """Test importing multi-omics models module."""
    import omicselector2.models.multi_omics

    assert omicselector2.models.multi_omics is not None


@pytest.mark.unit
def test_import_training_module() -> None:
    """Test importing training module."""
    import omicselector2.training

    assert omicselector2.training is not None


@pytest.mark.unit
def test_import_inference_module() -> None:
    """Test importing inference module."""
    import omicselector2.inference

    assert omicselector2.inference is not None


@pytest.mark.unit
def test_import_visualization_module() -> None:
    """Test importing visualization module."""
    import omicselector2.visualization

    assert omicselector2.visualization is not None


@pytest.mark.unit
def test_import_tasks_module() -> None:
    """Test importing tasks module."""
    import omicselector2.tasks

    assert omicselector2.tasks is not None


@pytest.mark.unit
def test_import_utils_module() -> None:
    """Test importing utils module."""
    import omicselector2.utils

    assert omicselector2.utils is not None
