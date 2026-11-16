from importlib.metadata import version

import tenspec


def test_version_matches_metadata_when_distribution_is_installed() -> None:
    # Arrange
    installed_version = version("tenspec")

    # Act
    reported_version = tenspec.__version__

    # Assert
    assert reported_version == installed_version
