import sys
from pathlib import Path

import numpy as np
import pytest

# Add the parent directory to the path so we can import from qarm_rl
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.types import Waypoint


# --- FIXTURES ---
# Elles permettent de créer des objets réutilisables pour plusieurs tests
@pytest.fixture
def valid_pos():
    return np.array([[1.0], [2.0], [3.0]])


@pytest.fixture
def valid_vel():
    return np.array([[0.1], [0.0], [0.0]])


# --- TESTS ---


def test_waypoint_initialization_success(valid_pos, valid_vel):
    """Vérifie qu'un waypoint valide est créé correctement."""
    wp = Waypoint(position=valid_pos, velocity=valid_vel)

    assert wp.position.shape == (3, 1)
    assert np.allclose(wp.position, valid_pos)
    # Vérifie que l'accélération a été initialisée à zéro par défaut
    assert np.all(wp.acceleration == 0)
    assert wp.acceleration.shape == (3, 1)


def test_waypoint_default_none_to_zeros(valid_pos):
    """Vérifie que None est bien converti en vecteurs de zéros."""
    wp = Waypoint(position=valid_pos)

    assert isinstance(wp.velocity, np.ndarray)
    assert wp.velocity.shape == (3, 1)
    assert np.all(wp.velocity == 0)


# --- TESTS D'ERREURS (Paramétrés) ---
# On utilise parametrize pour tester plusieurs cas d'échec en une seule fonction
@pytest.mark.parametrize(
    "bad_shape",
    [
        np.array([1, 2, 3]),  # 1D (3,)
        np.array([[1], [2]]),  # 2x1
        np.zeros((3, 3)),  # Matrice 3x3
        np.array([[[1], [2], [3]]]),  # 3D
    ],
)
def test_waypoint_invalid_shapes(bad_shape):
    """Vérifie que toutes ces formes invalides lèvent une ValueError."""
    with pytest.raises(ValueError, match="doit avoir la forme \(3, 1\)"):
        Waypoint(position=bad_shape)


def test_waypoint_invalid_type():
    """Vérifie qu'une liste au lieu d'un array lève une TypeError."""
    with pytest.raises(TypeError):
        Waypoint(position=[1.0, 2.0, 3.0])
