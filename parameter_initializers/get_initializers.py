"""This module is used to choose a initializer."""
from .gaussian_initialization import GaussianInitialization
from .zero_initialization import ZeroInitialization

def get_initializer(name: str) -> GaussianInitialization | ZeroInitialization:
    """
    Gets initializer.

    Args:
        name (str): Name of the initializer.

    Returns:
        GaussianInitialization | ZeroInitialization: The initializor class
    """
    initializer_map = {
        "gaussian": GaussianInitialization,
        "zero": ZeroInitialization
    }

    return initializer_map[name.lower()]()

if __name__ == "__main__":
    initializer = get_initializer("zero")
    print(initializer)
