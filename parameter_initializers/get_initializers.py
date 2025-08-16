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
    name_lower = name.lower()

    initializer_map = {
        "gaussian": GaussianInitialization,
        "zero": ZeroInitialization
    }

    if name_lower not in initializer_map:
        raise ValueError(
            "Initializer doesnt exist! Please choose a valid initializer [zero, gaussian]."
            )

    return initializer_map[name_lower]()

if __name__ == "__main__":
    initializer = get_initializer("zero")
    print(initializer)
