import pkg_resources

from .brie import Brie

__version__ = pkg_resources.get_distribution("brie").version
__all__ = ["Brie"]

del pkg_resources
