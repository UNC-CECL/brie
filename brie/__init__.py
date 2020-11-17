from .brie import Brie

import pkg_resources

#__version__ = pkg_resources.get_distribution("brie").version
__all__ = ["Brie"]

del pkg_resources
