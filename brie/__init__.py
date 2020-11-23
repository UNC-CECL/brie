from .brie import Brie
from .brie_bmi import BrieBMI

import pkg_resources

__version__ = pkg_resources.get_distribution("brie").version
__all__ = ["Brie", "BrieBMI"]

del pkg_resources
