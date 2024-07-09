import pkg_resources

from .brie_org import Brie
from .brie_bmi import BrieBMI

__version__ = pkg_resources.get_distribution("brie").version
__all__ = ["Brie", "BrieBMI"]

del pkg_resources
