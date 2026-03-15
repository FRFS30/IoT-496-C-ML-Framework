# python/iotids/__init__.py
#
# Kept minimal on purpose.  Eager "from . import ..." lines here cause
# circular imports because Python tries to initialize every subpackage at
# the same time.  Let callers import directly from submodule paths instead:
#
#   from python.iotids.data.csv_reader import read_csv          # correct
#   from python.iotids import data                              # also fine
#   from python.iotids import core, utils, data, ...            # WRONG — circular

__version__ = "0.1.0"
__version__ = "0.1.0"
__version__ = "0.1.0"
