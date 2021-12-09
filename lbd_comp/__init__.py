__version__ = "1.0.1"

# We add the module path to sys.path
# This is done so that the top-level modules can be imported easily
import sys
from pathlib import Path

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
