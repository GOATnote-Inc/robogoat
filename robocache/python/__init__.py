"""Python package entry point for the RoboCache library."""

from .robocache import *  # noqa: F401,F403

# Re-export the module level metadata explicitly so tools like help() work.
from .robocache import __all__, __doc__, __version__  # noqa: F401
