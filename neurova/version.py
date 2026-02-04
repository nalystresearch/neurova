# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Version information for Neurova package"""

__version__ = "0.0.1"
__version_info__ = (0, 0, 1)

# build information
__build__ = "stable"
__date__ = "2026-02-04"

# library metadata
NEUROVA_VERSION = __version__
NEUROVA_BUILD = __build__
NEUROVA_DATE = __date__

def get_version():
    """Get the current version string"""
    return __version__

def get_version_info():
    """Get the version as a tuple"""
    return __version_info__

def get_build_info():
    """Get complete build information"""
    return {
        "version": __version__,
        "build": __build__,
        "date": __date__,
        "version_info": __version_info__,
    }
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.