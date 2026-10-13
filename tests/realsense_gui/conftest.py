import os
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def qapp_args():
    from PyQt6.QtCore import QLibraryInfo
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / 'platforms')
    return []
