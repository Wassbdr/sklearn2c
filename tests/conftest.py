import shutil

import pytest

requires_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not found")
