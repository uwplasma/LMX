from pathlib import Path

import pytest

from scripts.patch_freemhd_darwin_headers import patch_darwin_header_rules


pytestmark = pytest.mark.unit


def test_patch_darwin_header_rules_updates_c_and_cxx(tmp_path: Path):
    repo_root = tmp_path / "FreeMHD"
    rules_dir = repo_root / "OpenFOAM-v2206" / "wmake" / "rules" / "darwin64Clang"
    rules_dir.mkdir(parents=True)
    cxx = rules_dir / "c++"
    c = rules_dir / "c"
    cxx.write_text(
        "include $(DEFAULT_RULES)/c++$(WM_COMPILE_OPTION)\n"
        "c++FLAGS = $(FOAM_EXTRA_CXXFLAGS) $(LIB_HEADER_DIRS) -fPIC\n"
    )
    c.write_text(
        "sinclude $(DEFAULT_RULES)/c$(WM_COMPILE_OPTION)\n"
        "cFLAGS = $(FOAM_EXTRA_CFLAGS) $(LIB_HEADER_DIRS) -fPIC\n"
    )

    report = patch_darwin_header_rules(repo_root)

    assert report["changed"] is True
    assert all(item["changed"] for item in report["files"])
    assert "DARWIN_LIB_HEADER_DIRS :=" in cxx.read_text()
    assert "$(DARWIN_LIB_HEADER_DIRS) -fPIC" in cxx.read_text()
    assert "DARWIN_LIB_HEADER_DIRS :=" in c.read_text()
    assert "$(DARWIN_LIB_HEADER_DIRS) -fPIC" in c.read_text()


def test_patch_darwin_header_rules_is_idempotent(tmp_path: Path):
    repo_root = tmp_path / "FreeMHD"
    rules_dir = repo_root / "OpenFOAM-v2206" / "wmake" / "rules" / "darwin64Clang"
    rules_dir.mkdir(parents=True)
    content = "include $(DEFAULT_RULES)/c++$(WM_COMPILE_OPTION)\n\nDARWIN_LIB_HEADER_DIRS := x\nc++FLAGS = $(FOAM_EXTRA_CXXFLAGS) $(DARWIN_LIB_HEADER_DIRS) -fPIC\n"
    (rules_dir / "c++").write_text(content)
    (rules_dir / "c").write_text(
        "sinclude $(DEFAULT_RULES)/c$(WM_COMPILE_OPTION)\n\nDARWIN_LIB_HEADER_DIRS := x\ncFLAGS = $(FOAM_EXTRA_CFLAGS) $(DARWIN_LIB_HEADER_DIRS) -fPIC\n"
    )

    report = patch_darwin_header_rules(repo_root)

    assert report["changed"] is False
    assert not any(item["changed"] for item in report["files"])
