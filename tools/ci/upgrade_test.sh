#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Upgrade paths of install.sh, as a non-root user NAME (created when missing,
# passwordless sudo) in its own ~/mpxlite-upgrade:
#   1. OLD_TARBALL installed fresh; a dummy package installed into venv/ and a
#      file written under plugins/ (what a plug-in installer does); the Gmsh
#      of the engine and the Gmsh programs of mpxlite removed and the gmsh
#      package of PyPI installed into venv/ (an installation where a plug-in or
#      a user installed Gmsh from PyPI);
#   2. NEW_TARBALL over it: upgraded in place (VERSION, mpxlite in venv/, the
#      Gmsh of the engine and the Gmsh jobs in engine/gmsh/, the gmsh command
#      of the engine on PATH, import gmsh
#      giving the module of the engine, the PyPI gmsh gone from venv/, a job
#      run), the dummy package and the plugins/ file kept, smoke test passes;
#   3. NEW_TARBALL again: no-op, exit 0, nothing changed; then, a file of the
#      mpxlite package in venv/ modified (a package of another build with the
#      same version), NEW_TARBALL again: repaired, the package of the bundle
#      back, nothing else changed;
#   4. OLD_TARBALL again: no downgrade, exit 0, nothing changed.
# The apt step is skipped from step 2 on (--no-apt), so these steps also
# check that install.sh needs no terminal and no sudo then.
#
# Usage: tools/ci/upgrade_test.sh OLD_TARBALL NEW_TARBALL [NAME]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || die "usage: $0 OLD_TARBALL NEW_TARBALL [NAME]"
OLD="$1"; NEW="$2"; U="${3:-mpxuser}"
INSTALL="$REPO/tools/ci/install_bundle.sh"
ver() { basename "$1" | cut -d- -f2; }
OLD_V="$(ver "$OLD")"; NEW_V="$(ver "$NEW")"
ensure_user "$U"
HOME_DIR="$(getent passwd "$U" | cut -d: -f6)"
P="$HOME_DIR/mpxlite-upgrade"
as_root rm -rf "$P"
RUN=(sudo -u "$U" -H)
in_env() {
    "${RUN[@]}" env -i HOME="$HOME_DIR" USER="$U" LOGNAME="$U" LANG=C.UTF-8 \
        PATH=/usr/sbin:/usr/bin:/sbin:/bin bash -c '. "$1/activate.sh" && shift && "$@"' _ "$P" "$@"
}
# The state of the installation: VERSION (but the time of the last run of
# install.sh, which a repair rewrites), the engine, activate.sh, the packages
# of the venv, plugins/.
state() { "${RUN[@]}" bash -c 'cd "$1" && grep -v "^installed=" VERSION && sha256sum engine/bin/getdp activate.sh && venv/bin/python -m pip freeze --all | sort && ls -R plugins' _ "$P"; }

say "1. fresh install of $OLD_V, then a plug-in"
"$INSTALL" --user "$U" "$OLD" "$P" --no-smoke
grep -qx "mpxlite=$OLD_V" "$P/VERSION" || die "VERSION is not $OLD_V"
DUMMY="$CI_WORK/dummy-plugin"
as_root rm -rf "$DUMMY"
mkdir -p "$DUMMY/dummy_plugin"
cat > "$DUMMY/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[project]
name = "dummy-plugin"
version = "1.0"
TOML
echo 'VALUE = 42' > "$DUMMY/dummy_plugin/__init__.py"
as_root rm -rf "$HOME_DIR/dummy-plugin"
as_root cp -r "$DUMMY" "$HOME_DIR/dummy-plugin"
as_root chown -R "$U:" "$HOME_DIR/dummy-plugin"
"${RUN[@]}" "$P/venv/bin/python" -m pip install -q "$HOME_DIR/dummy-plugin"
"${RUN[@]}" mkdir -p "$P/plugins/dummy-plugin"
"${RUN[@]}" bash -c 'echo "dummy-plugin=1.0" > "$1/plugins/dummy-plugin/INSTALLED"' _ "$P"
"${RUN[@]}" rm -r "$P/engine/gmsh"
GMSH_PIN="$(sed -n 's/^GMSH_VERSION=//p' "$REPO/tools/bundle/install.sh")"
"${RUN[@]}" "$P/venv/bin/python" -m pip install -q --no-deps --only-binary :all: "gmsh==$GMSH_PIN"
"${RUN[@]}" test -f "$P/venv/lib/libgmsh.so.4.15" || die "the PyPI gmsh did not install its library into venv/lib"

say "2. upgrade $OLD_V -> $NEW_V"
"$INSTALL" --user "$U" "$NEW" "$P" --no-apt
grep -qx "mpxlite=$NEW_V" "$P/VERSION" || die "VERSION is not $NEW_V after the upgrade"
in_env python -c "import mpxlite, dummy_plugin; assert mpxlite.__version__ == '$NEW_V', mpxlite.__version__; assert dummy_plugin.VALUE == 42"
grep -qx "dummy-plugin=1.0" "$P/plugins/dummy-plugin/INSTALLED" || die "plugins/ not kept"
[ -f "$P/engine/gmsh/jobs/layered_section_mesh.py" ] || die "no Gmsh jobs after the upgrade"
[ -f "$P/engine/gmsh/gmsh.py" ] && [ -f "$P/engine/lib/libgmsh.so.4.15" ] || die "no Gmsh in the engine after the upgrade"
[ "$(in_env bash -c 'command -v gmsh')" = "$P/engine/bin/gmsh" ] || die "the gmsh on PATH is not engine/bin/gmsh after the upgrade"
[ "$(in_env env -u DISPLAY gmsh --version)" = "$GMSH_PIN" ] || die "gmsh --version is not $GMSH_PIN after the upgrade"
! "${RUN[@]}" "$P/venv/bin/python" -m pip show -q gmsh 2>/dev/null || die "the PyPI gmsh is still in venv/"
! ls "$P"/venv/lib/libgmsh* 2>/dev/null || die "a Gmsh library is still in venv/lib"
in_env python - "$P" <<'EOF' || die "import gmsh in venv/ does not import the Gmsh module of the engine"
import sys
from pathlib import Path
import gmsh
assert Path(gmsh.__file__).resolve() == Path(sys.argv[1], "engine", "gmsh", "gmsh.py").resolve(), gmsh.__file__
EOF
in_env python - "$GMSH_PIN" <<'EOF' || die "the Gmsh jobs do not run the Gmsh of the engine"
import sys
import tempfile
import mpxlite
params = {"columns": {"x": [0.0, 1.0], "cells": [{"name": "a", "cells": 2}]},
          "core": {"y": [0.0, 1.0], "rows": 2}}
result = mpxlite.run_gmsh_job("layered_section_mesh", parameters=params,
                              runs_dir=tempfile.mkdtemp(), timeout=120)
assert result.gmsh_version == sys.argv[1], result.gmsh_version
assert result.report["mesh"]["nodes"] == 9, result.report["mesh"]
print(f"a Gmsh job runs Gmsh {result.gmsh_version}")
EOF
echo "the PyPI gmsh removed; Gmsh $GMSH_PIN of the engine and the Gmsh jobs installed by the upgrade"
echo "upgraded in place; dummy-plugin and plugins/dummy-plugin/INSTALLED kept"
S1="$(state)"

say "3. $NEW_V again: no-op"
out="$("$INSTALL" --user "$U" "$NEW" "$P" --no-apt 2>&1)" || die "same version: exit $?"
grep -q "nothing to do" <<< "$out" || die "same version: no 'nothing to do' message"$'\n'"$out"
[ "$(state)" = "$S1" ] || die "same version changed the installation"
grep "nothing to do" <<< "$out"

say "3b. $NEW_V again over a modified mpxlite package of the same version: repaired"
PKG="$(in_env python -c 'import mpxlite, os; print(os.path.dirname(mpxlite.__file__))')"
"${RUN[@]}" bash -c 'echo "# another build" >> "$1/gmshjob.py"' _ "$PKG"
out="$("$INSTALL" --user "$U" "$NEW" "$P" --no-apt 2>&1)" || die "modified package: exit $?"
grep -q "repairing it in place" <<< "$out" || die "modified package: not repaired"$'\n'"$out"
grep -q "is not the one of this bundle: replacing it" <<< "$out" \
    || die "modified package: the mpxlite package was not replaced"$'\n'"$out"
! "${RUN[@]}" grep -q "^# another build" "$PKG/gmshjob.py" || die "the modified mpxlite package is still installed"
[ "$(state)" = "$S1" ] || die "the repair changed more than the mpxlite package"
echo "a modified mpxlite package of the same version replaced by the one of the bundle"

say "4. $OLD_V over $NEW_V: no downgrade"
out="$("$INSTALL" --user "$U" "$OLD" "$P" --no-apt 2>&1)" || die "older bundle: exit $?"
grep -q "no downgrade" <<< "$out" || die "older bundle: no 'no downgrade' message"$'\n'"$out"
[ "$(state)" = "$S1" ] || die "the older bundle changed the installation"
grep -A1 "newer than this bundle" <<< "$out"
say "upgrade paths: PASS"
