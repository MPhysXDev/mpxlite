#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Nightly / release test of a bundle on a fresh machine or container of its
# Ubuntu release (nothing installed beyond the base image), run as root: sudo
# installed, then, as a new non-root user, the bundle installed into its
# default location ~/mpxlite in a clean environment (install.sh, apt step and
# smoke test included), the smoke test, the full unit suite and the full
# integration suite of the installed copy (tools/ci/bundle_tests.sh full),
# and the upgrade paths from OLD_TARBALL (tools/ci/upgrade_test.sh).
#
# Usage: tools/ci/nightly_test.sh TARBALL OLD_TARBALL REPORT_DIR [USER]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 3 ] || die "usage: $0 TARBALL OLD_TARBALL REPORT_DIR [USER]"
TARBALL="$1"; OLD="$2"; REPORTS="$3"; U="${4:-mpxuser}"
if ! command -v sudo >/dev/null; then
    [ "$(id -u)" = 0 ] || die "sudo missing and not root"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends sudo >/dev/null
fi
"$REPO/tools/ci/install_bundle.sh" --user "$U" "$TARBALL"
HOME_DIR="$(getent passwd "$U" | cut -d: -f6)"
rc=0
"$REPO/tools/ci/bundle_tests.sh" full "$HOME_DIR/mpxlite" "$REPORTS" || rc=1
"$REPO/tools/ci/upgrade_test.sh" "$OLD" "$TARBALL" "$U" || rc=1
exit "$rc"
