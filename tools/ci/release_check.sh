#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Release gate: the tag must be v<version of the package> and CHANGELOG.md
# must have an entry for that version.
#
# Usage: tools/ci/release_check.sh TAG
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -eq 1 ] || die "usage: $0 TAG"
VERSION="$(mpxlite_version)"
[ -n "$VERSION" ] || die "no __version__ in src/mpxlite/__init__.py"
[ "$1" = "v$VERSION" ] || die "tag $1 does not match the package version $VERSION (expected v$VERSION)"
grep -q "^## $VERSION " "$REPO/CHANGELOG.md" || die "CHANGELOG.md has no '## $VERSION' entry"
echo "tag $1 = package version $VERSION; CHANGELOG entry present"
