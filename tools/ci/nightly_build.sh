#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Nightly / release build on the Ubuntu release the bundle targets, as root or
# as a user with sudo: the build prerequisites (apt), the engine in
# /opt/mpxlite-engine (tools/ci/engine.sh: built, or completed and checked when
# a cache restored it there), the bundle (tools/ci/bundle.sh)
# and, for the upgrade test, a bundle of the same tree under a lower version
# (tools/ci/old_bundle.sh, never published).
#
# Usage: tools/ci/nightly_build.sh OUT_DIR UPGRADE_DIR [--jobs N]
# Writes OUT_DIR/mpxlite-<version>-ubuntu<release>-x86_64.tar.gz(.sha256) and
# UPGRADE_DIR/mpxlite-0.0.1.dev0-ubuntu<release>-x86_64.tar.gz(.sha256).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || die "usage: $0 OUT_DIR UPGRADE_DIR [--jobs N]"
OUT="$1"; UPG="$2"; shift 2
"$REPO/tools/ci/apt.sh" base engine
"$REPO/tools/ci/engine.sh" "$ENGINE_PREFIX_DEFAULT" "$@"
"$REPO/tools/ci/bundle.sh" "$ENGINE_PREFIX_DEFAULT" "$OUT"
"$REPO/tools/ci/old_bundle.sh" "$ENGINE_PREFIX_DEFAULT" "$UPG"
