#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Build a bundle of this checkout under another (lower) version number, for
# the upgrade test only: a copy of the tracked files with __version__ set to
# VERSION, made a git repository, bundled with the engine in ENGINE_PREFIX.
# Such a bundle is never published.
#
# Usage: tools/ci/old_bundle.sh ENGINE_PREFIX OUT_DIR [VERSION]   (default 0.0.1.dev0)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || die "usage: $0 ENGINE_PREFIX OUT_DIR [VERSION]"
ENGINE="$1"; OUT="$2"; OLD="${3:-0.0.1.dev0}"
COPY="$CI_WORK/old-checkout"
rm -rf "$COPY"
mkdir -p "$COPY"
(cd "$REPO" && git ls-files -z --cached --others --exclude-standard | tar --null -T - -cf -) | tar -C "$COPY" -xf -
sed -i "s/^__version__ = \".*\"/__version__ = \"$OLD\"/" "$COPY/src/mpxlite/__init__.py"
git -C "$COPY" init -q
git -C "$COPY" add -A
git -C "$COPY" -c user.name=ci -c user.email=contact@mphysx.com commit -q -m "upgrade test $OLD"
"$COPY/tools/ci/bundle.sh" "$ENGINE" "$OUT"
