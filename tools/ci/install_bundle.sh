#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Install a bundle the way a user does: check its .sha256, unpack it in a new
# directory, run ./install.sh [--prefix PREFIX] [INSTALL_ARGS...] in a clean
# environment (env -i: only HOME, USER, LOGNAME, LANG and
# PATH=/usr/sbin:/usr/bin:/sbin:/bin; no terminal), which also runs the smoke
# test. The apt step needs root or sudo. With --user NAME, a non-root account
# NAME (passwordless sudo, for the installer's apt step) is created when
# missing and does the install in its own home. Without PREFIX the installer's
# default applies (~/mpxlite of that user).
#
# Usage: tools/ci/install_bundle.sh [--user NAME] TARBALL [PREFIX [INSTALL_ARGS...]]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

AS_USER=""
if [ "${1:-}" = --user ]; then AS_USER="$2"; shift 2; fi
[ $# -ge 1 ] || die "usage: $0 [--user NAME] TARBALL [PREFIX [INSTALL_ARGS...]]"
TARBALL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift
ARGS=()
if [ $# -ge 1 ]; then ARGS=(--prefix "$1"); shift; fi
ARGS+=("$@")
NAME="$(basename "$TARBALL" .tar.gz)"
if [ -f "$TARBALL.sha256" ]; then (cd "$(dirname "$TARBALL")" && sha256sum -c "$(basename "$TARBALL").sha256"); fi

if [ -n "$AS_USER" ]; then
    ensure_user "$AS_USER"
    HOME_DIR="$(getent passwd "$AS_USER" | cut -d: -f6)"
    RUN=(sudo -u "$AS_USER" -H)
else
    AS_USER="$(id -un)"
    HOME_DIR="$HOME"
    RUN=()
fi

UNPACK="$HOME_DIR/bundle-unpack"
say "unpack $NAME as $AS_USER into $UNPACK"
"${RUN[@]}" rm -rf "$UNPACK"
"${RUN[@]}" mkdir -p "$UNPACK"
# Read by the caller, unpacked by the installing user (who may not see the caller's paths).
as_root cat "$TARBALL" | "${RUN[@]}" tar -C "$UNPACK" -xzf -
say "install.sh ${ARGS[*]}"
T0=$(date +%s)
"${RUN[@]}" env -i HOME="$HOME_DIR" USER="$AS_USER" LOGNAME="$AS_USER" LANG=C.UTF-8 \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    bash -c 'cd "$1" && shift && ./install.sh "$@" < /dev/null' _ "$UNPACK/$NAME" "${ARGS[@]}"
echo "install.sh: $(( $(date +%s) - T0 )) s"
