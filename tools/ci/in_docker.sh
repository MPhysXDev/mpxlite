#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Run tools/ci scripts locally in a throw-away Docker container of IMAGE, with
# the checkout at /w/mpxlite: a copy of the files git tracks plus untracked,
# not ignored files (working-tree content), made a git repository with one
# snapshot commit when git is installed (call `snapshot` after apt.sh base). The
# container is removed at the end; CI_WORK is /w/ci-work inside it (copied to
# --out DIR if given).
#
# Usage: tools/ci/in_docker.sh [--out DIR] [--put FILE]... [--mount SRC:DST]... [--hostname H] IMAGE 'COMMANDS'
#   --put FILE        copies FILE (e.g. a bundle and its .sha256) into /w/put/
#   --mount SRC:DST   bind-mounts SRC at DST (e.g. an engine prefix at its build path)
#   --hostname H      container host name (default mpxlite-ci)
#   e.g. tools/ci/in_docker.sh ubuntu:24.04 \
#          'mpxlite/tools/ci/apt.sh base && snapshot && mpxlite/tools/ci/unit.sh'
set -euo pipefail

OUT=""; PUT=(); MOUNTS=(); HOST=(--hostname mpxlite-ci)
while [ $# -gt 2 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --put) PUT+=("$2"); shift 2 ;;
        --mount) MOUNTS+=(-v "$2"); shift 2 ;;
        --hostname) HOST=(--hostname "$2"); shift 2 ;;
        *) break ;;
    esac
done
[ $# -eq 2 ] || { sed -n '5,19p' "$0"; exit 2; }
IMAGE="$1"; CMDS="$2"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="mpxlite-ci-$$"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

docker run -d --name "$NAME" "${HOST[@]}" "${MOUNTS[@]}" "$IMAGE" sleep infinity >/dev/null
docker exec "$NAME" mkdir -p /w/mpxlite /w/ci-work /w/put
for f in "${PUT[@]}"; do docker cp "$f" "$NAME:/w/put/"; done
(cd "$REPO" && git ls-files -z --cached --others --exclude-standard | tar --null -T - -cf -) |
    docker exec -i "$NAME" tar -C /w/mpxlite -xf -
docker exec -e CI_WORK=/w/ci-work -w /w "$NAME" bash -c "
set -euo pipefail
snapshot() {
    git -C /w/mpxlite init -q && git -C /w/mpxlite add -A &&
    git -C /w/mpxlite -c user.name=ci -c user.email=contact@mphysx.com commit -q -m snapshot
}
$CMDS"
if [ -n "$OUT" ]; then mkdir -p "$OUT" && docker cp "$NAME:/w/ci-work/." "$OUT/"; fi
