#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Install the Ubuntu packages a CI job needs (as root, or through sudo).
#
# Usage: tools/ci/apt.sh GROUP...
#   base     git, ca-certificates, sudo, python3 + venv (bare containers)
#   engine   build prerequisites of tools/engine/build_engine.sh (as listed in
#            its header) and of tools/bundle/make_bundle.sh (binutils, venv)
set -euo pipefail

declare -A GROUP_PKGS=(
    [base]="git ca-certificates sudo python3 python3-venv"
    [engine]="build-essential gfortran cmake git python3 python3-dev ca-certificates openmpi-bin libopenmpi-dev libopenblas-dev libfreetype-dev libx11-dev libxext-dev libxft-dev libxrender-dev libfontconfig-dev libgl-dev libglu1-mesa-dev libjpeg-dev libpng-dev zlib1g-dev binutils python3-venv"
)
[ $# -gt 0 ] || { sed -n '5,10p' "$0"; exit 2; }
pkgs=()
for g in "$@"; do
    [ -n "${GROUP_PKGS[$g]:-}" ] || { echo "unknown group: $g" >&2; exit 2; }
    read -r -a list <<< "${GROUP_PKGS[$g]}"
    pkgs+=("${list[@]}")
done
SUDO=()
if [ "$(id -u)" != 0 ]; then SUDO=(sudo); fi
export DEBIAN_FRONTEND=noninteractive
"${SUDO[@]}" apt-get update -qq
"${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${pkgs[@]}" >/dev/null
echo "installed: ${pkgs[*]}"
