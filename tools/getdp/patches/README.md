<!-- Copyright (c) 2026 MPhysX -->
<!-- SPDX-License-Identifier: MIT -->

# GetDP patches — licence notice

The files in this directory are patches of [GetDP](https://getdp.info), **not** part of
`mpxlite`'s MIT-licensed code.

`getdp-3.5.0-mpx3.patch`:

* derived from GetDP 3.5.0, Copyright (C) 1997-2022 P. Dular and C. Geuzaine, University of
  Liège;
* modifications Copyright (c) 2026 MPhysX;
* SPDX-License-Identifier: `GPL-2.0-or-later` — licensed under the GNU General Public License,
  version 2 or (at your option) any later version, like GetDP itself. It is **not** covered by
  the MIT License of `mpxlite`.

A full copy of the GNU GPL v2 accompanies the patch in `COPYING` (this directory). The GetDP licence is in `LICENSE.txt` of the GetDP sources
(<https://gitlab.onelab.info/getdp/getdp/-/blob/master/LICENSE.txt>); the GNU GPL v2 text is at
<https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.

A GetDP binary built with this patch as the mpxlite engine links METIS (Apache-2.0,
compatible with the GPL version 3 only): if distributed, such a binary is under the GNU GPL
version 3 or later (GPL-3.0-or-later). The patch itself stays `GPL-2.0-or-later`.

These patches are shipped in the source repository and in the `source/` directory of every
mpxlite release bundle; they are excluded from the Python distributions (wheel and sdist) of
`mpxlite`. How to apply and build: `tools/engine/build_engine.sh`.
