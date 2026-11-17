# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Periodic tiling of a cell mesh (:mod:`mpxlite.cells.tiling`).

A small Gmsh box cell (``tests/integration/cells/box.py``, untouched Gmsh
numbering, clean facet check) tiled ``1 x 1``, ``2 x 1``, ``2 x 2``,
``3 x 3``: node-merge counts against the distinct translated positions,
element counts (inner Bloch faces dropped), every copied element ranks its
nodes as the cell element, outer links (translation ``N T``, node pairs
translated by it, clean facet check), ``retag`` (renaming, dropping,
collisions), ``origin`` placement. The numbering pre-check accepts the
gmsh numbering, :func:`~mpxlite.cells.periodic.renumber_periodic` with
another origin and :func:`~mpxlite.cells.spacegroup.renumber_symmetric`, and
refuses a numbering that breaks the rank of a Bloch facet or that no
macrocell numbering can reproduce on every copy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from mpxlite.cells.periodic import _renumber_by, check_periodic_facet_order, renumber_periodic
from mpxlite.cells.symmetric import build_c4v_mesh, c4v_lattice_ops
from mpxlite.cells.tiling import tile_periodic
from tests.integration.cells.box import BoxParams, generate_mesh
from tests.unit.cells.test_mesh_symmetric import PHYS, L, _wedge

P = 7.5e-3
T1, T2 = (P, 0.0), (0.0, P)
BLOCH = ((301, 302), (303, 304))
_CORNERS = {2: 3, 4: 4}


@pytest.fixture(scope="module")
def cell_mesh(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiling") / "cell.msh"
    raw = path.with_name("raw.msh")
    generate_mesh(BoxParams(P, P, 5e-3, 5e-3, 2.5e-3), raw)
    assert check_periodic_facet_order(raw).ok  # the generator left the gmsh numbering
    shutil.copy(raw, path)
    return path


def _read(path: Path) -> tuple[dict[int, np.ndarray], list[tuple[int, int, int, list[int]]], list]:
    lines = path.read_text(encoding="utf-8").split("\n")
    i = lines.index("$Nodes")
    nodes = {int(t[0]): np.array([float(v) for v in t[1:4]])
             for t in (s.split() for s in lines[i + 2 : i + 2 + int(lines[i + 1])])}  # fmt: skip
    i = lines.index("$Elements")
    elems = []
    for s in lines[i + 2 : i + 2 + int(lines[i + 1])]:
        t = [int(v) for v in s.split()]
        elems.append((t[1], t[3], t[4], t[3 + t[2] :]))
    links = []
    j = lines.index("$Periodic") + 1
    for _ in range(int(lines[j])):
        d, sl, ma = (int(v) for v in lines[j + 1].split())
        aff = [float(v) for v in lines[j + 2].split()[1:]]
        n = int(lines[j + 3])
        pairs = [tuple(int(v) for v in lines[j + 4 + k].split()) for k in range(n)]
        links.append((d, sl, ma, aff, pairs))
        j += 3 + n
    return nodes, elems, links


def _ranks(elems, types=(2, 4)) -> list[tuple[int, ...]]:
    return [tuple(np.argsort(v[: _CORNERS[t]]).tolist()) for t, _, _, v in elems if t in types]


@pytest.mark.parametrize("shape", [(1, 1), (2, 1), (2, 2), (3, 3)])
def test_tiling_counts_ranks_and_links(cell_mesh: Path, tmp_path: Path, shape) -> None:
    n1, n2 = shape
    rep = tile_periodic(cell_mesh, tmp_path / "m.msh", n1, n2, T1, T2, bloch_tags=BLOCH)
    assert rep.shape == shape and rep.periodic.ok, rep.periodic.summary()
    cnodes, celems, clinks = _read(cell_mesh)
    nodes, elems, links = _read(rep.path)
    # merge counts: the macrocell nodes are the distinct translated cell positions
    xyz = np.array(list(cnodes.values()))
    shifts = [i * np.array([*T1, 0]) + j * np.array([*T2, 0]) for j in range(n2) for i in range(n1)]
    distinct = np.unique(np.round(np.concatenate([xyz + s for s in shifts]) / P * 1e6), axis=0)
    assert rep.n_nodes == len(nodes) == len(distinct)
    assert rep.n_merged == n1 * n2 * len(cnodes) - len(nodes)
    # elements: every copy, inner Bloch faces dropped, outer ones kept
    count = {tag: sum(1 for _, p, _, _ in celems if p == tag) for tag in (301, 302, 303, 304)}
    bloch = sum(count.values())
    outer = n2 * (count[301] + count[302]) + n1 * (count[303] + count[304])
    assert rep.n_elements == len(elems) == n1 * n2 * (len(celems) - bloch) + outer
    # every copied tetrahedron ranks its nodes as the cell tetrahedron it copies
    assert _ranks(elems, (4,)) == n1 * n2 * _ranks(celems, (4,))
    # outer links: translations N T1 / M T2, node pairs translated by it
    ent_count = {p: len({e for _, q, e, _ in celems if q == p}) for p in (302, 304)}
    assert len(links) == n2 * ent_count[302] + n1 * ent_count[304]
    for _, _, _, aff, pairs in links:
        vec = np.array([aff[3], aff[7], aff[11]])
        assert any(np.allclose(vec, v) for v in ([n1 * P, 0, 0], [0, n2 * P, 0]))
        for a, b in pairs:
            assert np.allclose(nodes[a], nodes[b] + vec, atol=1e-12)
    assert len(clinks) == ent_count[302] + ent_count[304]


def test_retag_and_origin(cell_mesh: Path, tmp_path: Path) -> None:
    def retag(dim, tag, name, i, j):
        if name == "vol_top":
            return (1000 + 10 * i + j, f"top_{i}_{j}")
        if name == "outer_bot":
            return None  # dropped on every copy
        return (tag, name)

    rep = tile_periodic(cell_mesh, tmp_path / "m.msh", 2, 2, T1, T2, bloch_tags=BLOCH,
                        retag=retag, origin=(-0.5, -0.5))  # fmt: skip
    nodes, elems, _ = _read(rep.path)
    text = rep.path.read_text(encoding="utf-8")
    for i in (0, 1):
        for j in (0, 1):
            assert f'3 {1000 + 10 * i + j} "top_{i}_{j}"' in text
    assert "outer_bot" not in text and not any(p == 102 for _, p, _, _ in elems)
    # origin (-1/2, -1/2): the 2 x 2 macrocell is centred on the cell (centred on 0)
    xyz = np.array(list(nodes.values()))
    assert np.allclose(xyz[:, 0].min(), -P) and np.allclose(xyz[:, 0].max(), P)
    assert np.allclose(xyz[:, 1].min(), -P) and np.allclose(xyz[:, 1].max(), P)
    with pytest.raises(ValueError, match="named both"):
        tile_periodic(cell_mesh, tmp_path / "x.msh", 2, 1, T1, T2, bloch_tags=BLOCH,
                      retag=lambda d, t, n, i, j: (t, f"{n}_{i}"))  # fmt: skip
    with pytest.raises(ValueError, match="positive integers"):
        tile_periodic(cell_mesh, tmp_path / "x.msh", 0, 1, T1, T2, bloch_tags=BLOCH)


def test_numbering_check_accepts_translation_invariant_numberings(
    cell_mesh: Path, tmp_path: Path
) -> None:
    # mesh-2: renumber_periodic with another origin; renumber_symmetric (the C4v wedge mesh)
    other = tmp_path / "other.msh"
    renumber_periodic(cell_mesh, T1, T2, output=other, origin=(-0.25, -0.25))
    coords, blocks = _wedge()
    ops = c4v_lattice_ops(L)
    sym = build_c4v_mesh(tmp_path / "sym.msh", coords, blocks, L, PHYS, numbering=ops)
    for src, (t1, t2) in ((other, (T1, T2)), (sym, ((L, 0.0), (0.0, L)))):
        before = src.read_bytes()
        rep = tile_periodic(src, tmp_path / f"m_{src.stem}.msh", 2, 2, t1, t2, bloch_tags=BLOCH)
        assert rep.periodic.ok
        assert _ranks(_read(rep.path)[1], (4,)) == 4 * _ranks(_read(src)[1], (4,))
        assert src.read_bytes() == before  # the cell is only read
    # no probe copy next to the output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m_other.msh", "m_sym.msh", "other.msh",
                                                          "sym.msh"]  # fmt: skip


def test_numbering_check_refuses_inconsistent_numberings(cell_mesh: Path, tmp_path: Path) -> None:
    # a Bloch facet whose node ranking is not its master's
    rng = np.random.default_rng(1)
    bad = tmp_path / "bad.msh"
    _renumber_by(cell_mesh, lambda x, y, z: rng.random(), output=bad)
    assert check_periodic_facet_order(bad).n_bad
    with pytest.raises(ValueError, match="rank its nodes as its master image"):
        tile_periodic(bad, tmp_path / "m.msh", 2, 1, T1, T2, bloch_tags=BLOCH)
    # Bloch facets consistent, but no macrocell numbering ranks both glued copies as the cell
    cyc = _cycle_cell(tmp_path / "cyc.msh")
    assert check_periodic_facet_order(cyc).ok
    with pytest.raises(ValueError, match="cycle of contradictory ranks"):
        tile_periodic(cyc, tmp_path / "m.msh", 2, 1, (1.0, 0.0), (0.0, 1.0), bloch_tags=BLOCH)


def _cycle_cell(path: Path) -> Path:
    """x = -1/2 face p0..p3 (two triangles; p0, p3 share none), x = +1/2 face q0..q3.

    Both faces rank every triangle alike, but p0 < p3 while q3 < q0; the
    segments p0 - y - p3 and q3 - x - q0 then give m3 < x < m0 on the left
    copy and m0 < y < m3 on the right one (m = q of the left = p of the right).
    """
    yz = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    num_p, num_q = (5, 1, 2, 7), (10, 3, 4, 8)
    nodes = {n: (-0.5, *yz[k]) for k, n in enumerate(num_p)}
    nodes |= {n: (0.5, *yz[k]) for k, n in enumerate(num_q)}
    nodes |= {6: (0.0, 0.6, 0.5), 9: (0.0, 0.3, 0.5)}
    tris = [(0, 1, 2), (1, 3, 2)]
    elems = [f"2 2 301 1 {' '.join(str(num_p[i]) for i in t)}" for t in tris]
    elems += [f"2 2 302 2 {' '.join(str(num_q[i]) for i in t)}" for t in tris]
    elems += ["1 2 7 3 5 6", "1 2 7 3 6 7", "1 2 7 3 8 9", "1 2 7 3 9 10"]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    lines += [f"{n} {x} {y} {z}" for n, (x, y, z) in sorted(nodes.items())]
    lines += ["$EndNodes", "$Elements", str(len(elems))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elems)]
    lines += ["$EndElements", "$Periodic", "1", "2 2 1",
              "Affine 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1", "4"]  # fmt: skip
    lines += [f"{q} {p}" for q, p in zip(num_q, num_p, strict=True)] + ["$EndPeriodic", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
