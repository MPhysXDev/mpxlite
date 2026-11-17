# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests of mpxlite.cells.periodic (facet node order of the Bloch maps).

Synthetic meshes: a unit cube whose x = 1 face (slave, entity 2) is the
x = 0 face (master, entity 1) translated by T_1 = (1, 0); each face is
split into two triangles. The tests on meshes built with Gmsh are in
``tests/integration/cells``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mpxlite.cells import (
    PeriodicFacetReport,
    UnsupportedMeshFormatError,
    align_periodic_facets,
    check_periodic_facet_order,
    renumber_periodic,
)

# Face corners (y, z); the two triangles of a face, as corner indices.
_CORNERS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
_TRIS = [(0, 1, 2), (1, 3, 2)]
_AFFINE = "1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1"


def _face_nodes(slave_tags: list[int]) -> dict[int, tuple[float, float, float]]:
    nodes = {k + 1: (0.0, y, z) for k, (y, z) in enumerate(_CORNERS)}
    nodes.update({tag: (1.0, y, z) for tag, (y, z) in zip(slave_tags, _CORNERS, strict=True)})
    return nodes


def _msh22(
    path: Path,
    slave_tags: list[int],
    *,
    periodic: bool = True,
    pairs: list[tuple[int, int]] | None = None,
    affine: bool = True,
    extra_elements: tuple[str, ...] = (),
    slave_tris: list[tuple[int, int, int]] | None = None,
) -> Path:
    """MSH 2.2 file: master corners 1..4, slave corners ``slave_tags``.

    ``slave_tris``: stored corner order of the slave triangles (default: the master's).
    """
    nodes = _face_nodes(slave_tags)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    lines += [f"{t} {x:g} {y:g} {z:g}" for t, (x, y, z) in sorted(nodes.items())]
    lines += ["$EndNodes", "$Elements"]
    elements = []
    for entity, tags, tris in ((1, [1, 2, 3, 4], _TRIS), (2, slave_tags, slave_tris or _TRIS)):
        for tri in tris:
            elements.append(f"2 2 {300 + entity} {entity} " + " ".join(str(tags[i]) for i in tri))
    elements += list(extra_elements)
    lines += [str(len(elements))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elements)]
    lines += ["$EndElements"]
    if periodic:
        pairs = pairs if pairs is not None else list(zip(slave_tags, [1, 2, 3, 4], strict=True))
        lines += ["$Periodic", "1", "2 2 1"]
        lines += [f"Affine {_AFFINE}"] if affine else []
        lines += [str(len(pairs))] + [f"{a} {b}" for a, b in pairs] + ["$EndPeriodic"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _msh41(path: Path, slave_tags: list[int], *, curve_link: bool) -> Path:
    """MSH 4.1 file; the surface link lists NO node (boundary nodes live on curves).

    With ``curve_link`` the nodes are listed under a dim-1 link of the same
    affine map, otherwise they are matched geometrically.
    """
    nodes = _face_nodes(slave_tags)
    lines = ["$MeshFormat", "4.1 0 8", "$EndMeshFormat"]
    lines += ["$Nodes", f"2 {len(nodes)} 1 {max(nodes)}"]
    for entity, tags in ((1, [1, 2, 3, 4]), (2, slave_tags)):
        lines += [f"2 {entity} 0 4"] + [str(t) for t in tags]
        lines += [" ".join(f"{c:g}" for c in nodes[t]) for t in tags]
    lines += ["$EndNodes", "$Elements", "2 4 1 4"]
    k = 0
    for entity, tags in ((1, [1, 2, 3, 4]), (2, slave_tags)):
        lines += [f"2 {entity} 2 2"]
        for tri in _TRIS:
            k += 1
            lines += [f"{k} " + " ".join(str(tags[i]) for i in tri)]
    lines += ["$EndElements", "$Periodic", "2" if curve_link else "1"]
    if curve_link:
        lines += ["1 7 5", f"16 {_AFFINE}", "4"]
        lines += [f"{a} {b}" for a, b in zip(slave_tags, [1, 2, 3, 4], strict=True)]
    lines += ["2 2 1", f"16 {_AFFINE}", "0", "$EndPeriodic"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


GOOD = [5, 6, 7, 8]  # same order as the master corners 1..4
BAD = [8, 7, 6, 5]  # reversed: every facet's rank is reversed


# ----- check -------------------------------------------------------------------


def test_good_order_is_ok(tmp_path: Path) -> None:
    report = check_periodic_facet_order(_msh22(tmp_path / "good.msh", GOOD))
    assert isinstance(report, PeriodicFacetReport)
    assert report.ok and report.has_periodic
    assert (report.n_links, report.n_facets, report.n_bad) == (1, 2, 0)
    assert report.msh_version == "2.2"
    assert report.path == (tmp_path / "good.msh").resolve()
    assert "0 with a node order not preserved" in report.summary()


def test_reversed_order_is_reported(tmp_path: Path) -> None:
    report = check_periodic_facet_order(_msh22(tmp_path / "bad.msh", BAD))
    assert not report.ok
    assert (report.n_facets, report.n_bad) == (2, 2)
    first = report.bad[0]
    assert (first.slave_entity, first.master_entity) == (2, 1)
    assert first.nodes == (8, 7, 6) and first.image == (1, 2, 3)
    assert "2 with a node order not preserved" in report.summary()


def test_partial_violation_counts_only_bad_facets(tmp_path: Path) -> None:
    # Corners -> slave tags (5, 6, 8, 7): triangle (0, 1, 2) -> (5, 6, 8) keeps the
    # rank of its image (1, 2, 3); (1, 3, 2) -> (6, 7, 8) does not: (2, 4, 3).
    report = check_periodic_facet_order(_msh22(tmp_path / "mixed.msh", [5, 6, 8, 7]))
    assert (report.n_facets, report.n_bad) == (2, 1)
    assert report.bad[0].nodes == (6, 7, 8)


def test_no_periodic_section(tmp_path: Path) -> None:
    report = check_periodic_facet_order(_msh22(tmp_path / "np.msh", BAD, periodic=False))
    assert report.ok and not report.has_periodic
    assert report.n_facets == 0


def test_duplicate_physical_elements_counted_once(tmp_path: Path) -> None:
    dup = ("2 2 999 2 8 7 6",)  # same triangle under a second physical group
    report = check_periodic_facet_order(_msh22(tmp_path / "d.msh", BAD, extra_elements=dup))
    assert (report.n_facets, report.n_bad) == (2, 2)


def test_quadrangles_on_slave_are_unchecked(tmp_path: Path) -> None:
    quad = ("3 2 302 2 5 6 8 7",)
    report = check_periodic_facet_order(_msh22(tmp_path / "q.msh", GOOD, extra_elements=quad))
    assert report.ok and report.n_unchecked == 1
    assert "1 non-triangular unchecked" in report.summary()


def test_missing_pairs_matched_geometrically(tmp_path: Path) -> None:
    # Only one pair listed: the other nodes are found through the Affine map.
    path = _msh22(tmp_path / "g.msh", BAD, pairs=[(8, 1)])
    report = check_periodic_facet_order(path)
    assert (report.n_bad, report.n_unmapped) == (2, 0)
    # No Affine line: the translation is derived from the listed pair.
    path = _msh22(tmp_path / "g2.msh", GOOD, pairs=[(5, 1)], affine=False)
    assert check_periodic_facet_order(path).ok


def test_unmapped_facets_are_not_ok(tmp_path: Path) -> None:
    # No Affine line and no usable pair: nothing can be mapped.
    path = _msh22(tmp_path / "u.msh", GOOD, pairs=[], affine=False)
    report = check_periodic_facet_order(path)
    assert report.n_unmapped == 2 and not report.ok
    assert "2 unmapped" in report.summary()


@pytest.mark.parametrize("curve_link", [True, False])
def test_msh41(tmp_path: Path, curve_link: bool) -> None:
    good = check_periodic_facet_order(_msh41(tmp_path / "g.msh", GOOD, curve_link=curve_link))
    assert good.msh_version == "4.1"
    assert good.ok and (good.n_facets, good.n_unmapped) == (2, 0)
    bad = check_periodic_facet_order(_msh41(tmp_path / "b.msh", BAD, curve_link=curve_link))
    assert (bad.n_bad, bad.n_unmapped) == (2, 0)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("$NOD\n0\n$ENDNOD\n", "no \\$MeshFormat"),
        ("$MeshFormat\n2.2 1 8\n$EndMeshFormat\n", "binary"),
        ("$MeshFormat\n4.0 0 8\n$EndMeshFormat\n", "MSH 4.0 is not supported"),
        ("$MeshFormat\n3.0 0 8\n$EndMeshFormat\n", "MSH 3.0 is not supported"),
        ("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", "no \\$Nodes"),
        ("$MeshFormat\nx\n$EndMeshFormat\n", "unreadable"),
    ],
)
def test_unsupported_formats(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "m.msh"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UnsupportedMeshFormatError, match=match):
        check_periodic_facet_order(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_periodic_facet_order(tmp_path / "nope.msh")


def test_report_cached_by_content(tmp_path: Path, mocker: object) -> None:
    from mpxlite.cells import periodic

    mocker.patch.object(periodic, "_digest_memo", {})  # type: ignore[attr-defined]
    mocker.patch.object(periodic, "_report_memo", {})  # type: ignore[attr-defined]
    parse = mocker.spy(periodic, "_parse")  # type: ignore[attr-defined]
    path = _msh22(tmp_path / "c.msh", BAD)
    first = check_periodic_facet_order(path)
    assert check_periodic_facet_order(path) is first  # same file: memo hit
    copy = tmp_path / "copy.msh"
    copy.write_bytes(path.read_bytes())
    other = check_periodic_facet_order(copy)  # same bytes, other path: content hit
    assert other.path == copy.resolve() and other.sha256 == first.sha256
    assert other.bad == first.bad
    assert parse.call_count == 1
    check_periodic_facet_order(path, use_cache=False)
    assert parse.call_count == 2
    # A rewritten file (new size / mtime) is re-hashed and re-checked.
    _msh22(path, GOOD)
    os.utime(path, ns=(1, 1))
    assert check_periodic_facet_order(path).ok
    assert parse.call_count == 3


class _FrozenStat:
    """A stat result whose ``st_mtime_ns`` never changes (coarse inode timestamps)."""

    def __init__(self, real: os.stat_result) -> None:
        self._real = real

    def __getattr__(self, name: str) -> object:
        return 1 if name == "st_mtime_ns" else getattr(self._real, name)


def test_writers_refresh_the_digest_memo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # mesh-3: an in-place same-size rewrite in one timestamp tick must not give the stale report
    from mpxlite.cells import periodic

    monkeypatch.setattr(periodic, "_digest_memo", {})
    real = Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, **kw: _FrozenStat(real(self, **kw)))
    path = _msh22(tmp_path / "f.msh", GOOD, slave_tris=[(2, 1, 0), (3, 2, 1)])
    size = path.stat().st_size
    assert check_periodic_facet_order(path).n_misoriented == 2
    report = align_periodic_facets(path)  # in place: same size, same (frozen) mtime
    assert path.stat().st_size == size
    assert report.ok and check_periodic_facet_order(path).ok
    # renumber_periodic in place, same size: the report is the new file's
    bad = _msh22(tmp_path / "r.msh", BAD)
    assert check_periodic_facet_order(bad).n_bad == 2
    assert renumber_periodic(bad, (1.0, 0.0), (0.0, 1.0)).ok
    assert check_periodic_facet_order(bad).ok


def test_misoriented_slave_facets_are_reported(tmp_path: Path) -> None:
    """Slave triangles stored with the opposite orientation (GetDP '7-Constraint Link')."""
    flipped = [(0, 2, 1), (1, 3, 2)]  # first triangle reversed, second as the master
    path = _msh22(tmp_path / "f.msh", GOOD, slave_tris=flipped)
    report = check_periodic_facet_order(path)
    assert report.n_bad == 0 and report.n_misoriented == 1 and not report.ok
    (m,) = report.misoriented
    assert m.nodes == (5, 7, 6) and m.image == (1, 3, 2) and m.master_nodes == (1, 2, 3)
    assert "1 stored with another orientation" in report.summary()


def test_align_repairs_misoriented_facets(tmp_path: Path) -> None:
    path = _msh22(tmp_path / "f.msh", GOOD, slave_tris=[(2, 1, 0), (3, 2, 1)])
    before = path.read_text(encoding="utf-8")
    out = tmp_path / "a.msh"
    report = align_periodic_facets(path, output=out)
    assert report.ok and report.n_facets == 2 and path.read_text(encoding="utf-8") == before
    text = out.read_text(encoding="utf-8")
    assert "2 2 302 2 5 6 7" in text and "2 2 302 2 6 8 7" in text
    # nodes and master elements untouched
    assert text.split("$Elements")[0] == before.split("$Elements")[0]
    assert "2 2 301 1 1 2 3" in text
    # renumber_periodic aligns too (rank AND orientation defects at once)
    both = _msh22(tmp_path / "b.msh", BAD, slave_tris=[(2, 1, 0), (3, 2, 1)])
    raw = check_periodic_facet_order(both)
    assert raw.n_bad == 2 and raw.n_misoriented == 2
    assert renumber_periodic(both, (1.0, 0.0), (0.0, 1.0), origin=(0.0, 0.0)).ok
    m41 = _msh41(tmp_path / "m41.msh", GOOD, curve_link=True)
    with pytest.raises(UnsupportedMeshFormatError, match=r"MSH 2\.x ASCII only"):
        align_periodic_facets(m41)


def test_msh41_refused_by_every_msh2_tool_with_the_conversion(tmp_path: Path) -> None:
    # mesh-m9: one refusal, naming the converter, for every tool that reads MSH 2.x only
    import numpy as np

    from mpxlite.cells import (
        LatticeOp,
        check_mesh_symmetry,
        check_symmetric_numbering,
        renumber_symmetric,
    )
    from mpxlite.cells.tiling import tile_periodic

    m41 = _msh41(tmp_path / "m41.msh", GOOD, curve_link=True)
    t = ((1.0, 0.0), (0.0, 1.0))
    ident = LatticeOp("E", np.eye(2), (0.0, 0.0))
    calls = {
        "align_periodic_facets": lambda: align_periodic_facets(m41),
        "renumber_periodic": lambda: renumber_periodic(m41, *t),
        "renumber_symmetric": lambda: renumber_symmetric(m41, t, [ident]),
        "check_symmetric_numbering": lambda: check_symmetric_numbering(m41, t, ident),
        "check_mesh_symmetry": lambda: check_mesh_symmetry(m41, t, [ident]),
        "tile_periodic": lambda: tile_periodic(m41, tmp_path / "x.msh", 1, 1, *t,
                                               bloch_tags=((301, 302), (303, 304))),
    }  # fmt: skip
    for name, call in calls.items():
        with pytest.raises(UnsupportedMeshFormatError, match=r"MSH 2\.x ASCII only .*msh22"):
            call()
        assert check_periodic_facet_order(m41).has_periodic, name  # the checks read 4.1


# ----- renumber ------------------------------------------------------------------


def test_renumber_repairs_reversed_mesh(tmp_path: Path) -> None:
    path = _msh22(tmp_path / "r.msh", BAD)
    out = tmp_path / "fixed.msh"
    before = path.read_bytes()
    report = renumber_periodic(path, (1.0, 0.0), (0.0, 1.0), output=out, origin=(0.0, 0.0))
    assert report.ok and report.n_facets == 2 and report.path == out.resolve()
    assert path.read_bytes() == before  # input untouched with output=
    # Periodic pairs still relate points translated by T_1, elements still conforming.
    text = out.read_text(encoding="utf-8").split("\n")
    i = text.index("$Nodes")
    coords = {
        int(t[0]): tuple(map(float, t[1:4]))
        for t in (line.split() for line in text[i + 2 : i + 2 + int(text[i + 1])])
    }
    j = text.index("$Periodic") + 5
    for line in text[j : j + 4]:
        a, b = map(int, line.split())
        assert coords[a] == pytest.approx((coords[b][0] + 1.0, *coords[b][1:]))
    # In place (default), any origin works.
    assert renumber_periodic(path, (1.0, 0.0), (0.0, 1.0)).ok
    assert check_periodic_facet_order(path).ok


def test_renumber_refuses(tmp_path: Path) -> None:
    m41 = _msh41(tmp_path / "m41.msh", BAD, curve_link=True)
    with pytest.raises(UnsupportedMeshFormatError, match=r"MSH 2\.x ASCII only"):
        renumber_periodic(m41, (1.0, 0.0), (0.0, 1.0))
    m22 = _msh22(tmp_path / "m22.msh", BAD)
    with pytest.raises(ValueError, match="not independent"):
        renumber_periodic(m22, (1.0, 0.0), (2.0, 0.0))
    nd = tmp_path / "nd.msh"
    nd.write_text(m22.read_text(encoding="utf-8") + "$NodeData\n$EndNodeData\n", "utf-8")
    with pytest.raises(UnsupportedMeshFormatError, match="NodeData"):
        renumber_periodic(nd, (1.0, 0.0), (0.0, 1.0))


def test_renumber_general_lattice_vector(tmp_path: Path) -> None:
    """T_1 not along x (general 2 x 2 inverse of the lattice coordinates)."""
    path = _msh22(tmp_path / "r.msh", BAD)
    # T_1 = (1, 0) expressed with a rotated companion vector: the lattice is the
    # same set of translations as far as the x-link is concerned.
    assert renumber_periodic(path, (1.0, 0.0), (1.0, 1.0), origin=(0.0, 0.0)).ok
    assert renumber_periodic(path, (1.0, 1.0), (1.0, 0.0), origin=(0.0, 0.0)).ok
