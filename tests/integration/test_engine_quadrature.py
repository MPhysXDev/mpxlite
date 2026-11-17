# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The tabulated Gauss rules of the engine integrate what they must.

One engine run integrates, with every tabulated rule of lines, triangles,
quadrangles, tetrahedra and hexahedra, a polynomial that contains every
monomial up to the degree of the rule, over one element of that type, and
the results are compared with the exact integrals. A wrong entry in a table
of the engine (two were found on quadrangles in GetDP 3.5.0 and are corrected
by the engine patch) shows as a relative error of a few percent or more.

The same run integrates with the ``GaussLegendre`` type in an ``Integral``
post-quantity, which GetDP 3.5.0 refuses and the engine patch accepts.

Not covered: prisms and pyramids.
"""

from __future__ import annotations

import itertools
import math

import pytest

from mpxlite import SolverLite, parse_real_table

# The polynomial integrated is (1 + A x + B y + C z) ** degree.
A, B, C = 0.7, 1.3, 0.9

# Degree up to which each tabulated rule is exact: number of points -> degree.
DEGREES: dict[str, dict[int, int]] = {
    # Gauss-Legendre: 2 n - 1; checked up to degree 15.
    "Line": {n: min(2 * n - 1, 15) for n in range(1, 21)},
    "Triangle": {1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 12: 6, 13: 7, 16: 8},
    "Quadrangle": {1: 1, 3: 2, 4: 3, 7: 5},
    "Tetrahedron": {1: 1, 4: 2, 5: 3, 15: 5, 16: 4, 17: 5, 29: 6},
    "Hexahedron": {6: 3, 14: 5, 34: 7, 77: 9},
}
# Tensor Gauss-Legendre rules in an Integral post-quantity: n x n points on
# the quadrangle, exact up to degree 2 n - 1 in each variable.
LEGENDRE: dict[int, int] = {4: 3, 9: 5, 16: 7}

# One element of each type: region number, dimension, Jacobian of GetDP,
# MSH 2.2 element type, nodes. Simplices are the unit simplices, the others
# the unit segment, square and cube.
ELEMENTS: dict[str, tuple[int, int, str, int, list[tuple[float, float, float]]]] = {
    "Line": (1, 1, "Lin", 1, [(0, 0, 0), (1, 0, 0)]),
    "Triangle": (2, 2, "Sur", 2, [(0, 0, 0), (1, 0, 0), (0, 1, 0)]),
    "Quadrangle": (3, 2, "Sur", 3, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
    "Tetrahedron": (4, 3, "Vol", 4, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    "Hexahedron": (
        5,
        3,
        "Vol",
        5,
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
    ),
}
SIMPLICES = ("Triangle", "Tetrahedron")


def _monomial_integral(element: str, powers: tuple[int, ...]) -> float:
    """Exact integral of ``x^p y^q z^r`` over the element of the mesh."""
    dimension = ELEMENTS[element][1]
    if any(powers[dimension:]):
        return 0.0  # the element lies in the plane or on the axis
    powers = powers[:dimension]
    if element in SIMPLICES:
        numerator = math.prod(math.factorial(p) for p in powers)
        return numerator / math.factorial(sum(powers) + dimension)
    return math.prod(1.0 / (p + 1) for p in powers)


def exact_integral(element: str, degree: int) -> float:
    """Exact integral of ``(1 + A x + B y + C z) ** degree`` over the element."""
    total = 0.0
    for p, q, r in itertools.product(range(degree + 1), repeat=3):
        if p + q + r > degree:
            continue
        coefficient = math.factorial(degree) / (
            math.factorial(p)
            * math.factorial(q)
            * math.factorial(r)
            * math.factorial(degree - p - q - r)
        )
        total += coefficient * A**p * B**q * C**r * _monomial_integral(element, (p, q, r))
    return total


def _mesh() -> str:
    nodes: list[tuple[float, float, float]] = []
    elements: list[str] = []
    for number, (region, _, _, kind, points) in enumerate(ELEMENTS.values(), start=1):
        first = len(nodes) + 1
        nodes.extend(points)
        tags = " ".join(str(first + i) for i in range(len(points)))
        elements.append(f"{number} {kind} 2 {region} {region} {tags}")
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    lines += [f"{i} {x} {y} {z}" for i, (x, y, z) in enumerate(nodes, start=1)]
    lines += ["$EndNodes", "$Elements", str(len(elements)), *elements, "$EndElements"]
    return "\n".join(lines) + "\n"


def _cases() -> list[tuple[str, str, str, int, int]]:
    """(name, element, integration type, number of points, degree) of every integral."""
    cases = [
        (f"{element}_{points}", element, "Gauss", points, degree)
        for element, rules in DEGREES.items()
        for points, degree in rules.items()
    ]
    cases += [
        (f"Legendre_{points}", "Quadrangle", "GaussLegendre", points, degree)
        for points, degree in LEGENDRE.items()
    ]
    return cases


def _formulation() -> str:
    """Problem definition: a trivial system on the tetrahedron, and the integrals."""
    jacobians = "\n".join(
        f"      {{ Region {element}; Jacobian {jacobian}; }}"
        for element, (_, _, jacobian, _, _) in ELEMENTS.items()
    )
    integrations = "\n".join(
        f"  {{ Name I_{name}; Case {{ {{ Type {kind}; Case {{ "
        f"{{ GeoElement {element}; NumberOfPoints {points}; }} }} }} }} }}"
        for name, element, kind, points, _ in _cases()
    )
    quantities = "\n".join(
        f"      {{ Name q_{name}; Value {{ Integral {{ [ poly[]^{degree} ]; In {element}; "
        f"Jacobian J; Integration I_{name}; }} }} }}"
        for name, element, _, _, degree in _cases()
    )
    prints = "\n".join(
        f'      Print[ q_{name}[{element}], OnGlobal, Format Table, File "q_{name}.dat" ];'
        for name, element, _, _, _ in _cases()
    )
    return f"""\
Function {{ poly[] = 1 + {A} * X[] + {B} * Y[] + {C} * Z[]; }}
Jacobian {{
  {{ Name J; Case {{
{jacobians}
  }} }}
}}
Integration {{
  {{ Name I0; Case {{ {{ Type Gauss; Case {{
    {{ GeoElement Tetrahedron; NumberOfPoints 4; }} }} }} }} }}
{integrations}
}}
FunctionSpace {{
  {{ Name H; Type Form0;
    BasisFunction {{ {{ Name sn; NameOfCoef un; Function BF_Node;
                      Support Tetrahedron; Entity NodesOf[All]; }} }}
  }}
}}
Formulation {{
  {{ Name Mass; Type FemEquation;
    Quantity {{ {{ Name u; Type Local; NameOfSpace H; }} }}
    Equation {{
      Galerkin {{ [ Dof{{u}}, {{u}} ]; In Tetrahedron; Jacobian J; Integration I0; }}
      Galerkin {{ [ -1, {{u}} ]; In Tetrahedron; Jacobian J; Integration I0; }}
    }}
  }}
}}
Resolution {{
  {{ Name Static;
    System {{ {{ Name S; NameOfFormulation Mass; }} }}
    Operation {{ Generate[S]; Solve[S]; SaveSolution[S]; }}
  }}
}}
PostProcessing {{
  {{ Name Post; NameOfFormulation Mass;
    Quantity {{
{quantities}
    }}
  }}
}}
PostOperation {{
  {{ Name integrals; NameOfPostProcessing Post;
    Operation {{
{prints}
    }}
  }}
}}
"""


@pytest.fixture(scope="module")
def integrals(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    """Every integral, computed by one run of the engine."""
    case = tmp_path_factory.mktemp("quadrature")
    (case / "rules.msh").write_text(_mesh(), encoding="utf-8")
    groups = "\n".join(
        f"  {element} = Region[{region}];" for element, (region, *_) in ELEMENTS.items()
    )
    (case / "rules.geo.pro").write_text(f"Group {{\n{groups}\n}}\n", encoding="utf-8")
    (case / "rules.physprop.pro").write_text("", encoding="utf-8")
    (case / "rules.cond.pro").write_text("", encoding="utf-8")
    (case / "rules.formulation.pro").write_text(_formulation(), encoding="utf-8")
    solver = SolverLite(
        case,
        "rules",
        "rules.msh",
        "Static",
        resource_files=[case / "rules.formulation.pro"],
        runs_dir=case / "runs",
    )
    result = solver.run("integrals")
    return {
        name: float(parse_real_table(result.path(f"q_{name}.dat"))[-1, -1]) for name, *_ in _cases()
    }


@pytest.mark.parametrize(
    ("name", "element", "degree"),
    [(name, element, degree) for name, element, _, _, degree in _cases()],
    ids=[name for name, *_ in _cases()],
)
def test_rule_is_exact_to_its_degree(
    integrals: dict[str, float], name: str, element: str, degree: int
) -> None:
    """The rule integrates a full polynomial of its degree exactly."""
    assert integrals[name] == pytest.approx(exact_integral(element, degree), rel=1e-7)


def test_the_check_detects_a_rule_one_degree_short() -> None:
    """The polynomial of degree d + 1 differs from what a rule of degree d gives.

    Guards the test itself: with the 4-point rule of the quadrangle (degree
    3), computed here, the polynomial of degree 4 is not integrated exactly.
    """
    point = 0.5 / math.sqrt(3.0)
    nodes = [(0.5 + sx * point, 0.5 + sy * point) for sx in (-1, 1) for sy in (-1, 1)]

    def rule(degree: int) -> float:
        return sum(0.25 * (1 + A * x + B * y) ** degree for x, y in nodes)

    assert rule(3) == pytest.approx(exact_integral("Quadrangle", 3), rel=1e-12)
    assert rule(4) != pytest.approx(exact_integral("Quadrangle", 4), rel=1e-4)
