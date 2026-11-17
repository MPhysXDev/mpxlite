// Copyright (c) 2025 MPhysX
// SPDX-License-Identifier: MIT
// rod: -u'' = S on Domain, u = 0 on Left, u = V on Right (P1 Lagrange).
// Exact solution u(x) = V x + S x (1 - x) / 2 (nodal values exact in P1).
// source[] and u_right[] are provided later by rod.physprop.pro and rod.cond.pro
// (a Constraint must be defined before the FunctionSpace that references it,
// a Function only declared).
Function { DefineFunction[ source, u_right ]; }
Constraint {
  { Name Dirichlet; Type Assign;
    Case {
      { Region Left; Value 0.; }
      { Region Right; Value u_right[]; }
    }
  }
}

Jacobian {
  { Name JVol; Case { { Region All; Jacobian Vol; } } }
}
Integration {
  { Name I1; Case { { Type Gauss; Case { { GeoElement Line; NumberOfPoints 2; } } } } }
}
FunctionSpace {
  { Name Hgrad; Type Form0;
    BasisFunction { { Name sn; NameOfCoef un; Function BF_Node;
                      Support Domain; Entity NodesOf[All]; } }
    Constraint { { NameOfCoef un; EntityType NodesOf; NameOfConstraint Dirichlet; } }
  }
}
Formulation {
  { Name Poisson; Type FemEquation;
    Quantity { { Name u; Type Local; NameOfSpace Hgrad; } }
    Equation {
      Galerkin { [ Dof{d u}, {d u} ]; In Domain; Jacobian JVol; Integration I1; }
      Galerkin { [ -source[], {u} ]; In Domain; Jacobian JVol; Integration I1; }
    }
  }
}
Resolution {
  { Name Static;
    System { { Name A; NameOfFormulation Poisson; } }
    Operation { Generate[A]; Solve[A]; SaveSolution[A]; }
  }
  // Solves nothing: takes the solution of the file given with -res.
  { Name Reread;
    System { { Name A; NameOfFormulation Poisson; } }
    Operation { InitSolution[A]; ReadSolution[A]; SaveSolution[A]; }
  }
}
PostProcessing {
  { Name Post; NameOfFormulation Poisson;
    Quantity {
      { Name u; Value { Local { [ {u} ]; In Domain; Jacobian JVol; } } }
      { Name intu; Value { Integral { [ {u} ]; In Domain; Jacobian JVol; Integration I1; } } }
    }
  }
}
PostOperation {
  { Name mid; NameOfPostProcessing Post;
    Operation { Print[ u, OnPoint {0.5, 0, 0}, Format Table, File "u_mid.dat" ]; }
  }
  { Name integral; NameOfPostProcessing Post;
    Operation { Print[ intu[Domain], OnGlobal, Format Table, File "u_int.dat" ]; }
  }
}
