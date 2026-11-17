// Copyright (c) 2025 MPhysX
// SPDX-License-Identifier: MIT
// 1-D unit segment [0, 1], 10 elements; regenerate with
//   gmsh -1 -format msh22 rod.geo -o rod.msh
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Line(1) = {1, 2};
Transfinite Curve{1} = 11;
Physical Point(1) = {1};
Physical Point(2) = {2};
Physical Curve(10) = {1};
