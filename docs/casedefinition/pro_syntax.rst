.. _mpxlite_pro_syntax:

GetDP ``.pro`` syntax (mini-reference)
======================================

The user case files (``<project>.geo.pro``, ``.physprop.pro``,
``.cond.pro``) and the plug-in resource files are all written in the
`GetDP problem-definition language
<http://getdp.info/dev/doc/texinfo/getdp.html>`_. GetDP's full
language is rich; this page documents the **subset** that surfaces
in user case files and that ``mpxlite`` therefore expects authors to
be familiar with.

For exhaustive coverage of the language --- including formulations,
function spaces, resolutions and post-operations, which live in the
plug-in's resource files and not in user case files --- refer to the
official GetDP reference manual.

General lexical structure
-------------------------

* The language is **case-sensitive**. ``Region``, ``region`` and
  ``REGION`` are three different tokens.
* Statements end with a semicolon ``;``.
* Whitespace is insignificant outside string literals.
* Comments use C-style ``/* ... */`` (multi-line) or C++ ``//``
  (to end-of-line).
* String literals are double-quoted: ``"hello"``.

Includes
--------

The ``Include`` directive splices the contents of another file at
the current point:

.. code-block:: c

    Include "some_other_file.pro";

`mpxlite` builds the auto-generated manifest ``mpxlite.pro``
exclusively out of ``Include`` directives; the user does not write
them by hand.

Numerical constants and expressions
-----------------------------------

Pure-constant expressions are computed at parse time and behave like
ordinary C-language constants:

.. code-block:: c

    c1 = 1.0;
    c2 = Log[2.0] + c1 + 3.0;

The constant :math:`\pi` is available as the predefined identifier
``Pi``.

Operators (excerpt)
+++++++++++++++++++

The arithmetic, comparison and logical operators follow the C
convention:

==============  =========================================================
Operator        Meaning
==============  =========================================================
``+``, ``-``    addition, subtraction (also unary minus)
``*``, ``/``    multiplication / scalar product, division
``%``           modulo (right-hand side scalar)
``^``           exponentiation (scalar arguments)
``/\``          cross product (vector arguments)
``==``, ``!=``  equality, inequality
``<``, ``>``,
``<=``, ``>=``  ordering (scalar arguments)
``&&``, ``||``  logical AND / OR (always evaluates both operands)
``!``           logical NOT
``( ... )``     grouping
==============  =========================================================

Mathematical functions
++++++++++++++++++++++

The standard C-math primitives are available with the GetDP
capitalisation: ``Exp``, ``Log``, ``Log10``, ``Sqrt``, ``Sin``,
``Cos``, ``Tan``, ``Asin``, ``Acos``, ``Atan``, ``Atan2``, ``Sinh``,
``Cosh``, ``Tanh``, ``Fabs``, ``Fmod``. All take square-bracketed
arguments: ``Cos[Pi/4]``, not ``Cos(Pi/4)``.

Vector and tensor utilities used in case files include ``Norm``,
``SquNorm``, ``Unit``, ``Transpose``, ``Vector``, ``TensorDiag``,
``TensorSym``, ``Complex``.

Coordinate access:

* ``XYZ[]`` --- the position vector at the integration point.
* ``X[]``, ``Y[]``, ``Z[]`` --- its three Cartesian components.

Loops and conditionals
++++++++++++++++++++++

GetDP supports a small parse-time control flow useful for
parametric expansion:

.. code-block:: c

    For i In {0:N - 1}
      // body, with i bound to the loop variable
    EndFor

    If (Meta_FE_OI == 2)
      // conditional body
    EndIf

These constructs are evaluated at parse time, not at solve time. They
appear in plug-in resource files (e.g. to expand the per-Fourier-order
post-processing tables) and are documented here because users who
read the resource files will encounter them.

The ``Group`` block --- topology
--------------------------------

The ``Group`` block declares **regions**: named subsets of mesh
entities, identified by their physical tags or by composition of
already-declared regions.

In the user case file ``<project>.geo.pro``, the typical job is to
populate the platform topology classes by mapping mesh tags:

.. code-block:: c

    Group {
        // Bind a single tag.
        Vol_substrate = Region[2];

        // Bind a list of tags.
        MetaPMLBot    = Region[{1}];
        MetaVol       = Region[{2, 3}];

        // Compose by name.
        MetaSurfFour  = Region[{MetaSurfTop, MetaSurfBot}];

        // Empty placeholder for an optional class.
        DefineGroup[MetaPEC];
    }

A few practical notes:

* ``Region[N]`` and ``Region[{N}]`` are equivalent for a single tag;
  the brace form is preferred for lists.
* A class declared with ``DefineGroup[Name]`` exists but is empty;
  this is the right way to satisfy a plug-in that expects a class
  but does not require it to be populated for every case.
* Names follow the GetDP identifier syntax
  ``[A-Za-z_][A-Za-z_0-9]*``.

The ``Function`` block --- physical properties
----------------------------------------------

The ``Function`` block declares **named expressions** that the plug-in
formulation references by name. Two flavours are useful in case files:

**Globally defined.** The function takes no region argument and is
the same throughout space:

.. code-block:: c

    Function {
        my_constant[]      = 1.5;
        my_vector[]        = Vector[1., 0., 0.];
        my_tensor[]        = TensorDiag[1., 1., 2.];
    }

**Piecewise defined per region.** The function takes a ``[Region]``
argument and may have different values on different regions:

.. code-block:: text

    Function {
        Meta_epsr[Vol_substrate]
            = TensorDiag[2.25, 2.25, 2.25];
        Meta_epsr[#{MetaVol, MetaPMLBot, MetaPMLTop}]
            = TensorDiag[1., 1., 1.];
    }

The ``#{R1, R2, ...}`` syntax matches a union of regions in a single
clause; equivalent to writing one clause per member region.

Pre-declared functions
++++++++++++++++++++++

Plug-ins typically forward-declare the names they expect the user to
populate via ``DefineFunction``:

.. code-block:: c

    Function {
        DefineFunction[Meta_epsr];     // user is expected to define this
        DefineFunction[Meta_mur];      // ditto
    }

After this, the case ``<project>.physprop.pro`` provides the
piecewise definitions documented in the plug-in chapter.

Complex numbers
+++++++++++++++

GetDP exposes complex-valued expressions through the ``Complex``
constructor:

.. code-block:: c

    z[]      = Complex[1.0, -0.5];           // 1 - 0.5j
    eps_lossy[Vol_metal]
             = TensorDiag[Complex[1., -1.], Complex[1., -1.], 1.];

The imaginary unit can be re-exported by the plug-in as
``Meta_J[] = Complex[0., 1.];`` for readability.

Tensors
+++++++

Three constructors cover the common cases:

* ``TensorDiag[a, b, c]`` --- diagonal tensor with the three given
  diagonal entries.
* ``TensorSym[xx, yy, zz, xy, yz, xz]`` --- symmetric tensor with
  the six independent components.
* ``Tensor[xx, xy, xz, yx, yy, yz, zx, zy, zz]`` --- generic 3x3
  tensor.

The ``Constraint`` block --- conditions
---------------------------------------

The ``Constraint`` block declares boundary conditions and
inter-region links. The most common kinds are:

**Dirichlet (essential)** --- prescribe a value on a region:

.. code-block:: c

    Constraint {
        { Name MyDirichlet;
            Case {
                { Region MySurf; Value 0.; }
            }
        }
    }

**Periodic linking with phase shift (LinkCplx)** --- relate the
tangential / normal traces of two paired faces by a complex
multiplicative coefficient. This is how plug-ins implement Bloch
periodicity:

.. code-block:: c

    Constraint {
        { Name MyBlochX;
            Case {
                { Region MyFacePlus;
                  Type LinkCplx;
                  RegionRef MyFaceMinus;
                  Coefficient deph_X[];
                  Function Vector[$X - PERIOD_X, $Y, $Z];
                }
            }
        }
    }

The ``Function`` here is the **mesh-to-mesh translation map** between
the two faces; the corresponding mesh ``Periodic`` directive must
already be present in the ``.geo`` (see :ref:`mpxlite_meshgeneration`).

In a typical ``mpxlite`` deployment the standard constraints (Bloch,
Dirichlet on outer walls) live in the **plug-in resource files**; the
case-side ``<project>.cond.pro`` is therefore often empty:

.. code-block:: c

    Constraint {
    }

It exists so that case-specific extra constraints can be added
without forking the plug-in.

What is *not* on this page
--------------------------

The following GetDP language constructs do appear in plug-in resource
files but **do not** belong in user case files; therefore they are
out of scope for this mini-reference:

* ``FunctionSpace`` --- the FE function spaces.
* ``Formulation`` --- the variational equation, in terms of test
  functions and DOFs.
* ``Resolution`` --- the imperative solve script (``Generate``,
  ``Solve``, ``SaveSolution``).
* ``Jacobian`` and ``Integration`` rules.
* ``PostProcessing`` and ``PostOperation`` --- the output pipeline.

Refer to the upstream GetDP manual for the full grammar of these
constructs. From the user's perspective, the plug-in handles them and
the case files only need to populate the ``Group``, ``Function`` and
``Constraint`` blocks documented above.
