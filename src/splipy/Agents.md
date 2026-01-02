# Splipy Agent Guidelines

This document outlines the design principles, architecture, and objectives of the Splipy library to help AI agents make informed decisions when contributing code, fixing bugs, or extending functionality.

## Project Overview

**Splipy** is a pure Python library for the creation, evaluation, and manipulation of B-spline and NURBS geometries. It is designed for:
- **Mathematical accuracy**: Precise B-spline mathematics with careful numerical handling
- **Generality**: Support for n-variate splines of arbitrary dimension
- **Fine-grained control**: Emphasis on curves, surfaces, and volumes with detailed parameter control
- **Analysis use**: Intended primarily for computational analysis and research, not CAD production workflows

## Core Design Principles

### 1. **Mathematical Correctness First**
- All spline operations must be mathematically rigorous and numerically stable
- B-spline basis functions, knot vector handling, and derivative calculations must follow standard mathematical definitions (de Boor, Cox-de Boor algorithm)
- Numerical precision is critical—test against analytical results and convergence behavior
- When fixing bugs, verify fixes with multiple test cases and mathematical validation

### 2. **Class Hierarchy: SplineObject-First Architecture**
The library uses a well-defined inheritance structure:
```
SplineObject (base class)
├── Curve (1D parametric space)
├── Surface (2D parametric space)
├── Volume (3D parametric space)
```

**Key Principle**: Code shared across spline types goes in `SplineObject` (base methods). Dimension-specific logic goes in subclasses.

Common attributes shared by all splines:
- `bases`: List of `BSplineBasis` objects (one per parametric dimension)
- `controlpoints`: NumPy array of control point coordinates
- `dimension`: Physical space dimension (2D, 3D, etc.)
- `rational`: Boolean indicating if this is a NURBS (rational) spline

### 3. **NumPy Vectorization**
- Operations should be vectorized using NumPy for performance
- Avoid Python loops where NumPy operations suffice
- Use `np.atleast_1d()`, `np.asarray()`, and broadcasting for flexible input handling
- Single evaluation points should squeeze output dimensions; multiple points should return arrays with shape `(n_points, dimension)`

### 4. **B-Spline Basis as Separate Entity**
The `BSplineBasis` class encapsulates:
- Knot vector management
- Basis function evaluation (sparse matrix form via `evaluate()`)
- Knot insertion and refinement
- Spline order and continuity information
- Information on periodicity of the basis functions (i.e. if evaluation points wrap around outside their domain)

**Principle**: Basis and control points are separate. Manipulating basis (e.g., raising order, inserting knots) doesn't directly modify the spline geometry.

### 5. **Rational (NURBS) Support**
- Rational control point points are stored as "projective controlpoints", i.e. they are already multiplied by their weights: (xw,yw,zw,w) for all points, with weights as the last coordinate
- The `rational` flag indicates NURBS status
- All operations must handle both rational and non-rational cases
- Direct evaluation then weights affect evaluation via division in the final step
- Derivatives require chain-rule evaluation to get the right answer

### 6. **Type Safety and Clarity**
- Use type hints extensively (Python 3.12+)
- Import from `.typing` module for custom types: `ArrayLike`, `FloatArray`, `Scalar`, `Direction`
- Validate input types and shapes; raise `ValueError` or `RuntimeError` with clear messages
- Use overload signatures for methods accepting multiple input types

## Key Objectives

### A. **Accurate Spline Evaluation and Derivatives**
- `evaluate()`: Compute geometric points on the spline
- `derivative()`: Compute tangent, normal, curvature (with correct derivatives)
- **Critical**: Derivatives must account for basis derivatives, not just control point differences
- For curves: use `dr/dt = basis_derivatives · controlpoints`
- **Critical**: Derivatives must account for chain rule when spline is rational

### B. **Knot Vector and Basis Manipulation**
- Insert knots without changing geometry (exact geometric preservation)
- Raise/lower spline order (p-elevation/reduction)
- Refine the spline through knot insertion or reparametrization
- Split splines at arbitrary parameter values
- Make splines periodic (wrap boundary conditions)

### C. **Numerical Integration for Analysis**
- `length()`, `area()`, `volume()`: Integrate geometric quantities using gaussian quadrature

### D. **File I/O**
Modules in `io/` handle various formats:
- `.g2` (GoTools): Native spline format, see https://github.com/SINTEF-Geometry/GoTools. Read/Write without loss
- `.spl`: Alternative spline format
- `.svg`: Scalable Vector Graphics, see https://en.wikipedia.org/wiki/SVG. Limiting read, write without loss of accuracy (only 2D)
- `.3dm`: Rhino 3D format, see https://github.com/mcneel/rhino3dm. Read/Write without loss
- `.stl`: Mesh export to stereolithography (tessellation-based), Only write. With loss. Specify accuracy
- `.grdecl`: Reservoir simulation grids, Only write. With loss.

**Principle**: I/O should round-trip without loss of geometric fidelity whenever possible

### E. **Utilities and Extensions**
Modules in `utils/` provide:
- `refinement.py`: B-spline refinement strategies
- `smooth.py`: Smoothing and curve fitting
- `nutils.py`: Integration with Nutils FEM library
- `image.py`: Image-based spline generation
- `curve.py`: Curve-specific utilities (arc-length parametrization)

## Common Bug Patterns and How to Fix Them

### 1. **Quadrature and Numerical Integration Issues**
**Pattern**: Fixed quadrature rules fail for high-order or extreme control points
- **Fix Strategy**: Use adaptive or per-knot-span quadrature; increase quadrature points for high-order basis
- **Validation**: Test with analytical curves (circles, ellipses) and convergence studies
- **Example**: `curve.length()` should use `order(0) + 1` points per knot span, not globally

### 2. **Parameter Shadowing in List Comprehensions**
**Pattern**: Function parameters with the same name as loop variables cause confusion
- **Fix**: Use distinct variable names or refactor to avoid shadowing
- **Example**: In `length(t0, t1)`, the loop `for t0, t1 in zip(...)` shadows the parameters

### 3. **Incorrect Derivative Handling**
**Pattern**: Using finite differences instead of analytical basis derivatives
- **Fix**: Always use `BSplineBasis.evaluate_deriv()` or compute control point derivatives correctly
- **Validation**: Compare against finite differences as a check, not a replacement

### 4. **Rational Spline Division Not Applied**
**Pattern**: Forgetting to divide by the weight (last coordinate) after evaluation
- **Fix**: Always check `if self.rational:` and perform weight division on result

### 5. **Knot Insertion Not Preserving Geometry**
**Pattern**: Naive knot insertion creates spurious changes
- **Fix**: Use the Oslo algorithm or de Boor's method to insert knots without geometry change

## Testing Philosophy

### Unit Tests (`tests/`)
- **Location**: `tests/curve_test.py`, `surface_test.py`, `volume_test.py`, etc.
- **Pattern**: Test single operations (evaluate, derivative, length, etc.)
- **Coverage**: Include edge cases: empty knots, single control point, high order, rational splines
- **AI modifictation**: The correct place for any new Copilot generated test cases.

### Generated Tests (`tests/generated/`)
- **Purpose**: Test complex operations (knot insertion, raise_order, rebuild) via code generation
- **Strategy**: Generate test data from reference implementations, validate against stored outputs
- **Example**: `generate_knot_insert.py` produces test cases for knot insertion
- **AI modifictation**: These tests are *not* to be altered by any Copilot suggestion.

### Benchmarks (`tests/benchmark_test.py`)
- **Purpose**: Track performance regressions
- **Pattern**: Use pytest-benchmark for consistent timing
- **When to Run**: Before and after performance-sensitive changes
- **AI modifictation**: When considering optimization changes, then copilot can make changes to these tests.

### Validation Against External References
- Compare geometry evaluation against analytical solutions (circles: $x^2 + y^2 = r^2$)
- Use finite differences as a sanity check for derivatives (not a replacement)
- Verify arc length with high-resolution approximations (many intervals with straight-line distances)

## Code Organization

### Main Classes
- `curve.py`: 1D splines with curve-specific methods (tangent, normal, curvature, torsion, arc length)
- `surface.py`: 2D splines with surface metrics (normal, curvature, area)
- `volume.py`: 3D splines (typically used in analysis)
- `trimmedsurface.py`: Dummy function, not maintained
- `splineobject.py`: Base class with shared functionality

### Factories
- `curve_factory.py`, `surface_factory.py`, `volume_factory.py`: Constructors for standard geometries
- **Pattern**: Named functions (e.g., `circle()`, `sphere()`, `bezier()`) return spline objects

### Basis
- `basis.py`: `BSplineBasis` class managing knot vectors and basis functions
- **Key Methods**: `evaluate()`, `evaluate_deriv()`, `insert_knot()`, `raise_order()`

### Utilities and I/O
- `io/`: Format readers/writers
- `utils/`: Refinement, smoothing, fitting, image processing

## Decision-Making Guide for Agents

### When Adding a New Feature
1. **Does it belong in `SplineObject`?** (shared across all spline types) → Base class
2. **Is it curve-specific?** (tangent, arc length) → `Curve` subclass
3. **Does it require file I/O?** → Create or extend a module in `io/`
4. **Is it a utility?** (fitting, refinement) → Create or extend a module in `utils/`
5. **Is it a constructor?** (build standard geometry) → Add to appropriate factory

### When Fixing a Bug
1. **Create a minimal reproducible test case** based on the bug report
2. **Verify the current behavior** with the test
3. **Check for mathematical correctness** (not just passing tests)
4. **Test edge cases**: empty inputs, high order, rational splines, periodic splines
5. **Validate numerically** against analytical solutions or high-precision approximations
6. **Document the fix** in test comments explaining why the old code failed

### When Optimizing Performance
1. **Profile first** using `tests/benchmark_test.py` or Python's `cProfile`
2. **Preserve correctness** — no numerical regressions
3. **Favor NumPy vectorization** over loops
4. **Avoid micro-optimizations** unless they significantly impact real-world usage
5. **Test on varied inputs** (small and large splines, many dimensions)

### When Choosing Between Multiple Approaches
- **Mathematical rigor** > Code simplicity
- **Numerical stability** > Raw speed
- **Generality** (n-variate, n-dimensional) > Special cases
- **Vectorization** > Readable loops (but prefer readable NumPy)
- **Test coverage** > Assumed correctness

## Known Limitations and Constraints

1. **Python 3.12+**: Type hints and modern Python features required
2. **Pure Python** (with C extension for core math via `splipy-core` package)
3. **NumPy-based**: Heavy reliance on NumPy for vectorization and linear algebra
4. **No GPU acceleration**: CPU-only (NumPy, SciPy)
5. **Curves are 1D, Surfaces 2D, Volumes 3D** in parameter space (not physical space)
6. **Periodic splines** have special knot vector structure; ensure methods handle them

## Recommended References

- **B-Splines**: "The NURBS Book" by Piegl & Tiller (standard reference)
- **Spline theory**: `theory/spline_methods_theory_from_university_of_oslo.pdf` 
- **Algorithms**: De Boor's algorithm for knot insertion, Cox-de Boor recursion for basis evaluation
- **Numerical Integration**: Gaussian quadrature for smooth functions (curves, arc length)
- **Existing Code**: Study `splineobject.py`, `basis.py`, and `curve.py` for patterns

## Summary for AI Agents

**When working on Splipy:**
- **Prioritize mathematical correctness** and numerical stability over everything else
- **Use the class hierarchy**: SplineObject for shared code, subclasses for specifics
- **Vectorize with NumPy**: Avoid loops, leverage broadcasting
- **Test thoroughly**: Unit tests, convergence checks, edge cases
- **Follow existing patterns**: Basis/controlpoints separation, input validation, type hints
- **Document assumptions**: Why quadrature order is chosen, why a certain algorithm is used
- **Validate against references**: Analytical solutions, high-precision approximations, published test cases

Good luck improving Splipy! 🎯
