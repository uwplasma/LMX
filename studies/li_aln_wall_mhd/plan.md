# LMX Study Plan: Parametric MHD Assessment of AlN/Metal Wall Concepts for Flowing Liquid Lithium

**Prepared for:** LMX developer / simulation owner
**Purpose:** Define a complete, reproducible LMX simulation campaign to evaluate the MHD performance of Aluminum Nitride (AlN)-based insulating wall concepts for liquid-lithium flow.
**Primary output:** A ranked MHD-performance assessment of candidate wall electrical models, not a materials-qualification result.

---

## 1. Executive Summary

A collaborator asked whether Aluminum Nitride (AlN), which appears promising in flowing lithium because it may resist disintegration, can be matched to a lithium-compatible metal and printed. LMX can contribute to this question only within a specific scope.

LMX should be used for a **parametric inductionless MHD sensitivity study**. The study should determine whether an AlN-like insulating layer meaningfully reduces MHD current closure, Lorentz drag, wall-current leakage, and required pressure-gradient/forcing in liquid-lithium flow under prescribed magnetic fields.

LMX should **not** be used to certify AlN/lithium compatibility, corrosion resistance, coating adhesion, wetting behavior, thermal-cycling survival, irradiation resistance, or additive-manufacturing printability. Those require separate material testing and process qualification.

The final result should answer:

> For a chosen lithium duct, magnetic field, and flow condition, how good must the AlN electrical insulation be for the MHD behavior to approach the ideal-insulator limit, and how much performance is lost if the AlN becomes conductive, pinholed, cracked, or effectively bypassed by a metal substrate?

The study should produce:

1. a reproducible LMX simulation package;
2. validated input decks;
3. scalar result tables;
4. field-output files;
5. plots suitable for a technical report;
6. autodifferentiation-based sensitivity results;
7. code additions if required for effective-wall, pinhole, and true multilayer AlN/metal wall models;
8. a final MHD ranking of wall-stack electrical models.

---

## 2. Decision Boundary

### 2.1 What LMX can answer

LMX can answer MHD performance questions such as:

- Does an electrically insulating AlN-like wall reduce MHD current closure compared with a bare metal wall?
- How much does the required forcing or pressure-drop proxy change with wall conductance ratio?
- How sensitive are Lorentz power, peak current density, and wall leakage to degraded AlN conductivity?
- How much pinhole or shunt conductance can be tolerated before the AlN no longer behaves like an insulator?
- Does a metal substrate matter electrically when AlN is intact, degraded, or bypassed?
- What are the gradients of MHD objectives with respect to conductance, magnetic field, flow speed, lithium conductivity, AlN conductivity, and AlN thickness?

### 2.2 What LMX cannot answer

Do not claim that this study proves any of the following:

- AlN chemical compatibility with lithium;
- corrosion resistance;
- dissolution resistance;
- lithium wetting or contact-angle behavior;
- AlN/metal bonding strength;
- coating adhesion;
- thermal-expansion mismatch survival;
- crack formation or delamination;
- additive-manufacturing feasibility;
- laser powder-bed fusion or directed-energy-deposition process quality;
- residual stress;
- AM porosity or lack-of-fusion defects;
- neutron irradiation effects;
- tritium permeation;
- free-surface lithium behavior;
- MHD turbulence qualification;
- full magnetic induction at high magnetic Reynolds number.

The final report must explicitly separate **MHD performance conclusions** from **materials/process conclusions**.

---

## 3. Core Study Question

The LMX study should be framed around the following question:

> Given liquid lithium flowing in a duct under a prescribed magnetic field, what wall electrical conductance, AlN degradation level, AlN thickness, and pinhole fraction are allowable before the system loses the MHD benefit of electrical insulation?

The practical design question is:

> Can an AlN-like electrical barrier over a Li-compatible metal reduce Lorentz drag and wall-current leakage enough to justify further materials testing?

The expected form of the final conclusion should be:

> LMX predicts that an AlN-like electrically insulating layer reduces MHD current closure and Lorentz drag relative to a bare conducting wall, provided the effective wall conductance remains below approximately \(c_\mathrm{crit}\). LMX does not establish lithium compatibility, coating survivability, or printability; those require coupon testing and materials qualification.

---

## 4. LMX Scope for This Project

Use LMX within its bounded scope:

- inductionless, low magnetic Reynolds number MHD;
- incompressible, single-phase liquid-metal flow;
- prescribed magnetic field;
- structured mesh;
- rectangular duct first;
- extruded/fringing geometry second, if field varies axially;
- steady or fully developed cases where applicable;
- conducting and insulating wall-region diagnostics;
- JAX-compatible sensitivity analysis where the model is smooth.

Preferred solver order:

1. **Fully developed rectangular duct** for controlled verification and parametric sweeps.
2. **Layered/effective wall models** for AlN degradation studies.
3. **True multilayer wall extension** if not already available.
4. **Extruded/fringing-field geometry** only after the rectangular cases are validated.

---

## 5. Candidate Wall Electrical Models

The study must compare several wall models. Each model must be clearly labeled in every figure, table, and conclusion.

### 5.1 Model A: Perfect electrical insulator

Purpose: optimistic upper-bound model for intact AlN.

Assumptions:

- no normal current leakage through wall;
- no tangential wall current closure;
- wall conductivity effectively zero;
- substrate metal is electrically irrelevant if the AlN is truly intact and continuous.

Use this model as the **ideal-insulator reference**.

---

### 5.2 Model B: Bare conducting metal wall

Purpose: pessimistic baseline with no AlN insulation.

Candidate metal substrates:

- 316L stainless steel;
- Inconel 625;
- molybdenum;
- optional tungsten;
- optional vanadium alloy.

Use this model as the **conducting-wall reference**.

---

### 5.3 Model C: Finite-conductivity or degraded AlN wall

Purpose: represent AlN that is not a perfect electrical insulator.

Use AlN as a solid wall region with finite conductivity. Sweep AlN conductivity over a wide logarithmic range rather than assuming one value.

Primary nondimensional parameter:

\[
c_\mathrm{AlN}
=
\frac{\sigma_\mathrm{AlN} t_\mathrm{AlN}}
{\sigma_\mathrm{Li} a}
\]

where:

- \(\sigma_\mathrm{AlN}\) is AlN electrical conductivity;
- \(t_\mathrm{AlN}\) is AlN thickness;
- \(\sigma_\mathrm{Li}\) is lithium electrical conductivity;
- \(a\) is the characteristic duct length scale used consistently by LMX.

Minimum sweep:

\[
c_\mathrm{AlN}
=
0,\;10^{-10},\;10^{-9},\;10^{-8},\;10^{-7},\;10^{-6},\;10^{-5},\;10^{-4},\;10^{-3},\;10^{-2},\;10^{-1},\;1,\;10
\]

Refine near transition regions where forcing, leakage, or Lorentz power changes rapidly.

---

### 5.4 Model D: Pinholed or defective AlN effective model

Purpose: approximate cracks, coating defects, or local shunts to the metal substrate.

Use at least a smooth effective pinhole fraction:

\[
f_p \in [0,1]
\]

where:

- \(f_p = 0\): intact AlN;
- \(f_p = 1\): fully bare metal behavior.

A simple bounding model is:

\[
c_\mathrm{eff}
=
(1-f_p)c_\mathrm{AlN}
+
f_p c_\mathrm{metal}
\]

This is only an effective model. It does not resolve actual crack geometry unless explicit pinhole patches are added.

Required sweep:

\[
f_p
=
0,\;10^{-6},\;10^{-5},\;10^{-4},\;10^{-3},\;10^{-2},\;10^{-1},\;1
\]

Optional extension:

- localized conducting patches in the AlN wall;
- comparison between effective pinhole model and explicit defect geometry.

---

### 5.5 Model E: Effective AlN + metal stack

Purpose: practical reduced model for an AlN-coated metal substrate before a full multilayer implementation.

Represent the wall using an effective conductance:

\[
c_\mathrm{eff}
=
\frac{G_\mathrm{wall}}
{\sigma_\mathrm{Li}a}
\]

where \(G_\mathrm{wall}\) is an effective surface conductance.

Important modeling note:

A coated metal wall has two different possible electrical pathways:

1. **Tangential wall-current closure** along a conducting layer, usually represented by

   \[
   c_\parallel
   =
   \frac{\sigma_w t_w}{\sigma_\mathrm{Li}a}
   \]

2. **Normal leakage through an insulating coating** into a metal substrate, represented by a normal shunt conductance, for example

   \[
   g_\perp
   =
   \frac{\sigma_\mathrm{coat}/t_\mathrm{coat}}
   {\sigma_\mathrm{Li}/a}
   =
   \frac{\sigma_\mathrm{coat} a}
   {\sigma_\mathrm{Li}t_\mathrm{coat}}
   \]

The developer must define clearly which pathway is being modeled in each case.

For an intact insulating AlN coating, the metal substrate should have little or no electrical effect on the fluid-side MHD solution unless current can leak through, around, or across the AlN. If the AlN is pinholed, cracked, or finite-conductivity, the substrate can become important.

---

### 5.6 Model F: True AlN/metal multilayer wall

Purpose: highest-fidelity wall electrical model within LMX's inductionless framework.

Desired wall geometry:

\[
\text{Li fluid}
\;|\;
\text{AlN layer}
\;|\;
\text{metal substrate}
\]

Each layer must have its own:

- thickness;
- electrical conductivity;
- region mask;
- interface treatment;
- diagnostic outputs.

Minimum implementation requirements:

- rectangular ducts;
- arbitrary number of wall layers per side;
- different layer stacks on different duct sides if needed;
- independent conductivities for AlN and metal;
- interface-current continuity;
- diagnostics for interface-current residuals;
- mesh refinement in thin layers.

Optional extension:

- extruded/fringing geometry with multilayer walls.

---

## 6. Primary Nondimensional Groups

All input files and reports must use consistent definitions.

### 6.1 Wall conductance ratio

\[
c
=
\frac{\sigma_w t_w}
{\sigma_\mathrm{Li}a}
\]

Use this for thin-wall tangential conductance comparisons and Hunt-style validation.

For effective or multilayer stacks, define:

\[
c_\mathrm{eff}
=
\frac{G_\mathrm{wall}}
{\sigma_\mathrm{Li}a}
\]

The report must state exactly how \(G_\mathrm{wall}\) is computed.

---

### 6.2 Hartmann number

Use the convention appropriate to LMX and document it explicitly.

One common form is:

\[
Ha
=
B_0 a
\sqrt{
\frac{\sigma_\mathrm{Li}}
{\rho_\mathrm{Li}\nu_\mathrm{Li}}
}
\]

where:

- \(B_0\) is magnetic-field strength;
- \(a\) is characteristic duct scale;
- \(\sigma_\mathrm{Li}\) is lithium electrical conductivity;
- \(\rho_\mathrm{Li}\) is lithium density;
- \(\nu_\mathrm{Li}\) is lithium kinematic viscosity.

If LMX internally uses dynamic viscosity \(\mu\), convert consistently:

\[
\nu = \frac{\mu}{\rho}
\]

---

### 6.3 Reynolds number

\[
Re
=
\frac{Ua}{\nu_\mathrm{Li}}
\]

Use for context. Do not claim turbulent-flow validation unless explicitly demonstrated.

---

### 6.4 Interaction parameter

\[
N
=
\frac{\sigma_\mathrm{Li}B_0^2a}
{\rho_\mathrm{Li}U}
\]

Use to characterize Lorentz-force importance relative to inertia.

---

### 6.5 Magnetic Reynolds number

\[
Rm
=
\mu_0\sigma_\mathrm{Li}Ua
\]

Confirm that \(Rm \ll 1\) for the study cases to justify the inductionless assumption.

---

## 7. Required Input Structure

Create a structured case input system, preferably YAML or JSON. Every production case should be reproducible from a single input file.

### 7.1 Required geometry inputs

For rectangular ducts:

- fluid half-width \(a\);
- duct aspect ratio or full height/width;
- wall thickness;
- AlN thickness;
- metal substrate thickness;
- mesh resolution in the fluid;
- mesh resolution in each wall layer;
- optional wall-layer refinement settings;
- region names and material assignments.

For extruded/fringing cases:

- axial length;
- number of axial stations;
- axial mesh spacing;
- cross-section mesh;
- magnetic-field model;
- inlet/outlet assumptions;
- target mean velocity or imposed forcing;
- stationwise diagnostic settings.

---

### 7.2 Required lithium property inputs

Include temperature-dependent material properties:

\[
\sigma_\mathrm{Li}(T),\quad
\rho_\mathrm{Li}(T),\quad
\mu_\mathrm{Li}(T)\;\text{or}\;\nu_\mathrm{Li}(T)
\]

The viscosity convention must be explicit.

Required lithium cases:

- nominal property set;
- low-property-bound set;
- high-property-bound set;
- temperature sweep.

Suggested configurable temperature sweep:

\[
T
=
250^\circ\mathrm{C},\;
300^\circ\mathrm{C},\;
400^\circ\mathrm{C},\;
500^\circ\mathrm{C},\;
600^\circ\mathrm{C}
\]

The developer must fill actual property values from agreed sources or user-provided material tables. Do not silently hard-code unverified values.

---

### 7.3 Required wall material inputs

For each material:

- material name;
- electrical conductivity;
- unit;
- temperature dependence, if available;
- density, if needed by LMX;
- viscosity only for fluid regions;
- source or assumption;
- uncertainty range;
- whether the value is measured, literature-based, estimated, or purely parametric.

Minimum materials:

- liquid lithium;
- ideal insulator;
- AlN, finite conductivity;
- degraded AlN;
- 316L stainless steel;
- Inconel 625;
- molybdenum.

Optional materials:

- tungsten;
- vanadium alloy.

---

### 7.4 Required magnetic-field inputs

Support at least three magnetic-field types.

#### Constant transverse field

\[
\mathbf{B} = B_0 \hat{z}
\]

or the equivalent LMX coordinate convention.

#### Analytic fringing field

Example:

\[
B(x)
=
\frac{B_0}{2}
\left[
\tanh\left(\frac{x+x_0}{\ell_B}\right)
-
\tanh\left(\frac{x-x_0}{\ell_B}\right)
\right]
\]

#### Tabulated field

Allow tabulated \(B(x,y,z)\) input with interpolation.

Suggested magnetic-field sweep:

\[
B_0
=
0.5,\;1,\;2,\;4,\;6,\;8\;\mathrm{T}
\]

Use project-relevant field strengths if known.

---

### 7.5 Required flow inputs

Support two modes:

1. fixed forcing, report flow rate;
2. fixed target mean velocity, solve for required forcing.

Preferred mode for comparison:

> Fix \(U_\mathrm{target}\), then compute the required forcing or pressure-gradient proxy for each wall model.

Suggested target velocity sweep:

\[
U_\mathrm{target}
=
0.01,\;0.05,\;0.1,\;0.5,\;1.0\;\mathrm{m/s}
\]

---

## 8. Example Input File Structure

The developer should implement something close to the following structure.

```yaml
case_id: rect_li_aln_c_sweep_001
solver: fully_developed_inductionless

geometry:
  type: rectangular_duct
  a_m: 0.01
  aspect_ratio: 1.0
  wall_thickness_m: 0.001
  mesh:
    fluid_ny: 96
    fluid_nz: 96
    wall_layers:
      aln: 8
      metal: 8
    refinement: hartmann_and_interfaces

fluid:
  material: liquid_lithium
  temperature_C: 300
  sigma_S_per_m: null   # filled from property table
  rho_kg_per_m3: null
  viscosity_type: kinematic
  nu_m2_per_s: null

magnetic_field:
  type: constant
  B0_T: 2.0
  direction: z

flow:
  mode: fixed_mean_velocity
  U_target_m_per_s: 0.1

wall_model:
  type: effective_aln_plus_metal
  substrate: molybdenum
  aln:
    thickness_m: 0.0002
    sigma_S_per_m: 1.0e-8
  metal:
    thickness_m: 0.001
    sigma_S_per_m: null
  effective_model:
    parameterization: log10_c_eff
    log10_c_eff: -6.0
    pinhole_fraction: 0.0

outputs:
  scalar_metrics: true
  field_outputs: true
  save_potential: true
  save_current_density: true
  save_lorentz_force: true
  save_velocity: true
  save_wall_leakage: true
```

---

## 9. Required Code Checks and Additions

Before the production study, inspect the current LMX implementation and add the following if missing.

### 9.1 Unit-consistent material model

Clarify whether LMX expects:

- dynamic viscosity \(\mu\) in Pa·s; or
- kinematic viscosity \(\nu\) in m²/s.

Required actions:

- audit solver usage of viscosity;
- audit Hartmann-number helper;
- audit documentation;
- add conversion utilities;
- add tests for \(Ha\), \(Re\), \(N\), and \(Rm\);
- make input files explicit about viscosity type.

Deliverables:

- unit-consistency note;
- code changes if needed;
- tests showing expected nondimensional numbers.

---

### 9.2 Multilayer wall support

If LMX cannot currently model fluid | AlN | metal on the same wall, implement rectangular-duct multilayer wall support.

Minimum functionality:

- arbitrary number of layers per wall side;
- independent material assignment per layer;
- correct solid/fluid masks;
- conductivity field assignment by layer;
- interface-current continuity;
- diagnostics at each material interface;
- thin-layer mesh controls.

Tests:

- single-layer wall reduces to existing wall model;
- two identical layers reduce to one equivalent layer;
- insulating layer over metal approaches ideal-insulator behavior as \(\sigma_\mathrm{AlN}\to0\);
- conducting layer recovers Hunt-style conducting-wall behavior where applicable;
- mesh refinement improves interface residuals.

---

### 9.3 Effective wall conductance model

Implement a fast, smooth effective-wall model with continuous \(c_\mathrm{eff}\).

Requirements:

- parameterize by \(\log_{10}(c_\mathrm{eff})\);
- accept scalar or side-specific conductance values;
- support JAX autodiff;
- support comparison to explicit wall regions;
- document whether the conductance represents tangential wall conduction or normal leakage.

---

### 9.4 Pinhole/degradation model

Implement required effective model:

\[
c_\mathrm{eff}
=
(1-f_p)c_\mathrm{AlN}
+
f_p c_\mathrm{metal}
\]

Optional explicit model:

- local conductive patches through AlN;
- defect patch size and location as input parameters;
- comparison of localized defects to effective \(f_p\).

Important:

- use autodiff only for smooth effective models;
- use finite differences or brute-force sweeps for discrete defect geometries.

---

### 9.5 Fixed-flow forcing solver

Add a wrapper that solves for the forcing or pressure-gradient proxy needed to achieve target mean velocity:

\[
\bar{U}(F_\mathrm{req}) = U_\mathrm{target}
\]

For each case, output:

- required forcing \(F_\mathrm{req}\);
- pressure-gradient proxy if defined;
- achieved mean velocity;
- solver residual for the fixed-flow solve.

Implementation options:

- scalar root finding;
- linear scaling if validated for the chosen equations;
- differentiable root solve if using autodiff through the wrapper;
- implicit differentiation if appropriate.

---

### 9.6 Autodiff-compatible objective functions

Implement JAX-compatible scalar objectives for smooth cases:

- required forcing / drag proxy;
- Lorentz power;
- wall leakage;
- mean current density;
- peak current density, or a smooth approximation to peak current;
- velocity flattening / uniformity metric;
- charge-balance residual;
- interface-current residual.

For peak-like quantities, consider smooth approximations, for example log-sum-exp, if direct max operations make gradients fragile.

---

## 10. Simulation Campaign Overview

The work should proceed in phases. Do not run the full production matrix before verification and convergence are complete.

| Phase | Name | Main Purpose | Go/No-Go Criterion |
|---:|---|---|---|
| 0 | Repository preparation | Confirm code state and baseline tests | Existing validation passes or deviations documented |
| 1 | Unit and property audit | Ensure physical consistency | Viscosity and nondimensional numbers verified |
| 2 | Model implementation | Add effective, pinhole, and multilayer wall models if needed | New tests pass |
| 3 | Verification and convergence | Establish numerical credibility | Mesh and limiting-case behavior acceptable |
| 4 | 2D rectangular-duct sweep | Main parametric MHD result | Complete scalar table and field plots |
| 5 | AlN degradation and threshold study | Find allowable degradation | \(c_\mathrm{crit}\), \(t_\mathrm{min}\), \(f_{p,\max}\) reported |
| 6 | Effective and multilayer stack comparison | Compare wall-stack electrical models | Ranking table completed |
| 7 | Extruded/fringing-field cases | Assess spatially varying B-field behavior | Stationwise diagnostics produced |
| 8 | Autodiff sensitivity and inverse design | Quantify gradients and thresholds | AD gradients validated against finite differences |
| 9 | Final reporting | Produce report and reproducible archive | All deliverables complete |

Current implementation status:

- Phases 0-2 have an executable reduced artifact in
  `examples/li_aln_wall_stack_phase0_2.py`.
- Phases 3-6 now have a bounded reduced artifact in
  `examples/li_aln_wall_stack_phase3_6.py`, covering the operating matrix,
  AlN degradation thresholds, pinhole limits, and substrate-conductivity
  comparisons.
- The true `fluid | AlN | metal` multilayer rectangular mesh now has an
  executable geometry gate in `examples/li_aln_multilayer_mesh_qa.py`, with
  aligned interfaces, region IDs, and explicit conductivity fields.
- Solved multilayer limiting cases now have an internal gate in
  `examples/li_aln_multilayer_solve.py`. The gate runs ideal-insulator,
  intact-AlN, degraded-AlN, and bare-metal electrical wall models on the
  explicit finite-volume mesh and records pressure proxy, current magnitude,
  dimensional charge residuals for audit, normalized charge balance, normalized
  local current divergence, and normalized interface-current residual.
- External-code limiting-case comparisons and heavier high-Hartmann-number
  mesh ladders remain the next high-fidelity solver-extension lane.
- The representative solved mesh ladder is implemented in
  `examples/li_aln_multilayer_convergence.py` for intact-AlN and bare-metal
  electrical wall limits. It records pressure/current convergence to the
  finest retained mesh and normalized current-closure diagnostics. A matching
  FreeMHD/OpenFOAM limiting-case comparison remains separate.

---

## 11. Phase 0: Repository Preparation

### Goals

- Establish reproducible code baseline.
- Confirm LMX is working before modifying it.
- Record software environment.

### Required actions

Run:

- current test suite;
- existing validation scripts;
- existing release-readiness or benchmark scripts, if available;
- basic example cases.

Record:

- commit hash;
- package version;
- Python version;
- JAX version;
- operating system;
- hardware used;
- solver settings;
- pass/fail status.

### Required baseline validations

Confirm existing benchmark behavior for:

- Hartmann duct;
- Shercliff duct;
- Hunt duct;
- current closure;
- charge balance;
- interface-current diagnostics;
- extruded/fringing benchmark if available.

### Deliverables

- `phase0_environment.md`
- `phase0_test_results.csv`
- `phase0_validation_summary.md`

---

## 12. Phase 1: Unit and Property Audit

### Goals

- Ensure material properties are interpreted correctly.
- Prevent viscosity-unit errors.
- Confirm nondimensional parameters are computed consistently.

### Required checks

For representative lithium and wall inputs, compute and report:

- \(Ha\);
- \(Re\);
- \(N\);
- \(Rm\);
- \(c\);
- \(c_\mathrm{eff}\);
- \(g_\perp\), if normal leakage is modeled.

### Acceptance criteria

- All nondimensional numbers match hand calculations.
- Viscosity convention is explicit in every input file.
- Temperature-dependent properties are labeled with units and source notes.

### Deliverables

- `phase1_units_and_properties.md`
- `material_properties.csv`
- `nondimensional_check.csv`
- unit tests for nondimensional utilities.

---

## 13. Phase 2: Model Implementation

### Goals

Add missing wall models required for the study.

### Required implementation order

1. effective wall conductance model;
2. smooth pinhole/degradation model;
3. fixed-flow forcing wrapper;
4. autodiff scalar objectives;
5. true multilayer wall model;
6. optional localized pinhole patches.

### Required tests

| Test | Expected Result |
|---|---|
| \(c_\mathrm{eff}\to0\) | approaches ideal-insulator solution |
| \(c_\mathrm{eff}\) large | approaches conducting-wall solution where applicable |
| \(f_p=0\) | equals intact AlN/effective-insulator case |
| \(f_p=1\) | equals bare-metal effective case |
| identical multilayers | match equivalent single layer |
| mesh refinement | current residuals decrease or remain acceptably small |
| AD gradients | agree with finite differences for smooth cases |

### Deliverables

- code patch or branch;
- tests;
- documentation for new input options;
- example cases.

---

## 14. Phase 3: Verification and Numerical Convergence

### 14.1 Mesh convergence

Run at least three mesh levels:

- coarse;
- medium;
- fine.

Representative cases:

1. ideal insulating wall;
2. bare conducting wall;
3. intermediate conductance \(c=10^{-3}\);
4. low Hartmann number;
5. high Hartmann number;
6. multilayer AlN/metal wall, if implemented.

Track convergence of:

- mean velocity;
- required forcing;
- Lorentz power;
- mean current density;
- peak current density;
- wall leakage;
- charge-balance residual;
- interface-current residual;
- velocity-flattening metric.

Acceptance target:

- key integral metrics change by less than 1–2% from medium to fine mesh;
- residuals remain small and trend acceptably with refinement;
- field structures are not qualitatively mesh-dependent.

---

### 14.2 Limiting-case validation

Verify:

- \(c\to0\) approaches ideal-insulating-wall behavior;
- large \(c\) approaches conducting-wall behavior;
- \(f_p=0\) equals intact AlN effective model;
- \(f_p=1\) equals bare-metal effective model;
- current conservation is maintained;
- wall leakage behaves monotonically where expected.

---

### 14.3 Autodiff validation

For smooth effective-wall cases, compare JAX autodiff gradients against centered finite differences.

For a scalar objective \(J\):

\[
\epsilon_\mathrm{grad}
=
\frac{|g_\mathrm{AD}-g_\mathrm{FD}|}
{|g_\mathrm{FD}|+\epsilon}
\]

Target:

- \(10^{-3}\) to \(10^{-2}\) relative error for smooth, well-conditioned cases;
- document deviations near thresholds, nonsmooth max operations, root-solve discontinuities, or ill-conditioned parameters.

### Deliverables

- `mesh_convergence.csv`
- `limiting_case_validation.csv`
- `autodiff_validation.csv`
- convergence plots.

---

## 15. Phase 4: Main 2D Rectangular-Duct Sweep

The 2D rectangular duct is the primary study geometry because it is easiest to validate and interpret.

### 15.1 Baseline case

Define one nominal baseline:

- rectangular duct;
- liquid lithium at nominal temperature;
- fixed target mean velocity;
- constant transverse magnetic field;
- specified wall thickness;
- specified AlN thickness;
- specified candidate metal substrate;
- verified medium or fine mesh.

Run baseline for:

1. ideal insulator;
2. bare 316L;
3. bare IN625;
4. bare Mo;
5. degraded AlN;
6. AlN + 316L effective stack;
7. AlN + IN625 effective stack;
8. AlN + Mo effective stack;
9. pinholed AlN effective model;
10. true AlN/metal multilayer, if implemented.

---

### 15.2 Conductance-ratio sweep

Sweep:

\[
c
=
0,\;10^{-10},\;10^{-9},\;10^{-8},\;10^{-7},\;10^{-6},\;10^{-5},\;10^{-4},\;10^{-3},\;10^{-2},\;10^{-1},\;1,\;10
\]

For each case output:

- required forcing;
- normalized forcing;
- Lorentz power;
- mean current density;
- peak current density;
- wall leakage;
- charge-balance residual;
- interface-current residual;
- velocity-flattening metric;
- field outputs for selected cases.

Refine the sweep around sharp changes to identify transition behavior.

---

### 15.3 Hartmann-number sweep

Sweep either magnetic-field strength or Hartmann number.

Suggested Hartmann values:

\[
Ha
=
10,\;30,\;100,\;300,\;1000
\]

At each \(Ha\), run selected conductance values:

\[
c
=
0,\;10^{-8},\;10^{-6},\;10^{-4},\;10^{-2},\;1
\]

Report how \(c_\mathrm{crit}\) changes with \(Ha\).

---

### 15.4 Lithium-property and temperature sweep

For each temperature, update:

- \(\sigma_\mathrm{Li}(T)\);
- \(\rho_\mathrm{Li}(T)\);
- \(\nu_\mathrm{Li}(T)\) or \(\mu_\mathrm{Li}(T)\);
- \(Ha\);
- \(Re\);
- \(N\);
- \(Rm\).

Suggested temperatures:

\[
250^\circ\mathrm{C},\;300^\circ\mathrm{C},\;400^\circ\mathrm{C},\;500^\circ\mathrm{C},\;600^\circ\mathrm{C}
\]

Run at least selected wall models:

- ideal insulator;
- bare metal;
- degraded AlN at representative \(c\);
- pinholed AlN at representative \(f_p\).

---

### 15.5 Flow-speed sweep

Sweep:

\[
U_\mathrm{target}
=
0.01,\;0.05,\;0.1,\;0.5,\;1.0\;\mathrm{m/s}
\]

Report:

- required forcing;
- Lorentz power;
- \(N\);
- \(Re\);
- whether inductionless assumption remains valid through \(Rm\).

---

## 16. Phase 5: AlN Degradation and Threshold Study

### Goals

Find threshold values below which the AlN behaves close to an ideal insulator.

### Parameters to sweep

- \(\sigma_\mathrm{AlN}\);
- \(t_\mathrm{AlN}\);
- \(c_\mathrm{AlN}\);
- \(g_\perp\), if normal leakage is modeled;
- \(f_p\);
- substrate metal conductivity;
- temperature.

### Threshold definitions

Find values satisfying each criterion:

#### Forcing tolerance

\[
F_\mathrm{req}
\leq
(1+\delta)F_\mathrm{insulator}
\]

where:

\[
\delta = 0.05,\;0.10,\;0.25
\]

#### Leakage tolerance

\[
J_\mathrm{leak}
\leq
\alpha J_\mathrm{bare\ metal}
\]

where:

\[
\alpha = 0.01,\;0.05,\;0.10
\]

#### Lorentz-power tolerance

\[
P_\mathrm{Lorentz}
\leq
(1+\delta)P_\mathrm{insulator}
\]

where:

\[
\delta = 0.05,\;0.10,\;0.25
\]

### Required threshold outputs

For each geometry and field condition, report:

- \(c_\mathrm{crit}\);
- \(\sigma_{\mathrm{AlN},\mathrm{crit}}\);
- \(t_{\mathrm{AlN},\min}\);
- \(g_{\perp,\mathrm{crit}}\), if used;
- \(f_{p,\max}\);
- substrate dependence;
- uncertainty from lithium properties.

---

## 17. Phase 6: AlN/Metal Stack Comparison

For each substrate:

- 316L;
- IN625;
- Mo;
- optional W;
- optional vanadium alloy;

compare:

1. bare metal;
2. ideal AlN over metal;
3. finite-conductivity AlN over metal;
4. pinholed AlN over metal;
5. effective AlN+metal stack;
6. true multilayer AlN/metal, if implemented.

### Required conclusions

Answer:

- Does substrate conductivity matter when AlN is ideal?
- Does substrate conductivity matter when AlN is degraded?
- Which metal substrate produces the worst MHD penalty if the coating is pinholed?
- Is the MHD result controlled primarily by AlN integrity rather than substrate choice?
- Which stack should be prioritized for experimental coupon testing from an MHD standpoint?

---

## 18. Phase 7: Extruded / Fringing-Field Study

Run this only after the rectangular-duct study is complete.

### Purpose

Evaluate whether the rectangular-duct conclusions remain qualitatively valid when the magnetic field varies axially.

### Required cases

At minimum:

1. ideal insulator;
2. bare conducting metal;
3. \(c=10^{-6}\);
4. \(c=10^{-4}\);
5. \(c=10^{-2}\);
6. pinholed effective model;
7. true multilayer case if available.

### Required stationwise outputs

- axial position \(x\);
- magnetic field \(B(x)\);
- mean velocity;
- local forcing or pressure-gradient proxy;
- Lorentz power;
- wall leakage;
- mean current density;
- peak current density;
- charge-balance residual;
- interface-current residual.

### Required field snapshots

At selected axial stations:

- low field;
- rising field;
- peak field;
- falling field.

For each station plot:

- velocity;
- electric potential;
- current-density magnitude;
- current vectors or streamlines;
- Lorentz-force magnitude;
- wall-normal current leakage.

---

## 19. Phase 8: Autodifferentiation and Sensitivity Analysis

Use JAX autodifferentiation for smooth cases.

### 19.1 Scalar objectives

Define scalar objectives:

#### Required forcing

\[
J_1 = F_\mathrm{req}
\]

#### Lorentz power

\[
J_2 = P_\mathrm{Lorentz}
\]

#### Wall-current leakage

\[
J_3 = J_\mathrm{leak}
\]

#### Peak current or smooth peak-current approximation

\[
J_4 = J_\mathrm{peak}
\]

#### Velocity flattening

\[
J_5 = \Phi_\mathrm{flat}
\]

#### Composite objective, optional

\[
J_\mathrm{total}
=
w_F \frac{F_\mathrm{req}}{F_\mathrm{ref}}
+
w_L \frac{J_\mathrm{leak}}{J_{\mathrm{leak},\mathrm{ref}}}
+
w_P \frac{P_\mathrm{Lorentz}}{P_\mathrm{ref}}
\]

Weights must be documented.

---

### 19.2 Required gradients

Compute gradients with respect to:

\[
\log_{10}(c_\mathrm{eff})
\]

\[
B_0
\]

\[
U_\mathrm{target}
\]

\[
\sigma_\mathrm{Li}
\]

\[
\rho_\mathrm{Li}
\]

\[
\nu_\mathrm{Li}
\]

\[
\sigma_\mathrm{AlN}
\]

\[
t_\mathrm{AlN}
\]

\[
f_p
\]

Use \(f_p\) gradients only for the smooth effective pinhole model.

---

### 19.3 Normalized sensitivities

Report normalized sensitivities:

\[
S_p^J
=
\frac{p}{J}
\frac{\partial J}{\partial p}
\]

For log-parameters:

\[
S_{\log_{10}p}^J
=
\frac{1}{J}
\frac{\partial J}{\partial \log_{10}p}
\]

### 19.4 Autodiff validation

Compare AD gradients to finite differences for representative cases:

- low \(c\);
- transition \(c\);
- high \(c\);
- low \(Ha\);
- high \(Ha\);
- low \(f_p\);
- high \(f_p\).

### 19.5 Optional inverse design

Solve inverse problems:

1. Minimum AlN thickness satisfying:

   \[
   F_\mathrm{req}
   \leq
   1.1F_\mathrm{insulator}
   \]

2. Maximum conductance satisfying:

   \[
   J_\mathrm{leak}
   \leq
   0.1J_\mathrm{bare\ metal}
   \]

3. Maximum pinhole fraction satisfying:

   \[
   P_\mathrm{Lorentz}
   \leq
   1.2P_\mathrm{insulator}
   \]

Use gradient-based optimization only where the objective is smooth and gradients are validated. Otherwise use root finding, bisection, or brute-force sweeps.

---

## 20. Required Diagnostic Definitions

The report and code must use precise definitions.

### 20.1 Mean velocity

\[
\bar{U}
=
\frac{1}{A_f}
\int_{A_f} u\,dA
\]

### 20.2 Flow rate

\[
Q
=
\int_{A_f} u\,dA
\]

### 20.3 Required forcing / pressure-drop proxy

\[
\bar{U}(F_\mathrm{req}) = U_\mathrm{target}
\]

Report normalized forcing:

\[
R_F
=
\frac{F_\mathrm{req}}{F_\mathrm{insulator}}
\]

and conducting-wall penalty fraction:

\[
\Delta_F
=
\frac{F_\mathrm{req}-F_\mathrm{insulator}}
{F_\mathrm{bare\ metal}-F_\mathrm{insulator}}
\]

### 20.4 Lorentz power

Use a consistent positive drag-power measure, for example:

\[
P_\mathrm{Lorentz}
=
-
\int_V
\mathbf{u}\cdot(\mathbf{J}\times\mathbf{B})\,dV
\]

or the equivalent sign convention already used in LMX. State the convention explicitly.

### 20.5 Mean current density

\[
\langle |\mathbf{J}| \rangle
=
\frac{1}{V}
\int_V
|\mathbf{J}|\,dV
\]

### 20.6 Peak current density

\[
J_\mathrm{peak}
=
\max_V |\mathbf{J}|
\]

For autodiff, optionally use smooth max approximation.

### 20.7 Wall-current leakage

\[
J_\mathrm{leak}
=
\int_{\Gamma_\mathrm{wall}}
|\mathbf{J}\cdot\mathbf{n}|\,dA
\]

For multilayer walls, also report current crossing:

- fluid/AlN interface;
- AlN/metal interface;
- outer metal boundary, if relevant.

### 20.8 Charge-conservation residual

Report \(L_2\) and \(L_\infty\) norms of:

\[
\nabla\cdot\mathbf{J}
\]

### 20.9 Interface-current residual

For material interfaces:

\[
R_\mathrm{interface}
=
\left[\mathbf{J}\cdot\mathbf{n}\right]_\mathrm{jump}
\]

Report maximum and normed residuals.

### 20.10 Velocity-flattening metric

Use one primary metric:

\[
\Phi_\mathrm{flat}
=
1
-
\frac{
\sqrt{\langle(u-\bar{U})^2\rangle}
}{\bar{U}}
\]

Also plot centerline and wall-normal profiles so the metric can be interpreted.

---

## 21. Required Tables

The final report must include these tables.

### Table 1: Simulation matrix

| Column | Description |
|---|---|
| case_id | unique case name |
| geometry | rectangular, layered, extruded, etc. |
| solver | LMX solver used |
| wall_model | ideal, bare metal, degraded AlN, pinhole, effective stack, multilayer |
| substrate | 316L, IN625, Mo, etc. |
| temperature_C | lithium/wall temperature |
| sigma_Li | lithium conductivity |
| rho_Li | lithium density |
| viscosity | lithium viscosity with type and unit |
| B0_T | magnetic-field strength |
| U_target | target mean velocity |
| Ha | Hartmann number |
| Re | Reynolds number |
| N | interaction parameter |
| Rm | magnetic Reynolds number |
| c | wall conductance ratio |
| c_eff | effective wall conductance ratio |
| f_p | pinhole fraction |
| mesh | mesh identifier |
| solver_status | converged / failed / warning |

---

### Table 2: Material-property assumptions

| Column | Description |
|---|---|
| material | Li, AlN, 316L, IN625, Mo, etc. |
| property | conductivity, density, viscosity, thickness, etc. |
| value | numerical value |
| unit | SI unit |
| temperature_C | associated temperature |
| source_or_assumption | measured, literature, estimated, parametric |
| uncertainty_range | low/high or notes |
| cases_used | case groups using the value |

---

### Table 3: Verification and validation

| Column | Description |
|---|---|
| benchmark | Hartmann, Shercliff, Hunt, etc. |
| mesh | mesh level |
| expected_result | reference value |
| LMX_result | computed value |
| absolute_error | absolute difference |
| relative_error | percent or fraction |
| pass_fail | pass/fail |
| notes | comments |

---

### Table 4: Mesh convergence

| Column | Description |
|---|---|
| case_id | case name |
| mesh_level | coarse/medium/fine |
| cells | number of cells |
| F_req | required forcing |
| Lorentz_power | Lorentz power |
| leakage | wall-current leakage |
| mean_current | mean current density |
| peak_current | peak current density |
| charge_residual | charge-balance residual |
| interface_residual | interface-current residual |
| relative_change | change from previous mesh |

---

### Table 5: Wall-model ranking

| Column | Description |
|---|---|
| stack_model | wall electrical model |
| substrate | metal substrate |
| F_req | required forcing |
| normalized_forcing | relative to ideal insulator |
| Lorentz_power | integrated Lorentz power |
| mean_current | mean current density |
| peak_current | peak current density |
| leakage | wall-current leakage |
| flattening | velocity-flattening metric |
| degradation_sensitivity | high/medium/low or numeric sensitivity |
| LMX_confidence | good/moderate/pending |
| comments | interpretation |

---

### Table 6: AlN degradation thresholds

| Column | Description |
|---|---|
| geometry | duct geometry |
| B0_T | magnetic-field strength |
| U_target | target velocity |
| Ha | Hartmann number |
| tolerance | 5%, 10%, 25%, etc. |
| criterion | forcing, leakage, Lorentz power |
| c_crit | critical conductance ratio |
| sigma_AlN_crit | critical AlN conductivity |
| t_AlN_min | minimum AlN thickness |
| g_perp_crit | normal leakage threshold, if used |
| f_p_max | maximum pinhole fraction |
| substrate | substrate used |
| notes | interpretation |

---

### Table 7: Autodiff sensitivity

| Column | Description |
|---|---|
| case_id | case name |
| objective | forcing, leakage, Lorentz power, etc. |
| parameter | parameter differentiated |
| AD_gradient | autodiff result |
| FD_gradient | finite-difference result |
| relative_difference | AD/FD agreement |
| normalized_sensitivity | nondimensional sensitivity |
| pass_fail | pass/fail |
| notes | comments |

---

### Table 8: Extruded/fringing-field results

| Column | Description |
|---|---|
| case_id | case name |
| station | station index |
| x | axial location |
| B_x | local field value or magnitude |
| mean_velocity | stationwise mean velocity |
| local_forcing | local forcing / pressure-gradient proxy |
| Lorentz_power | local Lorentz power |
| leakage | local wall leakage |
| current_residual | current-closure residual |
| interface_residual | interface residual |

---

## 22. Required Plots

### 22.1 Geometry and setup plots

1. Rectangular duct cross-section with fluid, AlN, and metal regions labeled.
2. Mesh plot showing wall-layer resolution.
3. Magnetic-field profile for constant, analytic, and tabulated cases.
4. Conductance-ratio map for wall models.
5. Schematic showing tangential conductance versus normal leakage pathways.

---

### 22.2 Representative field plots

For each representative stack:

- ideal insulator;
- bare conducting metal;
- degraded AlN;
- pinholed AlN;
- AlN+metal effective stack;
- true multilayer stack, if implemented;

plot:

1. velocity contours;
2. electric potential contours;
3. current-density magnitude;
4. current vectors or streamlines;
5. Lorentz-force magnitude;
6. wall-normal current leakage;
7. line profiles through centerline and near wall.

---

### 22.3 Parametric sweep plots

Plot versus \(c\) on log scale:

1. required forcing / pressure-drop proxy;
2. normalized forcing \(R_F\);
3. Lorentz power;
4. mean current density;
5. peak current density;
6. wall-current leakage;
7. charge-balance residual;
8. interface-current residual;
9. velocity-flattening metric.

Plot versus \(Ha\):

1. required forcing;
2. Lorentz power;
3. wall-current leakage;
4. peak current density;
5. sensitivity to conductance ratio.

Plot versus \(f_p\):

1. required forcing;
2. leakage;
3. Lorentz power;
4. peak current;
5. transition from intact AlN to bare metal.

---

### 22.4 Sensitivity and autodiff plots

1. Normalized sensitivity bar chart for each objective.
2. Heat map of \(\partial J_\mathrm{drag}/\partial\log_{10}c\) over \(Ha\) and \(c\).
3. Heat map of leakage sensitivity over \(Ha\) and \(c\).
4. Autodiff-vs-finite-difference parity plot.
5. Threshold map for acceptable AlN degradation.
6. Inverse-design plot showing minimum \(t_\mathrm{AlN}\) versus \(B_0\) or \(Ha\).
7. Maximum allowable pinhole fraction versus \(Ha\) and \(U_\mathrm{target}\).

---

### 22.5 Extruded/fringing-field plots

If extruded cases are run, plot:

1. \(B(x)\);
2. stationwise required forcing or pressure-gradient proxy;
3. stationwise Lorentz power;
4. stationwise wall-current leakage;
5. stationwise current-closure residual;
6. stationwise interface residual;
7. selected cross-sectional field plots at low-field, rising-field, peak-field, and falling-field stations.

---

## 23. Expected Results Format

The study should not assume the answer in advance. It should produce numerical values and rankings from simulations.

Expected final ranking table format:

| Stack model | MHD pressure-drop reduction | Wall-current leakage | Sensitivity to AlN degradation | LMX confidence | Interpretation |
|---|---:|---:|---:|---|---|
| bare 316L / IN625 / Mo | high current closure expected | high expected | n/a | good | conducting-wall baseline |
| ideal AlN insulator | low current closure expected | low expected | optimistic | good for MHD only | upper-bound insulation case |
| degraded AlN | intermediate | intermediate | important | good if conductivity known | key parametric case |
| pinholed AlN | depends on \(f_p\) | depends on \(f_p\) | important | moderate | defect model, not material proof |
| AlN + metal effective wall | depends on effective conductance | depends | needs calibration | moderate | useful reduced model |
| true AlN/metal multilayer | best electrical representation | best diagnostic value | best | pending implementation | preferred once implemented |

The report must include actual computed numerical values in the final version.

---

## 24. Acceptance Criteria

The study is complete only if the following are satisfied.

1. Existing LMX validation tests pass or deviations are documented.
2. Viscosity convention is resolved and documented.
3. Nondimensional numbers are unit-tested.
4. Rectangular-duct limiting behavior is correct for insulating and conducting walls.
5. Mesh convergence is demonstrated for representative cases.
6. Charge conservation and interface-current residuals are reported.
7. Effective-wall and pinhole models are documented clearly.
8. Multilayer implementation is validated if added.
9. Autodiff gradients agree with finite-difference checks for smooth cases.
10. Conductance-ratio thresholds are reported.
11. AlN degradation thresholds are reported.
12. The final ranking separates ideal insulation, finite-conductivity AlN, pinhole approximation, effective-stack model, true multilayer model, and bare-metal cases.
13. All cases are reproducible from input files.
14. All scalar outputs are stored in machine-readable form.
15. The final report clearly states that LMX does not certify lithium compatibility or printability.

---

## 25. Deliverables

### 25.1 Code deliverables

- LMX branch or patch set;
- new wall-model implementation, if needed;
- fixed-flow forcing wrapper;
- autodiff objective functions;
- pinhole/effective degradation model;
- multilayer wall model, if implemented;
- tests for new functionality;
- documentation for new input fields.

### 25.2 Input and run deliverables

- input files for all major cases;
- case matrix file;
- case-runner script;
- post-processing script;
- plotting script;
- autodiff sensitivity script;
- optional notebook for exploration.

### 25.3 Data deliverables

- scalar metrics as CSV or Parquet;
- field outputs as NPZ, HDF5, NetCDF, or equivalent;
- mesh-convergence data;
- validation data;
- gradient data;
- material-property table;
- final case metadata table.

### 25.4 Report deliverables

The technical report should include:

1. executive summary;
2. collaborator question and study boundary;
3. LMX model description;
4. geometry and material assumptions;
5. wall electrical models;
6. nondimensional parameters;
7. code changes;
8. verification and validation;
9. mesh convergence;
10. rectangular-duct results;
11. AlN degradation sensitivity;
12. effective-stack results;
13. multilayer results, if implemented;
14. extruded/fringing-field results, if performed;
15. autodiff sensitivity analysis;
16. final ranking;
17. recommended next simulations;
18. required experimental/materials tests;
19. limitations;
20. appendices with input files and case matrix.

---

## 26. Recommended Repository Layout

Create a dedicated study directory:

```text
studies/li_aln_wall_mhd/
  README.md
  plan.md
  inputs/
    baseline/
    c_sweep/
    ha_sweep/
    temperature_sweep/
    degradation_sweep/
    pinhole_sweep/
    multilayer/
    extruded_fringing/
  scripts/
    run_cases.py
    postprocess.py
    make_plots.py
    autodiff_sensitivity.py
    validate_gradients.py
  results/
    scalars/
    fields/
    logs/
  figures/
    geometry/
    fields/
    sweeps/
    sensitivities/
    extruded/
  tables/
    simulation_matrix.csv
    material_properties.csv
    mesh_convergence.csv
    validation.csv
    ranking.csv
    degradation_thresholds.csv
    autodiff_sensitivity.csv
  report/
    report.md
    report.pdf
  tests/
    test_units.py
    test_effective_wall.py
    test_pinhole_model.py
    test_multilayer_wall.py
    test_autodiff_gradients.py
```

---

## 27. Final Decision Logic

At the end of the study, the developer should provide a concise decision table answering the following.

1. Does ideal AlN insulation provide a large MHD benefit relative to bare conducting metal?
2. At what conductance ratio \(c\) does the benefit begin to degrade?
3. What \(c_\mathrm{crit}\), \(\sigma_{\mathrm{AlN},\mathrm{crit}}\), \(t_{\mathrm{AlN},\min}\), or \(f_{p,\max}\) keeps the design within 5%, 10%, and 25% of the ideal-insulator case?
4. Does the metal substrate matter electrically when AlN is intact?
5. How badly does performance degrade if AlN becomes conductive or pinholed?
6. Which candidate wall model is best from an MHD standpoint?
7. Which result is most sensitive to uncertain material properties?
8. Which parameters should experimentalists measure first?
9. Which LMX predictions are robust?
10. Which predictions require better material-property data or new code validation?

---

## 28. Recommended Final Wording

The final report should use careful language like this:

> LMX predicts that an AlN-like electrically insulating wall can reduce MHD current closure, Lorentz drag, and wall-current leakage relative to a bare conducting metal wall. The benefit persists only while the effective wall conductance remains below the threshold identified in this study. If the AlN becomes sufficiently conductive, pinholed, cracked, or otherwise electrically bypassed by the metal substrate, the solution transitions toward conducting-wall behavior.

And:

> These simulations evaluate MHD performance only. They do not establish lithium compatibility, corrosion resistance, coating adhesion, thermal-cycling survival, irradiation response, or additive-manufacturing feasibility. Those must be assessed through separate material-property measurements, lithium exposure tests, thermal cycling, microscopy, electrical resistivity measurements before and after exposure, and process-development coupons.

---

## 29. Immediate Next Actions for Developer

1. Create the `studies/li_aln_wall_mhd/` directory.
2. Run and record baseline LMX validation.
3. Audit viscosity and nondimensional-number conventions.
4. Implement or verify effective wall conductance support.
5. Implement or verify fixed-target-flow forcing wrapper.
6. Implement smooth pinhole/degradation model.
7. Add autodiff objectives and gradient validation.
8. Determine whether true multilayer wall support exists; if not, implement rectangular-duct multilayer support.
9. Build the baseline rectangular duct input.
10. Run mesh convergence.
11. Run \(c\)-sweep, \(Ha\)-sweep, temperature sweep, and degradation sweep.
12. Run selected extruded/fringing-field cases.
13. Generate all required tables and plots.
14. Write the final report with clear MHD/materials boundary.

---

## 30. Summary of Expected Study Value

The study should give the project a quantitative MHD basis for deciding whether AlN-based insulation is worth further experimental development. The most important outputs are not just field plots, but thresholds:

- how low the effective wall conductance must be;
- how thick the AlN must be for assumed conductivity;
- how much pinhole fraction can be tolerated;
- how strongly the answer changes with lithium properties and magnetic field;
- whether the substrate metal matters electrically after insulation degradation;
- which measurements are most important for experimentalists to obtain.

The final product should make it easy to say:

> “This is what LMX predicts about MHD performance.”

and separately:

> “This is what still requires materials testing before selecting a printable AlN/metal component.”
