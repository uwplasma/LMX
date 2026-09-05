# Equations and assumptions

LMX solves incompressible, inductionless liquid-metal MHD. The magnetic
Reynolds number is assumed small, so the imposed magnetic field $\mathbf B$ is
not evolved. The full isothermal formulation below uses density $\rho$,
kinematic viscosity $\nu$, conductivity $\sigma$, pressure $p$, and body drive
$\mathbf f$. The implemented model reductions are described below it:

$$
\nabla\cdot\mathbf u = 0,
$$

$$
\rho\left(\frac{\partial\mathbf u}{\partial t}
+\mathbf u\cdot\nabla\mathbf u\right)
=-\nabla p+\rho\nu\nabla^2\mathbf u
+\mathbf J\times\mathbf B+\mathbf f,
$$

$$
\mathbf J=\sigma\left(-\nabla\phi+\mathbf u\times\mathbf B\right),
\qquad \nabla\cdot\mathbf J=0.
$$

The electric potential therefore satisfies the variable-conductivity equation

$$
\nabla\cdot(\sigma\nabla\phi)
=\nabla\cdot\left[\sigma(\mathbf u\times\mathbf B)\right].
$$

Fully developed models set $\mathbf u=(u(y,z),0,0)$ and solve the coupled
cross-section equations. Extruded models retain all three velocity, current,
and Lorentz-force components. The generic duct and pipe recurrences omit
$\mathbf u\cdot\nabla\mathbf u$ and use a collocated projection with flow
adjustments; they are Stokes-like models, not general inertial 3-D flow.
Specialized ALEX paths use separate finite-volume momentum and pressure
operators. Their reduced verification and open production-acceptance gates
are distinguished in the [validation matrix](../validation/index.md).

No-slip walls set velocity to zero. Insulating walls impose zero normal
current. Conducting solid regions solve potential with their own conductivity
and enforce potential and normal-current continuity at interfaces. Pipe meshes
use cylindrical/mapped electric metrics; rectangular electric operators use
nonuniform Cartesian finite-volume metrics. Generic momentum uses a distinct
discretization: electric closure alone does not establish nonuniform or
cylindrical vector-momentum consistency.

Common dimensionless groups are

$$
Ha=BL\sqrt{\frac{\sigma}{\rho\nu}},\qquad
Re=\frac{UL}{\nu},\qquad
N=\frac{Ha^2}{Re},\qquad
Rm=\mu_0\sigma UL.
$$

The inductionless model requires $Rm\ll1$. Geometry length, field orientation,
wall conductance ratio, and velocity scale must accompany any reported value.

(pressure-taps-and-mechanical-work)=
## Pressure taps and mechanical work

For outward normal $\mathbf n$, pressure supplies power
$-\int_{\partial V}p\mathbf u\cdot\mathbf n\,dA$ to a control volume;
see the [MIT integral energy derivation](https://ocw.mit.edu/courses/16-01-unified-engineering-i-ii-iii-iv-fall-2005-spring-2006/017b07723e0687d3025ba25a4f5d50ee_f12_sp.pdf).
The extruded reducer's `pressure_tap_flux_power` evaluates only the two
cell-center tap-plane contributions with fluid-area quadrature:

$$
P_{\rm taps}=\sum_i A_i(p_{0i}u_{0i}-p_{Ni}u_{Ni}).
$$

It differs from `pumping_power`, $(\bar p_0-\bar p_N)Q_N$, for correlated
nonuniform pressure/velocity profiles or unequal station flows. A pressure
gauge shift $p\mapsto p+c$ changes $P_{\rm taps}$ by $c(Q_0-Q_N)$; check mass
balance before interpreting it. Both quantities have power units when inputs
are SI, but neither is certified total pump power.

A complete mechanical balance also needs body-drive work
$\int_V\mathbf f\cdot\mathbf u\,dV$, kinetic-energy storage, viscous and
electrical work, and any advective boundary flux of the chosen model. Use
one common control volume: whole-domain body work cannot simply be added to
cell-center tap work. A prescribed pressure gradient represented as body drive
must not also be counted as boundary pressure work. Fixed-flow constraints
require their imposed-drive work. These terms and a discrete balance are not
yet supplied by this reducer; the generic recurrence is not energy certified.

## Quasi-two-dimensional model

For a strong uniform field normal to a shallow flow plane, the basic
Sommeria--Moreau closure evolves depth-averaged vorticity:

$$
\frac{\partial\omega}{\partial t}
+\mathbf u_\perp\cdot\nabla_\perp\omega
=\nu\nabla_\perp^2\omega-\frac{\omega}{\tau_H}+f_\omega,
$$

$$
\mathbf u_\perp=(\partial_y\psi,-\partial_x\psi),\qquad
-\nabla_\perp^2\psi=\omega.
$$

Here $1/\tau_H$ is the linear Hartmann-layer friction and $f_\omega$ is a
vorticity source. On a periodic domain the velocity is divergence free by
construction. With kinetic energy $E=\langle|\mathbf u_\perp|^2\rangle/2$ and
enstrophy $Z=\langle\omega^2\rangle/2$,

$$
\frac{dE}{dt}=-2\nu Z-\frac{2E}{\tau_H}+\langle\psi f_\omega\rangle.
$$

LMX reports a numerical defect for this identity; it is not an enforced
terminal acceptance criterion in `solve_q2d`. A `completed` status must not
substitute for checking the energy budget and mesh/time refinement.
The Q2D model is a
depth-averaged strong-field approximation; it does not resolve Hartmann-layer
profiles or replace the 3-D formulation in fringing regions.
