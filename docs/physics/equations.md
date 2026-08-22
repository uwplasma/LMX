# Equations and assumptions

LMX solves incompressible, inductionless liquid-metal MHD. The magnetic
Reynolds number is assumed small, so the imposed magnetic field $\mathbf B$ is
not evolved. With density $\rho$, kinematic viscosity $\nu$, conductivity
$\sigma$, pressure $p$, and body drive $\mathbf f$:

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
cross-section equations. The extruded model retains all three velocity,
current, and Lorentz-force components and enforces incompressibility by a
face-flux pressure projection.

No-slip walls set velocity to zero. Insulating walls impose zero normal
current. Conducting solid regions solve potential with their own conductivity
and enforce potential and normal-current continuity at interfaces. Pipe meshes
use cylindrical/mapped metrics; rectangular meshes use nonuniform Cartesian
finite-volume metrics.

Common dimensionless groups are

$$
Ha=BL\sqrt{\frac{\sigma}{\rho\nu}},\qquad
Re=\frac{UL}{\nu},\qquad
N=\frac{Ha^2}{Re},\qquad
Rm=\mu_0\sigma UL.
$$

The inductionless model requires $Rm\ll1$. Geometry length, field orientation,
wall conductance ratio, and velocity scale must accompany any reported value.

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

LMX evaluates this identity as a terminal numerical gate. The Q2D model is a
depth-averaged strong-field approximation; it does not resolve Hartmann-layer
profiles or replace the 3-D formulation in fringing regions.
