"""Hydrodynamic damping model (theta_anneal identification), shared by both
vessels. Pure function of body-frame velocities (u,v,r) -> damping
forces/moment and the resulting (control-independent) body acceleration."""
from config import M11, M22, M33, M23, M32, DELTA, DAMP_SIGN

# theta_anneal damping (dimensional), identical for both vessels
TA = dict(
    Dx_u=28.5721, Dx_u2=-15.982, Dx_v2=73.3826, Dx_r2=20.4587, Dx_vr=16.4984,
    Dx_u3=55.7021, Dx_uv2=-36.2765, Dx_ur2=-44.7408, Dx_uvr=-82.1422, Dx_absu_u=-44.2545,
    Dy_v=32.837, Dy_r=-6.80627, Dy_uv=-28.4202, Dy_ur=-16.6802, Dy_v3=66.2334,
    Dy_r3=8.26366, Dy_u2v=2.75094, Dy_u2r=-1.0423, Dy_v2r=148.983, Dy_vr2=-4.43673,
    Dy_absv_v=-8.63508, Dy_u_absv=5.81447,
    Dn_v=6.74496, Dn_r=0.974895, Dn_uv=-11.1648, Dn_ur=-9.47937, Dn_v3=43.9947,
    Dn_r3=16.8805, Dn_u2v=13.2957, Dn_u2r=9.21704, Dn_v2r=62.4145, Dn_vr2=-8.60609,
    Dn_absr_r=-6.27807, Dn_u_absr=-0.209728,
)


def damping(u, v, r, c=TA):
    au, av, ar = u.abs(), v.abs(), r.abs()
    X = (c["Dx_u"]*u + c["Dx_u2"]*u*u + c["Dx_v2"]*v*v + c["Dx_r2"]*r*r + c["Dx_vr"]*v*r
         + c["Dx_u3"]*u**3 + c["Dx_uv2"]*u*v*v + c["Dx_ur2"]*u*r*r + c["Dx_uvr"]*u*v*r
         + c["Dx_absu_u"]*au*u)
    Y = (c["Dy_v"]*v + c["Dy_r"]*r + c["Dy_uv"]*u*v + c["Dy_ur"]*u*r + c["Dy_v3"]*v**3
         + c["Dy_r3"]*r**3 + c["Dy_u2v"]*u*u*v + c["Dy_u2r"]*u*u*r + c["Dy_v2r"]*v*v*r
         + c["Dy_vr2"]*v*r*r + c["Dy_absv_v"]*av*v + c["Dy_u_absv"]*u*av)
    N = (c["Dn_v"]*v + c["Dn_r"]*r + c["Dn_uv"]*u*v + c["Dn_ur"]*u*r + c["Dn_v3"]*v**3
         + c["Dn_r3"]*r**3 + c["Dn_u2v"]*u*u*v + c["Dn_u2r"]*u*u*r + c["Dn_v2r"]*v*v*r
         + c["Dn_vr2"]*v*r*r + c["Dn_absr_r"]*ar*r + c["Dn_u_absr"]*u*ar)
    return X, Y, N


def nu_dot0(u, v, r):
    """Control-independent body acceleration (tau=0), coupled sway-yaw.
    Solves M v_dot = -C(v)v - D(v)v with the same coupled M and
    skew-symmetric Coriolis construction as src/dynamics.py."""
    X, Y, N = damping(u, v, r, TA)
    a = M22 * v + M23 * r
    b_u = DAMP_SIGN * X + a * r
    b_v = DAMP_SIGN * Y - M11 * u * r
    b_r = DAMP_SIGN * N - a * u + M11 * u * v
    du = b_u / M11
    dv = (M33 * b_v - M23 * b_r) / DELTA
    dr = (-M32 * b_v + M22 * b_r) / DELTA
    return du, dv, dr
