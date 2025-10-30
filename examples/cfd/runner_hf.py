import os
import numpy as np
import tqdm.autonotebook
import matplotlib.pyplot as plt
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.fem import (
    Constant, Function, functionspace, dirichletbc, form,
    locate_dofs_topological, assemble_scalar
)
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector, apply_lifting,
    create_vector, create_matrix, set_bc
)
from dolfinx.io import gmshio, VTXWriter
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from ufl import (
    TrialFunction, TestFunction, inner, dot, grad, div, nabla_grad,
    dx, FacetNormal, Measure, lhs, rhs, sym, CellDiameter, exp
)
import gmsh

def build_bfs_mesh_hf(h2=0.20, h1=0.10, L_up=0.10, L_down=0.40, lc_min=0.0005, lc_max=0.01, tol=1e-5):
    """
    Build backward-facing step geometry and mesh with specified downstream height h2.
    """
    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    # clip to avoid degeneracy
    h1 = max(h1, tol)
    h2 = max(h2, tol)
    L_up = max(L_up, 5 * tol)
    L_down = max(L_down, 5 * tol)
    
    if mesh_comm.rank == model_rank:
        L = L_up + L_down
        up = gmsh.model.occ.addRectangle(0.0, 0.0, 0, L_up, h1)
        down = gmsh.model.occ.addRectangle(L_up, 0.0, 0, L_down, h2)
        gmsh.model.occ.fuse([(2, up)], [(2, down)])
        gmsh.model.occ.synchronize()

        surfs = [s[1] for s in gmsh.model.occ.getEntities(dim=2)]
        gmsh.model.addPhysicalGroup(2, surfs, 1)
        gmsh.model.setPhysicalName(2, 1, "Fluid")

        inlet_marker, outlet_marker, wall_marker = 2, 3, 4
        bnds = gmsh.model.getBoundary([(2, s) for s in surfs], oriented=False)
        inflow, outflow, walls = [], [], []
        for (dim, tag) in bnds:
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.isclose(cx, 0.0, atol=1e-10):
                inflow.append(tag)
            elif np.isclose(cx, L, atol=1e-10):
                outflow.append(tag)
            else:
                walls.append(tag)

        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")

    mesh, cell_tags, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim)
    if mesh_comm.rank == model_rank:
        gmsh.finalize()

    return mesh, ft, L_up + L_down, h2

def run_hf_case(H_in=0.10, 
                U_in=1.5, 
                nu0_val=1e-3, 
                a_vis_val=0.5, 
                beta_val=3e-3):
    """
    Buoyant incompressible Navier–Stokes with temperature-dependent viscosity
    in a rectangular constriction.

    Returns:
        y_out (1D array): vertical coordinates at outlet
        u_out (1D array): streamwise velocity profile at outlet
    """
    # --------------------------
    # Controls
    # --------------------------
    DT = 0.001
    TMAX = 2
    rho_v = 1.0

    # Thermal parameters
    T_HOT, T_COLD = 10.0, 0.0
    CP, K_TH, ALPHA = 1.0, 0.01, -1.0

    try:
        # --------------------------
        # Geometry and mesh
        # --------------------------
        mesh, ft, L, H = build_bfs_mesh_hf(h1=H_in)  # pass the uncertain parameter here

    except Exception as e:
        # Geometry or meshing failed — return NaN profile
        print(f"[HF mesh error] H_in={H_in:.4f}, U_in={U_in:.4f} -> {e}")
        return np.nan, np.nan
    # --------------------------
    # Function spaces and constants
    # --------------------------
    dt = DT
    Tend  = TMAX
    num_steps = int(Tend/dt)
    k  = Constant(mesh, PETSc.ScalarType(dt))
    rho = Constant(mesh, PETSc.ScalarType(rho_v))

    # Thermal coefficients
    alpha_val = (K_TH/(rho_v*CP)) if ALPHA < 0 else ALPHA
    alpha = Constant(mesh, PETSc.ScalarType(alpha_val))

    v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
    V = functionspace(mesh, v_cg2)  # velocity
    Q = functionspace(mesh, s_cg1)  # pressure
    QT = Q                           # temperature space (CG1)
    fdim = mesh.topology.dim - 1

    # --------------------------
    # 3) Boundary conditions (velocity + temperature)
    # --------------------------
    class InletVelocity:
        def __call__(self, x):
            vals = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            vals[0] = U_in #4 * U_in * x[1] * (H - x[1]) / (H ** 2)
            return vals

    inlet_marker, outlet_marker, wall_marker = 2, 3, 4

    # Velocity BCs
    u_inlet = Function(V); u_inlet.interpolate(InletVelocity())
    bcu_in = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
    u_zero = np.array((0.0, 0.0), dtype=PETSc.ScalarType)
    bcu_w  = dirichletbc(u_zero, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
    bcu = [bcu_in, bcu_w]
    bcp_out = dirichletbc(PETSc.ScalarType(0.0),
                        locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
    bcp = [bcp_out]

    # Temperature BCs
    T_hot = Function(QT);  T_hot.x.array[:] = T_HOT
    T_cold = Function(QT); T_cold.x.array[:] = T_COLD
    bcT_in  = dirichletbc(T_hot,  locate_dofs_topological(QT, fdim, ft.find(inlet_marker)))
    bcT_wall= dirichletbc(T_cold, locate_dofs_topological(QT, fdim, ft.find(wall_marker)))
    bcT = [bcT_in, bcT_wall]      # outlet: natural

    # --------------------------
    # Unknowns
    # --------------------------
    u = TrialFunction(V); v = TestFunction(V)
    p = TrialFunction(Q); q = TestFunction(Q)
    Tt= TrialFunction(QT); w = TestFunction(QT)

    u_  = Function(V, name="u")            # corrected vel
    u_s = Function(V, name="u_tentative")  # tentative
    u_n = Function(V, name="u_n")
    u_nm1 = Function(V, name="u_nm1")
    p_  = Function(Q, name="p")
    phi = Function(Q, name="phi")

    T_  = Function(QT, name="T")
    T_n = Function(QT, name="T_n")

    nu0 = Constant(mesh, PETSc.ScalarType(nu0_val))
    a_vis = Constant(mesh, PETSc.ScalarType(a_vis_val))
    beta = Constant(mesh, PETSc.ScalarType(beta_val))
    T_ref = Constant(mesh, PETSc.ScalarType(T_HOT/2))
    g_vec = Constant(mesh, PETSc.ScalarType((0.0, -9.81)))

    def nu_of_T(Tf):
        return nu0 * exp(-a_vis * (Tf - T_ref))

    mu_eff = rho * nu_of_T(T_n)

    # --------------------------
    # Variational forms
    # --------------------------
    F1  = rho/k * dot(u - u_n, v) * dx
    F1 += inner(dot(1.5*u_n - 0.5*u_nm1, 0.5*nabla_grad(u + u_n)), v) * dx
    F1 += inner(mu_eff * grad(0.5*(u + u_n)), grad(v)) * dx
    F1 -= dot(p_, div(v)) * dx
    F1 += -rho * beta * (T_n - T_ref) * dot(g_vec, v) * dx

    a1, L1 = form(lhs(F1)), form(rhs(F1))
    A1 = create_matrix(a1); b1 = create_vector(L1)

    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho/k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp); A2.assemble()
    b2 = create_vector(L2)

    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3); A3.assemble()
    b3 = create_vector(L3)

    aT = form((1.0/k)*Tt*w*dx + dot(u_, grad(Tt))*w*dx + alpha*dot(grad(Tt), grad(w))*dx)
    LT = form((1.0/k)*T_n*w*dx)
    AT = assemble_matrix(aT, bcs=bcT); AT.assemble()
    bT = create_vector(LT)

    # --------------------------
    # Solvers (identical setup)
    # --------------------------
    solver1 = PETSc.KSP().create(mesh.comm); solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS); solver1.getPC().setType(PETSc.PC.Type.JACOBI)

    solver2 = PETSc.KSP().create(mesh.comm); solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC(); pc2.setType(PETSc.PC.Type.HYPRE)
    try: pc2.setHYPREType("boomeramg")
    except: pc2.setType(PETSc.PC.Type.JACOBI)

    solver3 = PETSc.KSP().create(mesh.comm); solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG); solver3.getPC().setType(PETSc.PC.Type.SOR)

    solverT = PETSc.KSP().create(mesh.comm); solverT.setOperators(AT)
    solverT.setType(PETSc.KSP.Type.GMRES); solverT.getPC().setType(PETSc.PC.Type.ILU)

    # --------------------------
    # Initial conditions
    # --------------------------
    T_n.interpolate(lambda x: np.full(x.shape[1], T_HOT, dtype=PETSc.ScalarType))

    # --------------------------
    # Time loop
    # --------------------------
    t = 0.0
    progress = tqdm.autonotebook.tqdm(total=num_steps, desc="DNS constriction")

    for nstep in range(num_steps):
        t += dt; progress.update(1)

        # Step 1: Tentative velocity
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu); A1.assemble()
        with b1.localForm() as loc: loc.set(0.0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()

        # Step 2: Pressure
        with b2.localForm() as loc: loc.set(0.0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()
        p_.x.petsc_vec.axpy(1.0, phi.x.petsc_vec)
        p_.x.scatter_forward()

        # Step 3: Velocity correction
        with b3.localForm() as loc: loc.set(0.0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        # Step 4: Temperature
        AT.zeroEntries()
        assemble_matrix(AT, aT, bcs=bcT); AT.assemble()
        with bT.localForm() as loc: loc.set(0.0)
        assemble_vector(bT, LT)
        apply_lifting(bT, [aT], [bcT])
        bT.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(bT, bcT)
        solverT.solve(bT, T_.x.petsc_vec)
        T_.x.scatter_forward()

        # Rotate history
        with (u_.x.petsc_vec.localForm() as lu,
              u_n.x.petsc_vec.localForm() as lun,
              u_nm1.x.petsc_vec.localForm() as lunm1):
            lun.copy(lunm1)
            lu.copy(lun)
        T_n.x.petsc_vec.array[:] = T_.x.petsc_vec.array[:]

    # --------------------------
    # Outlet velocity profile
    # --------------------------
    ny = 100
    y_vals = np.linspace(0.0, H, ny)
    pts_out = np.column_stack([np.full_like(y_vals, L), y_vals, np.zeros_like(y_vals)])
    tree_out = bb_tree(mesh, mesh.geometry.dim)
    cand_out = compute_collisions_points(tree_out, pts_out)
    coll_out = compute_colliding_cells(mesh, cand_out, pts_out)

    uxs = np.zeros_like(y_vals)
    for i, yv in enumerate(y_vals):
        links = coll_out.links(i)
        if len(links) > 0:
            uu = u_.eval(pts_out[i], links[:1])
            uxs[i] = uu[0]
        
    # --- Compute outlet velocity profile ---
    ny_out = 100  # number of vertical sample points
    y_vals = np.linspace(0.0, H, ny_out)
    pts_out = np.column_stack([np.full_like(y_vals, L), y_vals, np.zeros_like(y_vals)])
    tree_out = bb_tree(mesh, mesh.geometry.dim)
    cand_out = compute_collisions_points(tree_out, pts_out)
    coll_out = compute_colliding_cells(mesh, cand_out, pts_out)

    uprof = np.zeros_like(y_vals)
    for i in range(len(y_vals)):
        links = coll_out.links(i)
        if len(links) > 0:
            uu = u_.eval(pts_out[i], links[:1])
            uprof[i] = uu[0]

    return y_vals, uxs

def forward_model(h,u=1.5):
    y, u = run_hf_case(H_in=h,U_in=u)
    return y, u

# ============================================================
# Driver for multiple heights
# ============================================================
if __name__ == "__main__":
    heights = [0.05, 0.08, 0.10, 0.12, 0.15]
    profiles = []

    for h in heights:
        y, u = forward_model(h)
        profiles.append((h, u))

    plt.figure()
    for h, u in profiles:
        plt.plot(u, y, label=f"h2={h:.2f}")
    plt.xlabel("uₓ (outlet)")
    plt.ylabel("y")
    plt.legend()
    plt.title("Outlet velocity profile vs channel height")
    plt.tight_layout()
    plt.show()
    
    U_in = [0.5, 0.75, 1.0, 1.25, 1.5]
    profiles = []

    for u_in in U_in:
        y, u = forward_model(h=0.1,u=u_in)
        profiles.append((u_in, u))

    plt.figure()
    for u_in, u in profiles:
        plt.plot(u, y, label=f"l_in={u_in:.2f}")
    plt.xlabel("uₓ (outlet)")
    plt.ylabel("y")
    plt.legend()
    plt.title("Outlet velocity profile vs input velocity")
    plt.tight_layout()
    plt.show()
