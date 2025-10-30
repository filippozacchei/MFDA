import os
import numpy as np
import tqdm.autonotebook
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.fem import (
    Function, functionspace, Constant,
    dirichletbc, form, locate_dofs_topological
)
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector, apply_lifting,
    create_vector, set_bc
)
from dolfinx.io import gmshio
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from ufl import TrialFunction, TestFunction, inner, grad, dx
import gmsh
from pathlib import Path


# ============================================================
# Geometry builder (parametric height)
# ============================================================
def build_bfs_mesh_lf(h2=0.20, h1=0.10, L_up=0.10, L_down=0.40, lc_min=0.0005, lc_max=0.01):
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


# ============================================================
# Potential flow solver
# ============================================================
def run_lf_case(H_in=0.10, 
                U_in=1.5):
    """
    Potential flow in a backward-facing-step geometry
    - Input: channel height h2
    - Output: outlet velocity profile
    """
    try:
        # --------------------------
        # Geometry and mesh
        # --------------------------
        mesh, ft, L, H = build_bfs_mesh_lf(h1=H_in)  # pass the uncertain parameter here

    except Exception as e:
        # Geometry or meshing failed — return NaN profile
        print(f"[HF mesh error] H_in={H_in:.4f}, U_in={U_in:.4f} -> {e}")
        return np.nan, np.nan
    Vphi = functionspace(mesh, ("CG", 2))
    v_cg1 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    Vvec = functionspace(mesh, v_cg1)
    fdim = mesh.topology.dim - 1
    inlet_marker, outlet_marker, wall_marker = 2, 3, 4

    # BCs for potential
    Phi_in, Phi_out = 0.0, U_in * L
    phi_inlet = Function(Vphi); phi_inlet.x.array[:] = Phi_in
    phi_outlet = Function(Vphi); phi_outlet.x.array[:] = Phi_out

    inlet_dofs = locate_dofs_topological(Vphi, fdim, ft.find(inlet_marker))
    outlet_dofs = locate_dofs_topological(Vphi, fdim, ft.find(outlet_marker))
    bcs = [
        dirichletbc(phi_inlet, inlet_dofs),
        dirichletbc(phi_outlet, outlet_dofs)
    ]

    # Laplace problem
    phi = TrialFunction(Vphi)
    psi = TestFunction(Vphi)
    a_form = inner(grad(phi), grad(psi)) * dx
    rhs_form = Constant(mesh, PETSc.ScalarType(0.0)) * psi * dx

    A = assemble_matrix(form(a_form), bcs=bcs); A.assemble()
    b = create_vector(form(rhs_form))
    with b.localForm() as blf: blf.set(0.0)
    assemble_vector(b, form(rhs_form))
    apply_lifting(b, [form(a_form)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    Phi = Function(Vphi, name="Phi")
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12)
    solver.solve(b, Phi.x.petsc_vec)
    Phi.x.scatter_forward()

    # Project velocity = ∇Φ
    u = Function(Vvec, name="u")
    u_t, v_t = TrialFunction(Vvec), TestFunction(Vvec)
    Aproj = assemble_matrix(form(inner(u_t, v_t)*dx)); Aproj.assemble()
    bproj = create_vector(form(inner(grad(Phi), v_t)*dx))
    with bproj.localForm() as blf: blf.set(0.0)
    assemble_vector(bproj, form(inner(grad(Phi), v_t)*dx))

    kspP = PETSc.KSP().create(mesh.comm)
    kspP.setOperators(Aproj)
    kspP.setType(PETSc.KSP.Type.CG)
    kspP.getPC().setType(PETSc.PC.Type.JACOBI)
    kspP.setTolerances(rtol=1e-12, atol=1e-14)
    kspP.solve(bproj, u.x.petsc_vec)
    u.x.scatter_forward()

    # Outlet profile
    ny = 100
    y_vals = np.linspace(0.0, H, ny)
    pts_out = np.column_stack([np.full_like(y_vals, L), y_vals, np.zeros_like(y_vals)])
    tree = bb_tree(mesh, mesh.geometry.dim)
    cand = compute_collisions_points(tree, pts_out)
    coll = compute_colliding_cells(mesh, cand, pts_out)

    uprof = np.zeros_like(y_vals)
    for i in range(ny):
        links = coll.links(i)
        if len(links) > 0:
            uu = u.eval(pts_out[i], links[:1])
            uprof[i] = uu[0]

    return y_vals, uprof


# ============================================================
# Forward model for UQ
# ============================================================
def forward_model(h,u=1.5):
    y, u = run_lf_case(H_in=h,U_in=u)
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
    
    U_in = [1.0, 1.25, 1.5, 1.75, 2.0]
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
