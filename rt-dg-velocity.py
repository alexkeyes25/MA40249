from firedrake import *
from firedrake.petsc import PETSc
print = lambda x: PETSc.Sys.Print(x)
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':22})
deg_max = 3

n_vals = [4,8,16,32,64]

norm_sol = [ [] for i in range(deg_max)]
norm_hdiv  = [ [] for i in range(deg_max)]
norm_pres = [ [] for i in range(deg_max)]
degs = [i + 1 for i in range(deg_max)]
fig, axs = plt.subplots(nrows=1, ncols=deg_max)
fig.supxlabel('Mesh size (h = 1/n)')
fig.supylabel('Error (log scale)')

for deg in degs:
	for N in n_vals:
		mesh = UnitSquareMesh(N, N)
		v_q = 3
		w_q = 2
		V = FunctionSpace(mesh, "RT", deg+1) # velocity
		W = FunctionSpace(mesh, "DG", deg) # pressure
		C = FunctionSpace(mesh, "R", 0)  # extra
		Z = V * W *C
		#print(Z.dim())

		z = Function(Z)
		(u, p, c) = split(z)

		w = TestFunction(Z)
		(v, q, d) = split(w)

		n = FacetNormal(mesh)
		t = as_vector([-n[1], n[0]])

		(x, y) = SpatialCoordinate(mesh)
		h = FacetArea(mesh)

		u_ex = as_vector([sin(pi * x) * cos(pi * y),
		                -cos(pi * x) * sin(pi * y)])
		p_ex = (x*y*(1-x)*(1-y))**3
		p_ex -= assemble(p_ex*dx)

		f = -div(grad(u_ex)) + grad(p_ex)
		alpha = 40

		u_t = n[0]*u[1] - n[1]*u[0]

		u_ex_t = n[0]*u_ex[1] - n[1]*u_ex[0]
		v_t = n[0]*v[1] - n[1]*v[0]

		F = (
		      inner(grad(u),grad(v)) * dx
		    - inner(p, div(v))*dx
		    + inner(q, div(u))*dx
		    - inner(f, v)*dx
		    - inner(avg(inner(grad(inner(u, t)), n)), 2*avg(v_t)) * dS
		    - inner(avg(inner(grad(inner(v, t)), n)), 2*avg(u_t)) * dS
		    + 40/h * inner(avg(u_t), avg(v_t)) * dS

		    - inner(inner(grad(inner(u, t)), n), v_t) * ds
		    - inner(inner(grad(inner(v, t)), n), u_t - u_ex_t) * ds
		    + 40/h * inner(u_t - u_ex_t, v_t) * ds
		    + inner(c, q)*dx
		    + inner(d, p)*dx
		)


		bcs = DirichletBC(Z.sub(0), u_ex, 'on_boundary')

		sp = {
		         "mat_type": "matfree",
		        "snes_type": "newtonls",
		        "snes_monitor": None,
		        "snes_converged_reason": None,
		        "snes_linesearch_type": "basic",
		        "ksp_type": "fgmres",
		        "ksp_monitor_true_residual": None,
		        "ksp_max_it": 10,
		        "pc_type": "fieldsplit",
		        "pc_fieldsplit_type": "schur",
		        "pc_fieldsplit_schur_fact_type": "full",
		        "pc_fieldsplit_0_fields": "0,1",
		        "pc_fieldsplit_1_fields": "2",
		        "fieldsplit_0_ksp_type": "preonly",
		        "fieldsplit_0_pc_type": "python",
		        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
		        "fieldsplit_0_assembled_pc_type": "lu",
		        "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
		        "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
		        "mat_mumps_icntl_14": 200,
		        "fieldsplit_1_ksp_type": "gmres",
		        "fieldsplit_1_ksp_monitor_true_residual": None,
		        "fieldsplit_1_ksp_max_it": 1,
		        "fieldsplit_1_ksp_convergence_test": "skip",
		        "fieldsplit_1_pc_type": "none",}

		nlvp = NonlinearVariationalProblem(F, z,  bcs = bcs)
		solver = NonlinearVariationalSolver(nlvp ,solver_parameters=sp)
		solver.solve()
		sol_norm = norm(z.sub(0)-u_ex)
		div_norm = norm(div(z.sub(0)-u_ex))
		hdiv_norm = sqrt((sol_norm)**2 + (div_norm)**2)
		pres_norm = norm(z.sub(1) - p_ex)
		print(f'FEM is RT{deg+1} - DG{deg}')
		print(f'n = {N}')
		print(f'L2 velocity = {sol_norm}')
		print(f'H(div) velocity = {hdiv_norm}')
		print(f'L2 pressure = {pres_norm}')
		print(f'DOF velocity = {V.dim()}')
		print(f'DOF pressure = {W.dim()}')
		print(f'DOF pressure extra = {C.dim()}')
		print(f'DOF total = {Z.dim()}')
		norm_sol[deg-1].append(sol_norm)
		norm_hdiv[deg-1].append(hdiv_norm)
		norm_pres[deg-1].append(pres_norm)
# H1 plot - velocity
plt.subplot(1,3,1)

# first reference line
x0 = norm_hdiv[0][0]
y0 = n_vals[0]
r_line = 0.2 * x0 * (np.asarray(n_vals)/y0)**-2
plt.loglog(n_vals, r_line, linestyle='--' , label='O(h^2)', color='black')
# second reference line
x1 = norm_hdiv[1][0]
r_line2 = 0.2 * x1 * (np.asarray(n_vals)/y0)**-3
plt.loglog(n_vals, r_line2, linestyle='-.', label='O(h^3)', color='black')
# third reference line
x2 = norm_hdiv[2][0]
r_line3 = 0.2*x2*(np.asarray(n_vals)/y0)**-4
plt.loglog(n_vals, r_line3, linestyle=':', label='O(h^4)', color='black')

# Error plots
plt.loglog(n_vals, norm_hdiv[0], marker='o') #, label='CG2-CG1')
plt.loglog(n_vals, norm_hdiv[1], marker='s') #, label='CG3-CG2')
plt.loglog(n_vals, norm_hdiv[2], marker='^') #, label='CG4-CG3')
plt.title('Velocity - H(div)')
plt.ylim(1e-9, 1e1)
plt.xticks(n_vals, labels=["1/"+str(n) for n in n_vals])
plt.legend()

# L2 plot - velocity
plt.subplot(1,3,2)

# first reference line
x0 = norm_sol[0][0]
y0 = n_vals[0]
r_line = 0.2 * x0 * (np.asarray(n_vals)/y0)**-2
plt.loglog(n_vals, r_line, linestyle='--' , label='O(h^2)', color='black')
# second reference line
x1 = norm_sol[1][0]
r_line2 = 0.2 * x1 * (np.asarray(n_vals)/y0)**-3
plt.loglog(n_vals, r_line2, linestyle='-.', label='O(h^3)', color='black')
# third reference line
x2 = norm_sol[2][0]
r_line3 = 0.2*x2*(np.asarray(n_vals)/y0)**-4
plt.loglog(n_vals, r_line3, linestyle=':', label='O(h^4)', color='black')

# Error plots
plt.loglog(n_vals, norm_sol[0], marker='o') #, label='CG2-CG1')
plt.loglog(n_vals, norm_sol[1], marker='s') #, label='CG3-CG2')
plt.loglog(n_vals, norm_sol[2], marker='^') #, label='CG4-CG3')
plt.title('Velocity - L2')
plt.ylim(1e-9, 1e1)
plt.xticks(n_vals, labels=["1/"+str(n) for n in n_vals])
plt.yticks(visible=False)
plt.legend()

# L2 plot - pressure
plt.subplot(1,3,3)

# first reference line
x0 = norm_pres[0][0]
y0 = n_vals[0]
r_line = 0.2 * x0 * (np.asarray(n_vals)/y0)**-1
plt.loglog(n_vals, r_line, linestyle='--' , label='O(h^1)', color='black')
# second reference line
x1 = norm_pres[1][0]
r_line2 = 0.2 * x1 * (np.asarray(n_vals)/y0)**-2
plt.loglog(n_vals, r_line2, linestyle='-.', label='O(h^2)', color='black')
# third reference line
x2 = norm_pres[2][0]
r_line3 = 0.2*x2*(np.asarray(n_vals)/y0)**-3
plt.loglog(n_vals, r_line3, linestyle=':', label='O(h^3)', color='black')

plt.loglog(n_vals, norm_pres[0], marker='o', label='RT2-DG1')
plt.loglog(n_vals, norm_pres[1], marker='s', label='RT3-DG2')
plt.loglog(n_vals, norm_pres[2], marker='^', label='RT4-DG3')
plt.title('Pressure - L2')
plt.legend()
plt.ylim(1e-9, 1e1)
plt.xticks(n_vals, labels=["1/"+str(n) for n in n_vals])
plt.yticks(visible=False)
plt.show()

