from firedrake import *
from petsc4py import PETSc
print = lambda x: PETSc.Sys.Print(x)
from firedrake.__future__ import interpolate
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':22})
deg_max=3
n_vals = [4,8,16,32,64]

norm_sol = [ [] for i in range(deg_max)]
norm_h1  = [ [] for i in range(deg_max)]
norm_pres = [ [] for i in range(deg_max)]
degs = [i + 1 for i in range(deg_max)]

fig, axs = plt.subplots(nrows=1, ncols=deg_max)
fig.supxlabel('Mesh size (h = 1/n)')
fig.supylabel('Error (log scale)')

for deg in degs:
	for N in n_vals:
		mesh = UnitSquareMesh(N,N)
		V = VectorFunctionSpace(mesh, 'CG', deg+1) # Space for velocity vectors
		Q = FunctionSpace(mesh, 'CG', deg) # Space for pressure function
		W = V*Q

		z = TrialFunction(W)
		(u, p) = split(z)

		#u = Trial function for velocity
		#p = Trial function for pressure

		w = TestFunction(W)
		(v, q) = split(w)

		#v = Test function for velocity
		#q = Test function for pressure

		# Deciding a forcing function f
		(x,y) = SpatialCoordinate(mesh)
		u_ex = as_vector([sin(pi * x) * cos(pi * y),
		                -cos(pi * x) * sin(pi * y)])
		p_ex = (x*y*(1-x)*(1-y))**3
		f = -div(grad(u_ex)) + grad(p_ex)

		# Writing bilinear & linear forms
		a = (
		    inner(grad(u), grad(v)) * dx
		  - inner(p, div(v)) * dx
		  + inner(q, div(u)) * dx
		)

		L  = inner(f,v) * dx


		# Defining boundary conditions on FUNCTIONS on the first element of
		# W (V, i.e. velocity function)

		bc = [DirichletBC(W.sub(0), u_ex, 'on_boundary')]

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

		u = Function(W)
		solve(a == L, u, solver_parameters = {'ksp_monitor': None,}, bcs = bc)

		# Finding error norms on velocity
		sol_norm = norm(u.sub(0)-u_ex)
		grad_norm = norm(grad(u.sub(0)-u_ex))
		h1_norm = sqrt((sol_norm)**2 + (grad_norm)**2)

		# normalised calculated pressure
		p_h_corr = u.sub(1) - assemble(u.sub(1)*dx)
		# normalised exact pressure
		p_ex_corr = p_ex - assemble(p_ex * dx) 
		pres_norm = norm(p_h_corr - p_ex_corr)

		print(f'FEM is CG{deg+1} - CG{deg}')
		print(f'n = {N}')
		print(f'L2 velocity = {sol_norm}')
		print(f'H1 velocity = {h1_norm}')
		print(f'L2 pressure = {pres_norm}')
		print(f'DOF velocity = {V.dim()}')
		print(f'DOF pressure = {Q.dim()}')
		norm_sol[deg-1].append(sol_norm)
		norm_h1[deg-1].append(h1_norm)
		norm_pres[deg-1].append(pres_norm)
# H1 plot - velocity
plt.subplot(1,3,1)
# first reference line
x0 = norm_h1[0][0]
y0 = n_vals[0]
r_line = 0.2 * x0 * (np.asarray(n_vals)/y0)**-2
plt.loglog(n_vals, r_line, linestyle='--' , label='O(h^2)', color='black')
# second reference line
x1 = norm_h1[1][0]
r_line2 = 0.2 * x1 * (np.asarray(n_vals)/y0)**-3
plt.loglog(n_vals, r_line2, linestyle='-.', label='O(h^3)', color='black')
# third reference line
x2 = norm_h1[2][0]
r_line3 = 0.2*x2*(np.asarray(n_vals)/y0)**-4
plt.loglog(n_vals, r_line3, linestyle=':', label='O(h^4)', color='black')

# Error plots
plt.loglog(n_vals, norm_h1[0], marker='o') #label='CG2-CG1')
plt.loglog(n_vals, norm_h1[1], marker='s') #label='CG3-CG2')
plt.loglog(n_vals, norm_h1[2], marker='^') #label='CG4-CG3')
plt.title('Velocity - H1')
plt.ylim(1e-11, 1e1)
plt.xticks(n_vals, labels=["1/"+str(n) for n in n_vals])
plt.legend()


# L2 plot - velocity
plt.subplot(1,3,2)

# first reference line
x0 = norm_sol[0][0]
y0 = n_vals[0]
r_line = 0.2 * x0 * (np.asarray(n_vals)/y0)**-3
plt.loglog(n_vals, r_line, linestyle='--' , label='O(h^3)', color='black')
# second reference line
x1 = norm_sol[1][0]
r_line2 = 0.2 * x1 * (np.asarray(n_vals)/y0)**-4
plt.loglog(n_vals, r_line2, linestyle='-.', label='O(h^4)', color='black')
# third reference line
x2 = norm_sol[2][0]
r_line3 = 0.2*x2*(np.asarray(n_vals)/y0)**-5
plt.loglog(n_vals, r_line3, linestyle=':', label='O(h^5)', color='black')

# Error plots
plt.loglog(n_vals, norm_sol[0], marker='o')  #, label='CG2-CG1')
plt.loglog(n_vals, norm_sol[1], marker='s')  #, label='CG3-CG2')
plt.loglog(n_vals, norm_sol[2], marker='^')  #, label='CG4-CG3')
plt.title('Velocity - L2')
plt.ylim(1e-11, 1e1)
plt.xticks(n_vals, labels=["1/"+str(n) for n in n_vals])
plt.yticks(visible=False)
plt.legend()

# L2 plot - pressure
plt.subplot(1,3,3)

# first reference line
x0 = norm_pres[0][0]
y0 = n_vals[0]
r_line = 0.2 * x0 * (np.asarray(n_vals)/y0)**-3
plt.loglog(n_vals, r_line, linestyle='--' , label='O(h^3)', color='black')
# second reference line
x1 = norm_pres[1][0]
r_line2 = 0.2 * x1 * (np.asarray(n_vals)/y0)**-4
plt.loglog(n_vals, r_line2, linestyle='-.', label='O(h^4)', color='black')
# third reference line
x2 = norm_pres[2][0]
r_line3 = 0.2*x2*(np.asarray(n_vals)/y0)**-5
plt.loglog(n_vals, r_line3, linestyle=':', label='O(h^5)', color='black')


plt.loglog(n_vals, norm_pres[0], marker='o', label='CG2-CG1')
plt.loglog(n_vals, norm_pres[1], marker='s', label='CG3-CG2')
plt.loglog(n_vals, norm_pres[2], marker='^', label='CG4-CG3')
plt.title('Pressure - L2')
plt.legend()
plt.ylim(1e-11, 1e1)
plt.xticks(n_vals, labels=["1/"+str(n) for n in n_vals])
plt.yticks(visible=False)
plt.show()
