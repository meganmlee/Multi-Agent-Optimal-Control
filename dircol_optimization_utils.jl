using LinearAlgebra
using StaticArrays
import ForwardDiff as FD
import FiniteDiff
using Printf
using SparseArrays
import DifferentiableCollisions as dc
import MathOptInterface as MOI
import Ipopt

# Ensure these are included and paths are correct
include("fmincon.jl")
include("robot_kinematics_utils.jl") # For RobotKinematics, FK, etc.

# Helper to create indices for decision variables Z = [x_0, u_0, h_0, x_1, u_1, h_1, ..., x_{N-1}, u_{N-1}, h_{N-1}, x_N]
# N is the number of knot points in the trajectory. So, N-1 intervals.
function create_dircol_indices(nx::Int, nu::Int, N_knot_points::Int, min_time::Bool)
    num_intervals = N_knot_points - 1
    
    vars_per_interval_no_h = nx + nu
    vars_per_interval_with_h = nx + nu + 1

    x_indices = Vector{UnitRange{Int}}(undef, N_knot_points)
    u_indices = Vector{UnitRange{Int}}(undef, num_intervals)
    h_indices = min_time ? Vector{UnitRange{Int}}(undef, num_intervals) : Vector{UnitRange{Int}}(undef, 0)

    current_idx = 0
    for k = 1:num_intervals
        # State x_k-1
        x_indices[k] = current_idx .+ (1:nx)
        current_idx += nx
        
        # Control u_k-1
        u_indices[k] = current_idx .+ (1:nu)
        current_idx += nu

        if min_time
            h_indices[k] =  (current_idx + 1):(current_idx + 1)
            current_idx += 1
        end
    end
    # Final state x_N-1
    x_indices[N_knot_points] = current_idx .+ (1:nx)
    current_idx += nx
    
    nz = current_idx
    return (nx=nx, nu=nu, N=N_knot_points, N_knot_points=N_knot_points, num_intervals=num_intervals, nz=nz, x=x_indices, u=u_indices, h=h_indices, min_time=min_time)
end

function build_initial_Z(X_init_guess::Vector{Vector{Float64}}, U_init_guess::Vector{Vector{Float64}}, indices, nominal_dt::Float64)
    Z0 = zeros(Float64, indices.nz)
    for k = 1:indices.N_knot_points
        Z0[indices.x[k]] = X_init_guess[k]
    end
    for k = 1:indices.num_intervals
        Z0[indices.u[k]] = U_init_guess[k]
        if indices.min_time
            Z0[indices.h[k]] .= 1.0 # Initial guess for time scaling factor
        end
    end
    return Z0
end

function extract_trajectory_from_Z(Z_sol::Vector{Float64}, indices)
    X_sol = [Z_sol[indices.x[k]] for k = 1:indices.N_knot_points]
    U_sol = [Z_sol[indices.u[k]] for k = 1:indices.num_intervals]
    H_sol = indices.min_time ? [Z_sol[first(indices.h[k])] for k = 1:indices.num_intervals] : ones(Float64, indices.num_intervals)
    return X_sol, U_sol, H_sol
end

# --- Cost Function ---
function dircol_cost_function(Z::AbstractVector, p) # p is params_tuple
    J = 0.0
    indices = p.indices
    
    for k = 1:indices.num_intervals
        u_k = Z[indices.u[k]]
        h_k_scalar = indices.min_time ? Z[first(indices.h[k])] : 1.0
        
        # Control cost (jerks)
        current_dt_val = indices.min_time ? h_k_scalar * p.nominal_dt : p.nominal_dt
        J += 0.5 * dot(u_k, p.R_cost_matrix * u_k) * current_dt_val
        
        if indices.min_time
            J += p.time_penalty_weight * h_k_scalar * p.nominal_dt # Cost for time itself using scalar h_k_scalar
        end
    end
    # Terminal state cost (often handled by hard constraints, but can be added here if soft)
    # x_N = Z[indices.x[indices.N_knot_points]]
    # if isdefined(p, :Qf_cost_matrix) && isdefined(p, :xg_target)
    #     J += 0.5 * dot(x_N - p.xg_target, p.Qf_cost_matrix * (x_N - p.xg_target))
    # end
    return J
end

# --- Dynamics (Triple Integrator for each robot) ---
# x = [q, dq, ddq] for one robot
function single_robot_dynamics_continuous(x_robot::AbstractVector, u_robot_jerk::AbstractVector, NJ::Int)
    dq = x_robot[NJ+1 : 2*NJ]
    ddq = x_robot[2*NJ+1 : 3*NJ]
    return vcat(dq, ddq, u_robot_jerk)
end

function combined_robot_dynamics_continuous(x_combined::AbstractVector, u_combined_jerk::AbstractVector, p_dynamics)
    # x_combined = [x_robot1; x_robot2], u_combined_jerk = [u_robot1; u_robot2]
    NJ = p_dynamics.NJ
    NX_PER_ROBOT = 3 * NJ
    NU_PER_ROBOT = NJ

    x_robot1 = x_combined[1:NX_PER_ROBOT]
    u_robot1 = u_combined_jerk[1:NU_PER_ROBOT]
    xdot1 = single_robot_dynamics_continuous(x_robot1, u_robot1, NJ)

    x_robot2 = x_combined[NX_PER_ROBOT+1 : 2*NX_PER_ROBOT]
    u_robot2 = u_combined_jerk[NU_PER_ROBOT+1 : 2*NU_PER_ROBOT]
    xdot2 = single_robot_dynamics_continuous(x_robot2, u_robot2, NJ)
    
    return vcat(xdot1, xdot2)
end

# Hermite-Simpson Collocation Constraint
function hermite_simpson_constraint(x_k, x_k_plus_1, u_k, dt_actual, p_dynamics)
    f_k = combined_robot_dynamics_continuous(x_k, u_k, p_dynamics)
    f_k_plus_1 = combined_robot_dynamics_continuous(x_k_plus_1, u_k, p_dynamics) # Control is constant over interval
    
    # All multiplications with dt_actual here are scalar * vector, which is fine.
    x_midpoint = 0.5 * (x_k + x_k_plus_1) + (dt_actual / 8.0) * (f_k - f_k_plus_1)
    f_midpoint = combined_robot_dynamics_continuous(x_midpoint, u_k, p_dynamics)
    
    return x_k + (dt_actual / 6.0) * (f_k + 4.0 * f_midpoint + f_k_plus_1) - x_k_plus_1
end

# --- Equality Constraints ---
function dircol_equality_constraints(Z::AbstractVector, p) # p is params_tuple
    indices = p.indices
    N_knot_points = indices.N_knot_points
    num_intervals = indices.num_intervals
    nx = indices.nx

    eq_constraints = Vector{eltype(Z)}()

    # 1. Initial state constraint: x_0 = x0_target
    x_0_vars = Z[indices.x[1]]
    append!(eq_constraints, x_0_vars - p.x0_target)

    # 2. Dynamics constraints (Hermite-Simpson)
    p_dynamics = (NJ = p.NJ_per_robot,) # Pass necessary params for dynamics
    for k = 1:num_intervals
        x_k = Z[indices.x[k]]
        x_k_plus_1 = Z[indices.x[k+1]]
        u_k = Z[indices.u[k]]
        h_k_scalar = indices.min_time ? Z[first(indices.h[k])] : 1.0 # Extract scalar
        dt_actual = h_k_scalar * p.nominal_dt                         # dt_actual is now scalar
        
        dyn_defect = hermite_simpson_constraint(x_k, x_k_plus_1, u_k, dt_actual, p_dynamics)
        append!(eq_constraints, dyn_defect)
    end

    # 3. Terminal state constraint: x_N = xg_target (can be partial, e.g., only position)
    x_N_vars = Z[indices.x[N_knot_points]]
    # Example: constrain full state. Modify if only q, or q and dq.
    append!(eq_constraints, x_N_vars - p.xg_target) 
                                                     
    return eq_constraints
end

# --- Inequality Constraints ---
function dircol_inequality_constraints(Z::AbstractVector, p) # p is params_tuple
    indices = p.indices
    N_knot_points = indices.N_knot_points
    num_intervals = indices.num_intervals
    NJ = p.NJ_per_robot
    NX_PER_ROBOT = 3 * NJ

    ineq_constraints = Vector{eltype(Z)}()
    
    # Collision Constraints (example structure)
    # This requires careful implementation using DifferentiableCollisions.jl
    # and the robot kinematics from robot_kinematics_utils.jl
    temp_P_links = deepcopy(p.P_links) # To avoid modifying the original params
    
    for k = 1:N_knot_points # Check collisions at each knot point
        x_k_state = Z[indices.x[k]]
        q1_k = x_k_state[1:NJ]
        q2_k = x_k_state[NX_PER_ROBOT .+ (1:NJ)]

        # Update collision primitive poses based on q1_k, q2_k using FK
        poses1_world_k, _, _, _ = forward_kinematics_poe(q1_k, p.robot_kin1)
        poses2_world_k, _, _, _ = forward_kinematics_poe(q2_k, p.robot_kin2)

        all_poses_world_k = [poses1_world_k, poses2_world_k]

        for r = 1:NUM_ROBOTS
            for l = 1:N_LINKS # N_LINKS should be NJ
                primitive_obj = temp_P_links[r][l]
                T_link_center = all_poses_world_k[r][l] 
                primitive_obj.r = T_link_center.translation
                primitive_obj.p = Rotations.params(Rotations.MRP(T_link_center.linear))
            end
        end
        
        # # Self-collisions for Robot 1
        # for (i, j) in p.self_collision_pairs
        #     prox_val, _ = dc.proximity(temp_P_links[1][i], temp_P_links[1][j])
        #     append!(ineq_constraints, p.collision_clearance - prox_val) # clearance - prox <= 0
        # end
        # # Self-collisions for Robot 2
        # for (i, j) in p.self_collision_pairs
        #     prox_val, _ = dc.proximity(temp_P_links[2][i], temp_P_links[2][j])
        #     append!(ineq_constraints, p.collision_clearance - prox_val)
        # end
        # Inter-robot collisions
        for i = 1:N_LINKS
            for j = 1:N_LINKS
                prox_val, _ = dc.proximity(temp_P_links[1][i], temp_P_links[2][j])
                append!(ineq_constraints, p.collision_clearance - prox_val)
            end
        end
    end
    return ineq_constraints
end

function analytical_combined_constraints_jacobian(params, Z::AbstractVector{T_Z}) where T_Z
    N = params.N
    idx = params.idx
    NX = params.idx.nx
    NU = params.idx.nu
    NJ = NUM_JOINTS_PER_ROBOT # From const in multi_arm_dircol.jl
    NX_PER_ROBOT_const = NX_PER_ROBOT # From const

    # --- Jacobian of Equality Constraints ---
    # For now, using ForwardDiff for the equality part.
    # This can be made fully analytical later if needed.
    J_eq = ForwardDiff.jacobian(z_ -> dircol_equality_constraints(z_, params), Z)

    # Recalculate n_ineq to be sure
    dummy_ineq_cons = dircol_inequality_constraints(Z, params) # Evaluate to get size
    n_ineq_actual = length(dummy_ineq_cons)

    # Initialize sparse Jacobian for inequalities
    # J_ineq needs to be built carefully, perhaps using sparse matrix constructors (I, J, V)
    # For simplicity here, we'll make it dense then convert, or fill a preallocated sparse matrix.
    # Let's assume it's dense for now and can be sparsified.
    J_ineq_rows = Int[]
    J_ineq_cols = Int[]
    J_ineq_vals = T_Z[]

    current_row_ineq = 0
 
    
    # 1. Collision Avoidance Constraints
    # inequality_constraint updates P_links poses internally. We must replicate that.
    P_links_current = deepcopy(params.P_links) # Make a mutable copy for FK updates

    for i_knot = 1:N
        x_i = Z[idx.x[i_knot]]
        q1 = x_i[1:NJ]
        q2 = x_i[NX_PER_ROBOT_const .+ (1:NJ)]

        poses1_world, _, _, _ = forward_kinematics_poe(q1, params.robot_kin1)
        poses2_world, _, _, _ = forward_kinematics_poe(q2, params.robot_kin2)
        
        all_poses_world_knot = [poses1_world, poses2_world]
        for r_update = 1:NUM_ROBOTS
            poses_r_knot = all_poses_world_knot[r_update]
            for j_link_update = 1:N_LINKS
                T_link_center_knot = poses_r_knot[j_link_update]
                P_links_current[r_update][j_link_update].r = T_link_center_knot.translation
                P_links_current[r_update][j_link_update].p = Rotations.params(Rotations.MRP(RotMatrix(T_link_center_knot.linear)))
            end
        end

        q1_indices_in_Z = idx.x[i_knot][1:NJ]
        q2_indices_in_Z = idx.x[i_knot][NX_PER_ROBOT_const .+ (1:NJ)]
        pose1_slice = 1:6
        pose2_slice = 7:12

        for link1_idx = 1:N_LINKS
            for link2_idx = 1:N_LINKS
                current_row_ineq += 1
                
                P1 = P_links_current[1][link1_idx]
                P2 = P_links_current[2][link2_idx]

                _prox_val, J_prox_combined = dc.proximity_gradient(P1, P2)
                J_prox_P1_pose = J_prox_combined[pose1_slice] 
                J_prox_P2_pose = J_prox_combined[pose2_slice]

                J_fk1_q1 = calculate_link_pose_jacobian_geom(q1, link1_idx, params.robot_kin1)
                J_fk2_q2 = calculate_link_pose_jacobian_geom(q2, link2_idx, params.robot_kin2)
                
                # Constraint is: params.collision_threshold - prox <= 0
                # Jacobian of constraint w.r.t. prox is -1.
                # Jacobian of prox w.r.t. pose1 is J_prox_P1_pose'
                # Jacobian of pose1 w.r.t. q1 is J_fk1_q1
                # So, d(constraint)/dq1 = -1 * J_prox_P1_pose' * J_fk1_q1
                
                # Contribution from q1
                jac_contrib_q1 = -J_prox_P1_pose' * J_fk1_q1
                for col_idx = 1:NJ
                    if abs(jac_contrib_q1[col_idx]) > 1e-9 # Add if non-zero
                        push!(J_ineq_rows, current_row_ineq)
                        push!(J_ineq_cols, q1_indices_in_Z[col_idx])
                        push!(J_ineq_vals, jac_contrib_q1[col_idx])
                    end
                end
                
                # Contribution from q2
                jac_contrib_q2 = -J_prox_P2_pose' * J_fk2_q2
                 for col_idx = 1:NJ
                    if abs(jac_contrib_q2[col_idx]) > 1e-9 # Add if non-zero
                        push!(J_ineq_rows, current_row_ineq)
                        push!(J_ineq_cols, q2_indices_in_Z[col_idx])
                        push!(J_ineq_vals, jac_contrib_q2[col_idx])
                    end
                end
            end
        end
    end
    
    if current_row_ineq != n_ineq_actual
        @warn "Row count mismatch in analytical_combined_constraints_jacobian for inequalities! Expected $n_ineq_actual, got $current_row_ineq"
    end

    J_ineq_sparse = sparse(J_ineq_rows, J_ineq_cols, J_ineq_vals, n_ineq_actual, idx.nz)

    # Combine Jacobians
    # J_eq is dense from ForwardDiff. J_ineq_sparse is sparse.
    # vcat should handle this and produce a sparse matrix if J_eq is converted to sparse first.
    J_combined = SparseArrays.vcat(SparseArrays.sparse(J_eq), J_ineq_sparse)
    
    return J_combined
end

function trajectory_length(params,X,U)
    N = params.N
    L = 0.0
    for k = 1:N-1
        L += norm(X[k+1][1:6] - X[k][1:6])
        L += norm(X[k+1][19:24] - X[k][19:24])
    end
    return L
end
function trajectory_jerk_avg(params,U)
    N = params.N
    J = 0.0
    for k = 1:N-1
        if params.min_time
            J += norm(U[k][2:end])
        else
            J += norm(U[k])
        end
    end
    J /= (N-1)
    return J
end
function trajectory_time(params,U)
    N = params.N
    T = 0.0
    for k = 1:N-1
        if params.min_time
            T += U[k][1]
        else
            T += params.dt
        end
    end
    return T
end

# Main DIRCOL optimization function
function optimize_trajectory_dircol(x0_target, xg_target, X_init_guess, U_init_guess, dircol_params, ipopt_settings)
    
    N_knot_points = dircol_params.N_knot_points
    nx = dircol_params.nx_total
    nu = dircol_params.nu_total # Total control dim (jerks for all robots)
    min_time = dircol_params.min_time

    indices = create_dircol_indices(nx, nu, N_knot_points, min_time)
    Z0 = build_initial_Z(X_init_guess, U_init_guess, indices, dircol_params.nominal_dt)

    # --- Define bounds for fmincon ---
    # Variable bounds (z_l, z_u)
    z_l = fill(-Inf, indices.nz)
    z_u = fill(Inf, indices.nz)

    # Example: if q_min/max are simple bounds on parts of x_k in Z
    for k_idx = 1:N_knot_points
        # x_k limits (q, dq, ddq for each robot)
        for r_idx = 0:NUM_ROBOTS-1
            q_indices_in_xk = (r_idx * dircol_params.NX_PER_ROBOT) .+ (1:dircol_params.NJ_per_robot)
            dq_indices_in_xk = (r_idx * dircol_params.NX_PER_ROBOT) .+ dircol_params.NJ_per_robot .+ (1:dircol_params.NJ_per_robot)
            ddq_indices_in_xk = (r_idx * dircol_params.NX_PER_ROBOT) .+ 2*dircol_params.NJ_per_robot .+ (1:dircol_params.NJ_per_robot)

            z_l[indices.x[k_idx][q_indices_in_xk]] .= dircol_params.q_min
            z_u[indices.x[k_idx][q_indices_in_xk]] .= dircol_params.q_max
            z_l[indices.x[k_idx][dq_indices_in_xk]] .= dircol_params.dq_min
            z_u[indices.x[k_idx][dq_indices_in_xk]] .= dircol_params.dq_max
            z_l[indices.x[k_idx][ddq_indices_in_xk]] .= dircol_params.ddq_min
            z_u[indices.x[k_idx][ddq_indices_in_xk]] .= dircol_params.ddq_max
        end
    end
    for k_idx = 1:indices.num_intervals
        # u_k (jerk) limits
        z_l[indices.u[k_idx]] = dircol_params.u_jerk_min
        z_u[indices.u[k_idx]] = dircol_params.u_jerk_max
        if min_time
            z_l[indices.h[k_idx]] .= dircol_params.h_min
            z_u[indices.h[k_idx]] .= dircol_params.h_max
        end
    end
     # Fix initial state directly in Z bounds (if not using equality constraint for it, but equality is cleaner)
    z_l[indices.x[1]] = x0_target
    z_u[indices.x[1]] = x0_target
    z_l[indices.x[indices.N_knot_points]] = xg_target
    z_u[indices.x[indices.N_knot_points]] = xg_target


    # The `params` argument to fmincon's cost/constraint functions will be `dircol_params_for_fmincon`.
    # It must also contain `indices` for cost/con to use.
    dircol_params_for_fmincon = merge(dircol_params, (indices=indices, idx=indices,))

    _cost_func(p_fm, Z) = dircol_cost_function(Z, p_fm)
    _equality_constraints_func(p_fm, Z) = dircol_equality_constraints(Z, p_fm)
    _inequality_constraints_func(p_fm, Z) = dircol_inequality_constraints(Z, p_fm)

    # Determine number of inequality constraints for their specific bounds
    num_ineq_constraints = length(_inequality_constraints_func(dircol_params_for_fmincon, Z0))

    # Bounds for inequality constraints: c_l <= inequality_constraint(Z) <= c_u
    # Assuming dircol_inequality_constraints are formulated as g(Z) <= 0
    cons_l_ineq = fill(-Inf, num_ineq_constraints)
    cons_u_ineq = zeros(num_ineq_constraints)

    # Prepare analytical Jacobian function if provided and diff_type is :analytical
    analytical_jac_func_to_pass = nothing
    if Symbol(ipopt_settings.diff_type) == :analytical
        if haskey(ipopt_settings, :analytical_constraint_jacobian_func) && 
           ipopt_settings.analytical_constraint_jacobian_func !== nothing
            analytical_jac_func_to_pass = ipopt_settings.analytical_constraint_jacobian_func
        else
            @warn "diff_type is :analytical but no analytical_constraint_jacobian_func provided in ipopt_settings. Ipopt may error or fmincon wrapper might try to use ForwardDiff/FiniteDiff if not strictly handled."
            # The fmincon.jl wrapper will error if analytical_constraint_jacobian_func is nothing and diff_type is :analytical.
        end
    end

    println("Starting DIRCOL optimization with Ipopt...")
    opt_time = @elapsed begin
        Z_sol = fmincon(
            _cost_func,
            _equality_constraints_func,
            _inequality_constraints_func,
            z_l, z_u,
            cons_l_ineq, cons_u_ineq, # Bounds for inequality constraints ONLY
            Z0,
            dircol_params_for_fmincon, # Params passed to cost and constraint funcs
            Symbol(ipopt_settings.diff_type);
            analytical_constraint_jacobian = analytical_jac_func_to_pass, # Must be Jacobian of COMBINED constraints
            tol = ipopt_settings.tol,
            c_tol = ipopt_settings.c_tol,
            max_iters = ipopt_settings.max_iters,
            verbose = ipopt_settings.verbose
        )
    end
    
    X_sol, U_sol, H_sol = extract_trajectory_from_Z(Z_sol, indices)
    
    total_time_optimized = 0.0
    if min_time
        for k=1:indices.num_intervals
            total_time_optimized += H_sol[k] * dircol_params.nominal_dt
        end
    else
        total_time_optimized = (N_knot_points - 1) * dircol_params.nominal_dt
    end

    # TODO: Check convergence status from Ipopt (fmincon should return it)
    # For now, assume success if it runs.
    success = true # Placeholder

    # Calculate final cost (optional, Ipopt provides it)
    # final_cost = dircol_cost_function(Z_sol, dircol_params_for_fmincon)

    # For DIRCOL, iterations are internal to Ipopt. We don't get a simple count like iLQR.
    # We can report Ipopt's iteration count if fmincon exposes it.
    iterations = -1 # Placeholder for Ipopt iterations

    # Trajectory length and avg jerk can be calculated similarly to iLQR if needed for benchmarks
    traj_length = trajectory_length(dircol_params, X_sol, U_sol) # Define this if needed
    avg_jerk = trajectory_jerk_avg(dircol_params, U_sol) # Define this if needed

    results = Dict(
        :success => success,
        :iterations => iterations, # Ipopt internal iterations
        :time_s => opt_time,
        # :final_cost => final_cost, # Cost from Ipopt objective
        :X_sol => X_sol,
        :U_sol => U_sol, # Jerk controls
        :H_sol => H_sol, # Time scaling factors
        :traj_time => total_time_optimized, # Actual trajectory duration
        :traj_length => traj_length, 
        :avg_jerk => avg_jerk  
    )
    return results
end
