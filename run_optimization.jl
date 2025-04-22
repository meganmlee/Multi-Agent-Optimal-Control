# run_optimization.jl
using StaticArrays
using LinearAlgebra
using Printf
using TickTock # For timing

include("simple_altro.jl")
include("min_time_altro.jl")

# Make sure RobotKinematics and NJ/NX/NU etc are defined or imported
# Assuming the main script defines these and include("simple_altro.jl")

"""
    setup_ilqr_params(x0, xg, N, dt, robot_kin1, robot_kin2, cost_params, limit_params, coll_params)

Helper function to create the 'params' tuple needed by iLQR.
"""
function setup_ilqr_params(x0, xg, N, dt, robot_kin1, robot_kin2, cost_params, limit_params, coll_params)
    T = eltype(x0) # Infer type
    NJ = Int(length(x0) / 6) # Infer NJ assuming x0 has size 2 * 3 * NJ

    # Unpack parameters
    Qf_q_weight = cost_params.Qf_q
    Qf_dq_weight = cost_params.Qf_dq
    Qf_ddq_weight = cost_params.Qf_ddq
    R_jerk_weight = cost_params.R_jerk

    q_lim_single = limit_params.q_lim_single
    dq_lim_single = limit_params.dq_lim_single
    ddq_lim_single = limit_params.ddq_lim_single
    u_min = limit_params.u_min
    u_max = limit_params.u_max

    collision_threshold = coll_params.threshold
    self_collision_pairs = coll_params.self_collision_pairs
    P_links = coll_params.P_links # Needs to be pre-created

    # Rebuild cost matrices etc. based on NJ
    NX = 2 * 3 * NJ
    NU = 2 * NJ
    if cost_params.min_time
        NU = NU+1 # For min_time, need additional time dimension as control input
    end
    Q = Diagonal(zeros(T, NX))
    Qf_diag_single = vcat(fill(T(Qf_q_weight), NJ), fill(T(Qf_dq_weight), NJ), fill(T(Qf_ddq_weight), NJ))
    Qf = Diagonal(vcat(Qf_diag_single, Qf_diag_single))
    
    R = T(R_jerk_weight) * Diagonal(ones(T, NU))
    

    q_lim_all = [q_lim_single, q_lim_single]
    dq_lim_all = [dq_lim_single, dq_lim_single]
    ddq_lim_all = [ddq_lim_single, ddq_lim_single]

    Xref = [deepcopy(xg) for i = 1:N]
    

    # Calculate ncx, ncu dynamically (important!)
    if cost_params.min_time
        Uref = [[1.0 ; zeros(T, NU-1)] for i = 1:N-1]
        h_min = 0.3; h_max = 5.0
        temp_params_for_eval = (;
            min_time = cost_params.min_time,
            robot_kin=[robot_kin1, robot_kin2], q_lim=q_lim_all, dq_lim=dq_lim_all, ddq_lim=ddq_lim_all,
            P_links, collision_threshold = T(collision_threshold),
            nu=NU-1, u_min_jerk = u_min, u_max_jerk=u_max, h_min, h_max, self_collision_pairs
        )
    else
        Uref = [zeros(T, NU) for i = 1:N-1]
        temp_params_for_eval = (;
            min_time = cost_params.min_time,
            robot_kin=[robot_kin1, robot_kin2], q_lim=q_lim_all, dq_lim=dq_lim_all, ddq_lim=ddq_lim_all,
            P_links, collision_threshold = T(collision_threshold),
            nu=NU, u_min, u_max, self_collision_pairs
        )
    end
    # Need access to ineq_con_x, ineq_con_u definitions
    # Assuming they are globally available or passed in
    ncx = length(ineq_con_x(temp_params_for_eval, x0))
    ncu = length(ineq_con_u(temp_params_for_eval, Uref[1]))
    # println("Dynamic ncx = $ncx, ncu = $ncu") # Debug

    if cost_params.min_time
        params = (
            min_time = true,
            nx = NX, nu = NU, ncx = ncx, ncu = ncu, N = N,
            Q = Q, R = R, Qf = Qf,
            u_min_jerk = u_min, u_max_jerk = u_max,h_min = h_min, h_max = h_max,
            q_lim = q_lim_all, dq_lim = dq_lim_all, ddq_lim = ddq_lim_all,
            Xref = Xref, Uref = Uref, dt = dt,
            P_links = P_links,
            robot_kin = [robot_kin1, robot_kin2],
            collision_threshold = T(collision_threshold),
            self_collision_pairs = self_collision_pairs
        )
    else
        params = (
            min_time = false,
            nx = NX, nu = NU, ncx = ncx, ncu = ncu, N = N,
            Q = Q, R = R, Qf = Qf,
            u_min = u_min, u_max = u_max,
            q_lim = q_lim_all, dq_lim = dq_lim_all, ddq_lim = ddq_lim_all,
            Xref = Xref, Uref = Uref, dt = dt,
            P_links = P_links,
            robot_kin = [robot_kin1, robot_kin2],
            collision_threshold = T(collision_threshold),
            self_collision_pairs = self_collision_pairs
        )
    end
    return params
end


"""
    optimize_trajectory(x0, xg, X_init, U_init, params, ilqr_settings)

Runs the iLQR optimization for a single case.

Args:
    x0 (Vector): Start state.
    xg (Vector): Goal state.
    X_init (Vector{Vector}): Initial state trajectory guess.
    U_init (Vector{Vector}): Initial control trajectory guess.
    params (NamedTuple): Parameters tuple for iLQR (from setup_ilqr_params).
    ilqr_settings (NamedTuple): Settings like atol, max_iters, ρ, ϕ, verbose.

Returns:
    Dict: Results including :success, :iterations, :time_s, :final_cost, :X_sol, :U_sol
"""
function optimize_trajectory(x0, xg, X_init, U_init, params, ilqr_settings)
    T = eltype(x0)
    NX = params.nx
    NU = params.nu
    N = params.N

    # Ensure initial guess has correct length
    if length(X_init) != N || length(U_init) != N - 1
         @error "Initial guess trajectory length mismatch! Expected X: $N, U: $(N-1). Got X: $(length(X_init)), U: $(length(U_init))"
         # Attempt recovery or return error
         # Simple recovery: Use interpolation if lengths mismatch significantly
         if abs(length(X_init) - N) > 1 || abs(length(U_init) - (N-1)) > 1
             println("Lengths mismatch, falling back to interpolation for initial guess.")
             X_init = [deepcopy(x0) + ((i-1)/(N-1))*(xg - x0) for i = 1:N]
             U_init = [zeros(T, NU) for i = 1:N-1]
         else # If only off by one endpoint perhaps, try truncating/padding
             if length(X_init) > N; X_init = X_init[1:N]; end
             if length(U_init) > N-1; U_init = U_init[1:N-1]; end
             # Padding is harder, fallback safer
             @warn "Slight length mismatch, using truncated guess or fallback."
             X_init = [deepcopy(x0) + ((i-1)/(N-1))*(xg - x0) for i = 1:N]
             U_init = [zeros(T, NU) for i = 1:N-1]
         end
    end

    # Deepcopy initial guesses to avoid modifying originals
    X = deepcopy(X_init)
    U = deepcopy(U_init)

    # iLQR Variables
    Xn = deepcopy(X); Un = deepcopy(U)
    P = [zeros(T, NX, NX) for i = 1:N]; p = [zeros(T, NX) for i = 1:N]
    if params.min_time
        d = [zeros(T, NU+1) for i = 1:N-1]; K = [zeros(T, NU+1, NX) for i = 1:N-1]
    else
        d = [zeros(T, NU) for i = 1:N-1]; K = [zeros(T, NU, NX) for i = 1:N-1]
    end

    success = false
    traj_length = Inf
    avg_jerk = Inf
    traj_time = Inf
    iterations = 0
    X_sol = nothing
    U_sol = nothing # Need to get U from iLQR result if possible, else it's mutated U

    println("Running iLQR...")
    time_s = @elapsed begin
        try
            # Assuming iLQR function returns history and maybe success/iterations info
            # Modify simple_altro.jl's iLQR to return more info if needed
            # For now, assume it mutates X, U and returns Xhist
            Xhist = iLQR(params, X, U, P, p, K, d, Xn, Un;
                         atol=ilqr_settings.atol, max_iters=ilqr_settings.max_iters,
                         verbose=ilqr_settings.verbose, ρ=ilqr_settings.rho, ϕ=ilqr_settings.phi)

            # How to determine success and iterations?
            # Option 1: Modify iLQR to return status tuple (success, iter_count, final_J)
            # Option 2: Infer from Xhist length vs max_iters
            iterations = length(Xhist) - 1 # Approx iterations
            if iterations < ilqr_settings.max_iters
                success = true # Converged before hitting max iterations
                # Recalculate final cost (J) based on final X, U
                # Need trajectory_cost function (similar to one inside iLQR)
                # For now, just indicate success based on iterations
            else
                success = false # Hit max iterations
            end
            # For simplicity, store the mutated X, U as the solution
            X_sol = Xhist[end]
            U_sol = deepcopy(U) # Assuming U is mutated in iLQR
            # Ideally get final cost from iLQR or recompute
            traj_length = trajectory_length(params, X_sol, U_sol)
            avg_jerk = trajectory_jerk_avg(params, U_sol)
            traj_time = trajectory_time(params, U_sol)

        catch e
            @error "iLQR failed with error: $e"
            # Optionally: Rethrow, or capture stacktrace
            # For benchmarking, just record failure
            success = false
            iterations = ilqr_settings.max_iters # Assume it ran until error/max
        end
    end
    println("iLQR finished. Success: $success, Iterations: $iterations, Time: $(round(time_s, digits=3))s")

    results = Dict(
        :success => success,
        :iterations => iterations,
        :time_s => time_s,
        :traj_length => traj_length, # Assuming this is the cost metric
        :avg_jerk => avg_jerk,
        :traj_time => traj_time,
        :X_sol => X_sol
    )
    return results
end