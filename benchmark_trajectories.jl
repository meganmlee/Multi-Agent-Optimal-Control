# benchmark_trajectories.jl
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.add(["LinearAlgebra", "StaticArrays", "ForwardDiff", "FiniteDiff", "Printf", "SparseArrays", 
         "MeshCat", "Random", "Colors", "Rotations", "CoordinateTransformations", 
         "Ipopt", "MathOptInterface"])
using StaticArrays
using LinearAlgebra
using CSV
using DataFrames
using TickTock
using Printf

# Include necessary files (adjust paths as needed)
include("robot_kinematics_utils.jl") # Contains RobotKinematics, FK, Jacobians, etc.
include("trajectory_loader.jl")
include("run_optimization.jl") # Contains setup_ilqr_params, optimize_trajectory
include("dircol_optimization_utils.jl") # Include the new DIRCOL utilities

function run_benchmark(trajectory_dir::String, output_file::String; 
    run_min_time::Bool = true, solver::Symbol = :ilqr,
    dt_ilqr::Float64 = 0.1, dt_dircol::Float64 = 2.0)
    println("Starting Benchmark...")
    println("Trajectory Directory: ", trajectory_dir)
    println("Output File: ", output_file)
    println("Solver: ", solver)
    println("Run Min Time: ", run_min_time)
    if solver == :ilqr
        println("iLQR dt: ", dt_ilqr)
    elseif solver == :dircol
        println("DIRCOL dt: ", dt_dircol)
    end

    # --- Fixed Parameters for Benchmark ---
    T = Float64
    
    # Define Robot Kinematics 
    # Twists ξᵢ (Defined in base frame {0} at q=0)
    w1 = SA[T(0), T(0), T(1)]; p1 = SA[T(0), T(0), T(0)]
    w2 = SA[T(0), T(1), T(0)]; p2 = SA[T(0), T(0), T(0.33)]
    w3 = SA[T(0), T(-1), T(0)]; p3 = SA[T(0), T(0), T(0.33+0.26)]
    w4 = SA[T(-1), T(0), T(0)]; p4 = SA[T(0.1), T(0), T(0.33+0.26+0.015)]
    w5 = SA[T(0), T(-1), T(0)]; p5 = SA[T(0.1+0.19), T(0), T(0.33+0.26+0.015)]
    w6 = SA[T(-1), T(0), T(0)]; p6 = SA[T(0.1+0.19+0.072), T(0), T(0.33+0.26+0.015)]
    twists_base = [ vcat(-cross(w, p), w) for (w, p) in zip([w1,w2,w3,w4,w5,w6], [p1,p2,p3,p4,p5,p6]) ]
    
    # M_frames_zero: Pose of frame {i} relative to base {0} at q=0
    M_frames_zero = Vector{AffineMap{RotMatrix3{T}, SVector{3, T}}}(undef, NJ + 1)
    M_frames_zero[1] = AffineMap(RotMatrix{3,T}(I), SA[T(0),T(0),T(0)])
    points_p = [p1, p2, p3, p4, p5, p6]
    M_frames_zero[2] = AffineMap(RotMatrix{3,T}(I), points_p[1])
    M_frames_zero[3] = AffineMap(RotMatrix{3,T}(I), points_p[2])
    M_frames_zero[4] = AffineMap(RotMatrix(RotY(π/2)), points_p[3])
    M_frames_zero[5] = AffineMap(RotMatrix(RotY(π/2)), points_p[4])
    M_frames_zero[6] = AffineMap(RotMatrix(RotY(π/2)), points_p[5])
    M_frames_zero[7] = AffineMap(RotMatrix(RotY(π/2)), points_p[6])
     # T_link_centers: Pose of link {L_i} center relative to frame {i}
    link_lengths_geom = [0.33, 0.26, 0.1, 0.19, 0.072, 0.11]
    T_link_centers = Vector{AffineMap{RotMatrix3{T}, SVector{3, T}}}(undef, NJ)
    for i = 1:NJ
        L_i = link_lengths_geom[i]
        T_link_centers[i] = AffineMap(RotMatrix(RotY(π/2)), SA[T(0.0), T(0.0), T(L_i/2.0)])
    end
    vis_link_radii = [0.095, 0.095, 0.06, 0.06, 0.05, 0.015]
    # Base transforms
    T_base1 = AffineMap(RotMatrix(RotZ(0.0)), SA[T(0.0), T(0.0), T(0)])
    T_base2 = AffineMap(RotMatrix(RotZ(3.13585279)), SA[T(0.88101), T(-0.01304), T(0)])
    robot_kin1 = RobotKinematics(twists_base, T_base1, M_frames_zero, T_link_centers, vis_link_radii)
    robot_kin2 = RobotKinematics(twists_base, T_base2, M_frames_zero, T_link_centers, vis_link_radii)
 
    # define self-collision pairs to check
    self_collision_pairs_to_check = [
        (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 5), (2, 6),
        (3, 5), (3, 6),
        (4, 6)
    ]
    # Filter pairs based on actual NJ
    self_collision_pairs = filter(p -> p[1] <= NJ && p[2] <= NJ, self_collision_pairs_to_check)
    println("Checking self-collision pairs: ", self_collision_pairs)

    # --- Create Collision Primitives ---
    P_links = [[], []]
    primitive_shape = ["capsule", "capsule", "capsule", "cylinder", "capsule", "cylinder"]
    for r = 1:NUM_ROBOTS
        current_robot_kin = (r == 1) ? robot_kin1 : robot_kin2
        for i = 1:N_LINKS # N_LINKS = NJ
            len = link_lengths_geom[i] # Use calculated geometric length
            rad = T(current_robot_kin.link_radius[i])
            if primitive_shape[i] == "capsule"
                # Capsule: radius and length
                link_prim = dc.CapsuleMRP(rad, len)
            elseif primitive_shape[i] == "cylinder"
                # Cylinder: radius and length
                link_prim = dc.CylinderMRP(rad, len)
            else
                error("Unknown primitive shape: $primitive_shape[i]")
            end
            link_prim.r_offset = SA[T(0), T(0), T(0)] # Offsets handled by T_link_centers now
            link_prim.Q_offset = SMatrix{3,3,T,9}(I)
            push!(P_links[r], link_prim)
        end
    end

    # Fixed Cost, Limit, Collision Parameters
    cost_params_ilqr = (Qf_q=100.0, Qf_dq=10.0, Qf_ddq=1.0, R_jerk=0.01, min_time=run_min_time)
    q_lim_single = (fill(T(Q_LIMITS[1]), NJ), fill(T(Q_LIMITS[2]), NJ))
    dq_lim_single = (fill(T(DQ_LIMITS[1]), NJ), fill(T(DQ_LIMITS[2]), NJ))
    ddq_lim_single = (fill(T(DDQ_LIMITS[1]), NJ), fill(T(DDQ_LIMITS[2]), NJ))
    
    # Jerk limits for NU (combined jerks for both robots)
    u_jerk_min_vector = fill(T(JERK_LIMITS[1]), NU) 
    u_jerk_max_vector = fill(T(JERK_LIMITS[2]), NU)

    limit_params_ilqr = (; q_lim_single, dq_lim_single, ddq_lim_single, u_min=u_jerk_min_vector, u_max=u_jerk_max_vector)
    coll_params_common = (threshold=T(COLLISION_THRESHOLD), self_collision_pairs=self_collision_pairs, P_links=P_links)


    # iLQR Settings
    ilqr_settings = (atol=1e-1, max_iters=3000, verbose=false, rho=1.0, phi=10.0)

    ipopt_settings_dircol = (
        tol = 1e-3, 
        c_tol = 1e-3, 
        max_iters = 1000, 
        verbose = false, 
        diff_type = "analytical",
        analytical_constraint_jacobian_func = analytical_combined_constraints_jacobian
    )

    # --- Benchmarking Loop ---
    all_results = []
    trajectory_files = filter(f -> endswith(f, ".csv"), readdir(trajectory_dir, join=true))
    println("Found $(length(trajectory_files)) trajectory files.")

    for (case_idx, filepath) in enumerate(trajectory_files)
        println("\nProcessing Case $(case_idx)/$(length(trajectory_files)): $(basename(filepath))")
        tick() # Start timer for this case

        # 1. Load Trajectory
        q_sequence_raw, q_start_loaded, q_goal_loaded, timestamps_loaded = load_trajectory_from_csv(filepath, NJ)
        if q_sequence_raw === nothing
            @warn "Skipping file due to loading error: $filepath"
            tock()
            continue
        end

        # Construct full start/goal states (x = [q; dq; ddq])
        x0 = vcat(q_start_loaded[1:NJ], zeros(NJ*2), q_start_loaded[NJ .+ (1:NJ)], zeros(NJ*2))
        xg = vcat(q_goal_loaded[1:NJ], zeros(NJ*2), q_goal_loaded[NJ .+ (1:NJ)], zeros(NJ*2))
    
        # --- Setup Params ---
        current_dt = (solver == :ilqr) ? dt_ilqr : dt_dircol
        # N_knot_points is the number of states in the trajectory X
        # num_intervals = N_knot_points - 1 (number of controls U and time steps h)
        current_N_knot_points = Int(floor(timestamps_loaded[end] / current_dt)) + 1
        if current_N_knot_points < 2
            @warn "Calculated N_knot_points is less than 2 ($current_N_knot_points) for $filepath with dt=$current_dt. Setting to 2."
            current_N_knot_points = 2
        end
        num_intervals = current_N_knot_points - 1

    
        local results_warm, results_cold # Ensure they are in scope for storing

        # --- Solver Specific Path ---
        if solver == :ilqr
            # --- iLQR Path ---
            params_ilqr = setup_ilqr_params(x0, xg, current_N_knot_points, current_dt, # current_dt is dt_ilqr
                                            robot_kin1, robot_kin2, 
                                            cost_params_ilqr, limit_params_ilqr, coll_params_common)
            if params_ilqr.ncx == 0 && solver == :ilqr; @warn("iLQR: ncx is zero for $filepath"); end
            
            # NU_ilqr depends on whether min_time is active for iLQR (adds one dim for dt)
            actual_NU_ilqr = params_ilqr.min_time ? NU + 1 : NU

            # --- Warm Start (iLQR) ---
            println(" Running Warm Start with iLQR...")
            q_sequence_resampled = resample_q_sequence(q_sequence_raw, timestamps_loaded, current_N_knot_points)
            X_warm_init_ilqr = create_state_trajectory(q_sequence_resampled, current_N_knot_points, current_dt, NJ)
            if X_warm_init_ilqr === nothing # Fallback
                 X_warm_init_ilqr = [deepcopy(x0) + ((i-1)/num_intervals)*(xg - x0) for i = 1:current_N_knot_points]
            end
            U_warm_init_ilqr_jerks = [zeros(T, NU) for _ = 1:num_intervals] # Jerks part
            U_warm_init_ilqr = if params_ilqr.min_time
                [vcat(current_dt, u_jerk) for u_jerk in U_warm_init_ilqr_jerks] # Prepend nominal dt
            else
                U_warm_init_ilqr_jerks
            end
            results_warm = optimize_trajectory(x0, xg, X_warm_init_ilqr, U_warm_init_ilqr, params_ilqr, ilqr_settings)

            # --- Cold Start (iLQR) ---
            println(" Running Cold Start with iLQR...")
            X_cold_init_ilqr = [deepcopy(x0) + ((i-1)/num_intervals)*(xg - x0) for i = 1:current_N_knot_points]
            U_cold_init_ilqr_jerks = [zeros(T, NU) for _ = 1:num_intervals]
            U_cold_init_ilqr = if params_ilqr.min_time
                [vcat(current_dt, u_jerk) for u_jerk in U_cold_init_ilqr_jerks]
            else
                U_cold_init_ilqr_jerks
            end
            results_cold = optimize_trajectory(x0, xg, X_cold_init_ilqr, U_cold_init_ilqr, params_ilqr, ilqr_settings)

        elseif solver == :dircol
            # --- DIRCOL Path ---
            dircol_params_setup = (
                N = current_N_knot_points,
                N_knot_points = current_N_knot_points,
                nominal_dt = current_dt, # This is dt_dircol
                min_time = run_min_time, 
                nx_total = NX, 
                nu_total = NU, # DIRCOL's U is just jerks, time is separate in Z
                NJ_per_robot = NJ,
                NX_PER_ROBOT = NX_PER_ROBOT,
                robot_kin1 = robot_kin1, robot_kin2 = robot_kin2,
                P_links = P_links, 
                self_collision_pairs = self_collision_pairs,
                collision_clearance = coll_params_common.threshold,
                R_cost_matrix = Diagonal(fill(T(cost_params_ilqr.R_jerk), NU)), # Reuse R_jerk weight
                time_penalty_weight = run_min_time ? T(1.0) : T(0.0), # Example weight for sum(h_k*dt)
                q_min = q_lim_single[1][1], q_max = q_lim_single[2][1], 
                dq_min = dq_lim_single[1][1], dq_max = dq_lim_single[2][1],
                ddq_min = ddq_lim_single[1][1], ddq_max = ddq_lim_single[2][1],
                u_jerk_min = u_jerk_min_vector, u_jerk_max = u_jerk_max_vector, # Pass vectors
                h_min = T(0.1), h_max = T(5), # Bounds for time scaling factor h_k (relative to nominal_dt)
                x0_target = x0,
                xg_target = xg,
            )

            # --- Warm Start (DIRCOL) ---
            println(" Running Warm Start with DIRCOL...")
            q_sequence_resampled = resample_q_sequence(q_sequence_raw, timestamps_loaded, current_N_knot_points)
            X_warm_init_dircol = create_state_trajectory(q_sequence_resampled, current_N_knot_points, current_dt, NJ)
            if X_warm_init_dircol === nothing # Fallback
                 X_warm_init_dircol = [deepcopy(x0) + ((i-1)/num_intervals)*(xg - x0) for i = 1:current_N_knot_points]
            end
            U_warm_init_dircol = [zeros(T, NU) for _ = 1:num_intervals] # Jerks only for DIRCOL U_init
            results_warm = optimize_trajectory_dircol(x0, xg, X_warm_init_dircol, U_warm_init_dircol, dircol_params_setup, ipopt_settings_dircol)

            # --- Cold Start (DIRCOL) ---
            println(" Running Cold Start with DIRCOL...")
            X_cold_init_dircol = [deepcopy(x0) + ((i-1)/num_intervals)*(xg - x0) for i = 1:current_N_knot_points]
            U_cold_init_dircol = [zeros(T, NU) for _ = 1:num_intervals] # Jerks only
            results_cold = optimize_trajectory_dircol(x0, xg, X_cold_init_dircol, U_cold_init_dircol, dircol_params_setup, ipopt_settings_dircol)
        else
            error("Unknown solver: $solver")
        end

        # --- 4. Store Results ---
        warm_traj_length = haskey(results_warm, :traj_length) ? results_warm[:traj_length] : -1.0
        warm_avg_jerk = haskey(results_warm, :avg_jerk) ? results_warm[:avg_jerk] : -1.0
        cold_traj_length = haskey(results_cold, :traj_length) ? results_cold[:traj_length] : -1.0
        cold_avg_jerk = haskey(results_cold, :avg_jerk) ? results_cold[:avg_jerk] : -1.0

        push!(all_results, Dict(
            :filename => basename(filepath),
            :solver => solver,
            :run_min_time => run_min_time,
            :dt_used => current_dt,
            :N_knot_points => current_N_knot_points,
            :warm_success => results_warm[:success],
            :warm_iterations => results_warm[:iterations],
            :warm_time_s => results_warm[:time_s],
            :warm_traj_actual_duration => results_warm[:traj_time], # Actual duration from solver
            :warm_traj_length_metric => warm_traj_length, 
            :warm_avg_jerk_metric => warm_avg_jerk,    
            :cold_success => results_cold[:success],
            :cold_iterations => results_cold[:iterations],
            :cold_time_s => results_cold[:time_s],
            :cold_traj_actual_duration => results_cold[:traj_time],
            :cold_traj_length_metric => cold_traj_length,
            :cold_avg_jerk_metric => cold_avg_jerk,
        ))
        tock() # Stop timer for this case

        # Optional: Save intermediate results periodically
        if case_idx % 10 == 0
             println("Saving intermediate results...")
             CSV.write(output_file * "_intermediate.csv", DataFrame(all_results))
        end

    end # End loop over files

    # --- 5. Save Final Results ---
    println("\nBenchmark Complete. Saving final results to $output_file")
    if !isempty(all_results)
        results_df = DataFrame(all_results)
        CSV.write(output_file, results_df)
        println(results_df) # Print summary
    else
        println("No results generated.")
    end
end

# --- Execution ---
const TRAJECTORY_DIR = "solutions" # <--- SET THIS PATH
const OUTPUT_CSV = "benchmark_results.csv"

# Make sure all required functions (FK, Jacobians, iLQR, etc.) are loaded
# from the included files before calling run_benchmark.
# Example call:
run_benchmark(TRAJECTORY_DIR, OUTPUT_CSV, run_min_time=false, solver=:ilqr)