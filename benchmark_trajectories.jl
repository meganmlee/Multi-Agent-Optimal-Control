# benchmark_trajectories.jl
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

function run_benchmark(trajectory_dir::String, output_file::String; run_min_time::Bool = true)
    println("Starting Benchmark...")
    println("Trajectory Directory: ", trajectory_dir)
    println("Output File: ", output_file)

    # --- Fixed Parameters for Benchmark ---
    T = Float64
    N = 51    # Number of time steps for optimization
    dt = 0.1   # Time step duration
    
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
    cost_params = (Qf_q=100.0, Qf_dq=10.0, Qf_ddq=1.0, R_jerk=0.01, min_time=run_min_time)
    q_lim_single = (fill(T(Q_LIMITS[1]), NJ), fill(T(Q_LIMITS[2]), NJ))
    dq_lim_single = (fill(T(DQ_LIMITS[1]), NJ), fill(T(DQ_LIMITS[2]), NJ))
    ddq_lim_single = (fill(T(DDQ_LIMITS[1]), NJ), fill(T(DDQ_LIMITS[2]), NJ))
    u_min = fill(T(JERK_LIMITS[1]), NU); u_max = fill(T(JERK_LIMITS[2]), NU)
    limit_params = (; q_lim_single, dq_lim_single, ddq_lim_single, u_min, u_max)
    coll_params = (threshold=T(COLLISION_THRESHOLD), self_collision_pairs=self_collision_pairs, P_links=P_links)

    # iLQR Settings
    ilqr_settings = (atol=1e-1, max_iters=3000, verbose=false, rho=1.0, phi=10.0) # Turn verbose off for benchmark

    # --- Benchmarking Loop ---
    all_results = []
    trajectory_files = filter(f -> endswith(f, ".csv"), readdir(trajectory_dir, join=true))
    println("Found $(length(trajectory_files)) trajectory files.")

    for (idx, filepath) in enumerate(trajectory_files)
        println("\nProcessing Case $(idx)/$(length(trajectory_files)): $(basename(filepath))")
        tick() # Start timer for this case

        # 1. Load Trajectory
        q_sequence_raw, q_start, q_goal, timestamps = load_trajectory_from_csv(filepath, NJ)
        if q_sequence_raw === nothing
            @warn "Skipping file due to loading error: $filepath"
            tock()
            continue
        end

        # Construct full start/goal states (x = [q; dq; ddq])
        x0 = vcat(q_start[1:NJ], zeros(NJ*2), q_start[NJ .+ (1:NJ)], zeros(NJ*2))
        xg = vcat(q_goal[1:NJ], zeros(NJ*2), q_goal[NJ .+ (1:NJ)], zeros(NJ*2))
    
        # --- Setup Params ---
        N = Int(floor(timestamps[end] / 0.1)) 
        dt = 0.1
    
        # Setup common iLQR params for this start/goal
        # Need to run setup *once* per start/goal to get ncx etc.
        # Use x0 for evaluation as dq/ddq are zero anyway
        params = setup_ilqr_params(x0, xg, N, dt, robot_kin1, robot_kin2, cost_params, limit_params, coll_params)
        if params.ncx == 0; @warn("ncx is zero for $filepath, check setup or constraints!"); end

        # --- 2. Warm Start ---
        println(" Running Warm Start...")
        q_sequence_resampled = resample_q_sequence(q_sequence_raw, timestamps, N)
        if q_sequence_resampled === nothing
             @warn "Resampling failed for $filepath, falling back to interpolation for warm start."
             X_warm_init = [deepcopy(x0) + ((i-1)/(N-1))*(xg - x0) for i = 1:N]
        else
             X_warm_init = create_state_trajectory(q_sequence_resampled, N, dt, NJ)
             if X_warm_init === nothing
                 @warn "State trajectory creation failed for $filepath, falling back to interpolation."
                 X_warm_init = [deepcopy(x0) + ((i-1)/(N-1))*(xg - x0) for i = 1:N]
             end
        end
        U_warm_init = [zeros(T, NU) for i = 1:N-1] # Start with zero control always
        if params.min_time
            # insert timestamps as the first column of U_warm_init
            U_warm_init = [vcat(dt, U_warm_init[i]) for i = 1:N-1]
        end
        results_warm = optimize_trajectory(x0, xg, X_warm_init, U_warm_init, params, ilqr_settings)

        # --- 3. Cold Start ---
        println(" Running Cold Start...")
        X_cold_init = [deepcopy(x0) + ((i-1)/(N-1))*(xg - x0) for i = 1:N]
        U_cold_init = [zeros(T, NU) for i = 1:N-1]
        if params.min_time
            # insert timestamps as the first column of U_cold_init
            U_cold_init = [vcat(dt, U_cold_init[i]) for i = 1:N-1]
        end
        results_cold = optimize_trajectory(x0, xg, X_cold_init, U_cold_init, params, ilqr_settings)

        # --- 4. Store Results ---
        push!(all_results, Dict(
            :filename => basename(filepath),
            :warm_success => results_warm[:success],
            :warm_iterations => results_warm[:iterations],
            :warm_time_s => results_warm[:time_s],
            :warm_traj_length => results_warm[:traj_length],
            :warm_avg_jerk => results_warm[:avg_jerk],
            :warm_traj_time => results_warm[:traj_time],
            :cold_success => results_cold[:success],
            :cold_iterations => results_cold[:iterations],
            :cold_time_s => results_cold[:time_s],
            :cold_traj_length => results_cold[:traj_length],
            :cold_avg_jerk => results_cold[:avg_jerk],
            :cold_traj_time => results_cold[:traj_time],
        ))
        tock() # Stop timer for this case

        # Optional: Save intermediate results periodically
        if idx % 10 == 0
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
run_benchmark(TRAJECTORY_DIR, OUTPUT_CSV, run_min_time=true)