# trajectory_loader.jl

using CSV
using DataFrames
using StaticArrays
using LinearAlgebra # For interpolation

# Define constants based on the main script or pass them if needed
# NJ = 6 # Number of joints per robot (Assuming 6 based on CSV)
# NX_PER_ROBOT = 3 * NJ
# NX = 2 * NX_PER_ROBOT

"""
    load_trajectory_from_csv(filepath::String, num_joints_per_robot::Int)

Loads joint angle data from a CSV file.

Args:
    filepath (String): Path to the CSV file.
    num_joints_per_robot (Int): Number of joints for each robot (e.g., 6).

Returns:
    Tuple:
        - q_sequence (Vector{Vector{Float64}}): Sequence of combined joint angles [q1; q2].
        - q_start (Vector{Float64}): Initial combined joint configuration.
        - q_goal (Vector{Float64}): Final combined joint configuration.
        - timestamps (Vector{Float64}): Timestamps from the file.
    Returns (nothing, nothing, nothing, nothing) if loading fails.
"""
function load_trajectory_from_csv(filepath::String, num_joints_per_robot::Int)
    println("Loading trajectory from: ", filepath)
    local df
    try
        df = CSV.read(filepath, DataFrame)
        println("Finished reading CSV file: ", filepath)
    catch e
        @warn "Failed to read CSV file: $filepath. Error: $e"
        return nothing, nothing, nothing, nothing
    end

    NJ = num_joints_per_robot
    num_robots = 2 # Assuming 2 robots based on columns

    # Dynamically generate column names
    joint_cols = Symbol[]
    for r = 1:num_robots
        for j = 1:NJ
            push!(joint_cols, Symbol("gpfour$(r-1)_arm_joint_$(j)"))
        end
    end

    # Check if all required columns exist
    missing_cols = setdiff(joint_cols, Symbol.(names(df)))
    if !isempty(missing_cols)
        @warn "Missing columns in $filepath: $missing_cols"
        # Attempt to proceed if only some are missing? Risky. Better to return nothing.
         println("Available columns: ", names(df))
        return nothing, nothing, nothing, nothing
    end

    try
        q_sequence = Vector{Vector{Float64}}()
        for row in eachrow(df)
            q_combined = Vector{Float64}(undef, num_robots * NJ)
            idx = 1
            for r = 0:num_robots-1
                for j = 1:NJ
                    col_name = Symbol("gpfour$(r)_arm_joint_$(j)")
                    q_combined[idx] = row[col_name]
                    idx += 1
                end
            end
            push!(q_sequence, q_combined)
        end

        if isempty(q_sequence)
            @warn "No data rows found in $filepath"
            return nothing, nothing, nothing, nothing
        end

        q_start = q_sequence[1]
        q_goal = q_sequence[end]
        timestamps = df[!, 1] # Assuming first column is time

        # print q_start and q_goal for debugging
        println("q_start: ", q_start)
        println("q_goal: ", q_goal)
        return q_sequence, q_start, q_goal, timestamps
    catch e
        @warn "Error processing data in $filepath. Error: $e"
        return nothing, nothing, nothing, nothing
    end
end

"""
    resample_q_sequence(q_sequence, timestamps, N_target::Int)

Resamples the joint angle sequence to have N_target points using linear interpolation.
"""
function resample_q_sequence(q_sequence::Vector{Vector{Float64}}, timestamps::Vector{Float64}, N_target::Int)
    if isempty(q_sequence) || length(q_sequence) != length(timestamps)
        @warn "Invalid input for resampling."
        return nothing
    end
    if length(q_sequence) == N_target
        return q_sequence # No resampling needed
    end

    dim_q = length(q_sequence[1])
    q_resampled = Vector{Vector{Float64}}(undef, N_target)

    t_start = timestamps[1]
    t_end = timestamps[end]
    target_times = range(t_start, stop=t_end, length=N_target)

    current_idx = 1
    for i = 1:N_target
        t = target_times[i]

        # Find interval [t_k, t_{k+1}] containing t
        while current_idx < length(timestamps) && timestamps[current_idx+1] < t
            current_idx += 1
        end
        # Ensure we don't go out of bounds if t is exactly t_end
        idx1 = min(current_idx, length(timestamps))
        idx2 = min(current_idx + 1, length(timestamps))

        t1 = timestamps[idx1]
        t2 = timestamps[idx2]
        q1 = q_sequence[idx1]
        q2 = q_sequence[idx2]

        # Linear interpolation factor
        alpha = (t2 ≈ t1) ? 0.0 : (t - t1) / (t2 - t1)
        alpha = clamp(alpha, 0.0, 1.0) # Clamp to handle potential float issues

        q_interpolated = q1 .+ alpha .* (q2 .- q1)
        q_resampled[i] = q_interpolated
    end

    return q_resampled
end


"""
    create_state_trajectory(q_sequence_resampled::Vector{Vector{Float64}}, N::Int, dt::Float64, nj::Int)

Creates the full state trajectory X = [q; dq; ddq] from a joint angle sequence.
Sets dq and ddq to zero for simplicity in warm-starting.

Args:
    q_sequence_resampled (Vector{Vector{Float64}}): Sequence of combined joint angles [q1; q2] of length N.
    N (Int): Target number of steps.
    dt (Float64): Time step duration.
    nj (Int): Number of joints per robot.

Returns:
    Vector{Vector{Float64}}: State trajectory X. Returns nothing on error.
"""
function create_state_trajectory(q_sequence_resampled::Vector{Vector{Float64}}, N::Int, dt::Float64, nj::Int)
    if length(q_sequence_resampled) != N
        @warn "Input q_sequence length ($(length(q_sequence_resampled))) does not match N ($N)."
        return nothing
    end

    num_robots = 2
    NX = num_robots * 3 * nj # q, dq, ddq
    NX_PER_ROBOT = 3 * nj
    X_warm_start = Vector{Vector{Float64}}(undef, N)

    # Simple warm start: Use loaded q, set dq=0, ddq=0
    for i = 1:N
        x = zeros(Float64, NX)
        q_combined = q_sequence_resampled[i]
        
        # Assign q values
        for r = 1:NUM_ROBOTS
            offset = (r - 1) * NX_PER_ROBOT
            offset_q = (r - 1) * nj
            q = q_combined[offset_q .+ (1:nj)]
            x[offset .+ (1:nj)] = q # Robot r

            # Estimate dq/ddq using finite differences (more complex)
            # For dq: (q[i+1] - q[i-1]) / (2*dt)
            # For ddq: (q[i+1] - 2*q[i] + q[i-1]) / (dt^2)
            # If it is an endpoint, set to 0
            if i > 1 && i < N
                dx_prev = q_sequence_resampled[i-1][offset_q .+ (1:nj)]
                dx_next = q_sequence_resampled[i+1][offset_q .+ (1:nj)]
                dq = (dx_next - dx_prev) / (2*dt) 
                x[offset .+ (nj+1:nj*2)] = dq # Robot r
            end
            if i > 2 && i < N - 1
                dx_prev = q_sequence_resampled[i-1][offset_q .+ (1:nj)]
                dx_next = q_sequence_resampled[i+1][offset_q .+ (1:nj)]
                dx = q_sequence_resampled[i][offset_q .+ (1:nj)]
                ddq = (dx_next - 2*dx + dx_prev) / (dt^2)
                x[offset .+ (nj*2+1:nj*3)] = ddq # Robot r
            end

        end 
        # dq and ddq remain zero
        X_warm_start[i] = x
    end


    return X_warm_start
end