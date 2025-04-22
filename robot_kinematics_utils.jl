# robot_kinematics_utils.jl

using LinearAlgebra
using StaticArrays
using Rotations
using CoordinateTransformations
import DifferentiableCollisions as dc # Needed for constraints
using SparseArrays                   # Needed for sparse Jacobians

# --- Robot Definition ---
# Re-define necessary constants within this file or ensure they are passed/globally available
# It's generally better practice to pass them or define them where used,
# but for simplicity with the existing structure, we define them here.
# Choose Number of Joints (6 or 7)
const NUM_JOINTS_PER_ROBOT = 6 # <--- CHANGE HERE FOR 6 or 7 DOF
const NJ = NUM_JOINTS_PER_ROBOT

# Structure to hold robot kinematics (PoE)
struct RobotKinematics{T, R <: RotMatrix{3, T}}
    twists::Vector{SVector{6, T}} # List of NJ twist coordinates [v; w] in base frame
    T_base::AffineMap{R, SVector{3, T}}   # Transform from world to robot base
    M_frames_zero::Vector{AffineMap{R, SVector{3, T}}} # Pose of frame {i} relative to base {0} at q=0 (M_i)
    T_link_centers::Vector{AffineMap{R, SVector{3, T}}} # Pose of link {L_i} center relative to frame {i} (T_{i, Li})
    link_radius::Vector{T}          # For collision geometry
end

# --- Updated Constants ---
const NUM_ROBOTS = 2
# const NJ defined above
const N_LINKS = NUM_JOINTS_PER_ROBOT # Assume one collision body per joint frame

# State/Control dimensions automatically update based on NJ
const NX_PER_ROBOT = 3 * NJ
const NX = NUM_ROBOTS * NX_PER_ROBOT
const NU_PER_ROBOT = NJ
const NU = NUM_ROBOTS * NU_PER_ROBOT

# Dynamics & Limits (per joint) - Keep these generic for now
const Q_LIMITS = (-π, π) # Generic limits, refine per robot if needed
const DQ_LIMITS = (-2.0, 2.0)
const DDQ_LIMITS = (-5.0, 5.0)
const JERK_LIMITS = (-10.0, 10.0)

# Collision
const COLLISION_THRESHOLD = 1.0 # DCOL proximity value threshold

# --- PoE Helper Functions ---

"""
skew(v) - Computes the skew-symmetric matrix (remains the same)
"""
@inline function skew(v::AbstractVector{T}) where T
    return SMatrix{3, 3, T, 9}(0, v[3], -v[2], -v[3], 0, v[1], v[2], -v[1], 0)
end

"""
exp_twist(twist::AbstractVector{T}, theta::Number) where T

Computes the SE(3) matrix exponential T = exp([ξ]θ) using Rodrigues' formula.
twist = [v; w] (6x1 vector)
"""
function exp_twist(twist::AbstractVector{T}, theta_val::Number) where T
    theta = T(theta_val) # Ensure correct type
    v = SVector{3,T}(twist[1:3])
    w = SVector{3,T}(twist[4:6])
    
    # Add robustness check
    if !isfinite(theta) || !all(isfinite.(v)) || !all(isfinite.(w))
        @warn "Non-finite input to exp_twist: theta=$theta"
        return AffineMap(RotMatrix{3,T}(I), SVector{3,T}(zeros(T, 3)))
    end

    w_norm = norm(w)

    # Declare R explicitly as RotMatrix3{T}
    local R::RotMatrix3{T}
    local p::SVector{3,T}

    if isapprox(w_norm, 0.0; atol=1e-12) # Pure translation
        R = RotMatrix{3,T}(I)
        p = v * theta
    else # General case (Screw motion)
        w_skew = skew(w)
        w_skew_sq = w_skew * w_skew
        # Rodrigues' formula for SO(3)
        R_smatrix = SMatrix{3,3,T}(I) + sin(w_norm * theta) / w_norm * w_skew + (1 - cos(w_norm * theta)) / (w_norm^2) * w_skew_sq
        # Explicitly construct RotMatrix3 from the calculated SMatrix
        R = RotMatrix3{T}(R_smatrix)
        # Translation part
        I3_smatrix = SMatrix{3,3,T}(I)
        p = ( (I3_smatrix - R_smatrix) * w_skew * v + w * dot(w, v) * theta ) / (w_norm^2)
    end
    # return AffineMap(SMatrix{3,3,T,9}(R), SVector{3,T}(p)) # This is more type stable usually
     return AffineMap(R, p)
end

# --- Updated Forward Kinematics (PoE) ---

"""
forward_kinematics_poe(q, robot_kin::RobotKinematics)

Calculates the world pose of each link's center frame using PoE.
Also returns intermediate joint frame poses, world axes, and world positions needed for Jacobian.

Returns:
    link_center_poses_world (Vector{AffineMap}): Pose of each link's center frame in world.
    joint_poses_world (Vector{AffineMap}): Pose of each joint frame in world (T_world_joint_i).
    world_joint_axes (Vector{SVector{3}}): Rotation axis k_j for each joint in world frame.
    world_joint_positions (Vector{SVector{3}}): Position p_j for each joint axis in world frame.
"""
function forward_kinematics_poe(q::AbstractVector{Q_T}, robot_kin::RobotKinematics{T, R}) where {Q_T, T, R<:RotMatrix{3, T}}
    NJ = length(q) # Number of joints (DOF)
    @assert NJ == length(robot_kin.twists) == length(robot_kin.M_frames_zero) - 1 == length(robot_kin.T_link_centers) "Inconsistent kinematic parameters" # M_frames includes base

    link_center_poses_world = Vector{AffineMap{R, SVector{3, T}}}(undef, NJ)
    joint_frames_world = Vector{AffineMap{R, SVector{3, T}}}(undef, NJ + 1) # World pose of frame {i} for i=0 to NJ
    world_joint_axes = Vector{SVector{3, T}}(undef, NJ)
    world_joint_positions = Vector{SVector{3, T}}(undef, NJ)

    # Cumulative product of exponentials, starting with Identity relative to base
    P_cumulative = AffineMap(RotMatrix{3,T}(I), SA[T(0), T(0), T(0)])
    joint_frames_world[1] = robot_kin.T_base # Pose of frame {0} in world is T_base
    T_world_frame_current = robot_kin.T_base # Initialize current world frame pose

    for i = 1:NJ
        # --- Information for Joint i based on Frame {i-1} ---
        # Calculate and Store axis/position for Jacobian *before* updating transform
        twist_i_base = robot_kin.twists[i]
        w_i_base = SVector{3,T}(twist_i_base[4:6])
        # Axis k_i in world frame (using rotation of frame {i-1})
        world_joint_axes[i] = T_world_frame_current.linear * w_i_base
        # Position pos_i (origin of frame {i-1}) in world frame
        world_joint_positions[i] = T_world_frame_current.translation

        # 1. Calculate exponential for joint i
        T_exp_i = exp_twist(robot_kin.twists[i], q[i]) # Transform relative to base due to joint i

        # 2. Update cumulative product P_i = P_{i-1} * exp([ξ_i]q_i)
        P_cumulative = P_cumulative ∘ T_exp_i

        # 3. Calculate world pose of frame {i} (output frame of joint i)
        # T_world_frame_i = T_world_base * P_i * M_i
        # Note: M_frames_zero[i+1] corresponds to M_i
        T_world_frame_next = robot_kin.T_base ∘ P_cumulative ∘ robot_kin.M_frames_zero[i+1]
        joint_frames_world[i+1] = T_world_frame_next # Store pose of frame {i}

        # 4. Calculate world pose of link {L_i}'s center
        # T_world_link_center_i = T_world_frame_i * T_{i, Li}
        T_link_center_relative = robot_kin.T_link_centers[i]
        link_center_poses_world[i] = T_world_frame_next ∘ T_link_center_relative # Use T_world_frame_i (T_world_frame_next)

        # --- Prepare for next iteration ---
        T_world_frame_current = T_world_frame_next # Update for next joint's axis/pos calculation
    end

    return link_center_poses_world, joint_frames_world, world_joint_axes, world_joint_positions
end

# --- Updated Analytical Jacobian (Geometric Approach for 3D) ---

"""
B_matrix_mrp(p) - Remains the same
"""
@inline function B_matrix_mrp(p::AbstractVector{T}) where T
    p_sq_norm = dot(p, p)
    I3 = SMatrix{3,3,T,9}(1.0I)
    p_outer = p * p'
    p_skew = skew(p)
    B = T(0.25) * ((T(1.0) - p_sq_norm) * I3 + T(2.0) * p_skew + T(2.0) * p_outer)
    return B
end

"""
calculate_link_pose_jacobian_geom(q_r, link_idx, robot_kin)

Calculates the analytical Jacobian (6xNJ) of a link's pose [r; p]
with respect to the robot's joint angles q_r using the geometric method.
"""
function calculate_link_pose_jacobian_geom(q_r::AbstractVector{T}, link_idx::Int, robot_kin::RobotKinematics{T_kin}) where {T, T_kin}
    NJ = length(q_r)
    nx_pose = 6 # Size of [r; p]
    J_pose = zeros(T, nx_pose, NJ)

    # --- Perform FK to get necessary info ---
    link_center_poses, _, world_joint_axes, world_joint_positions = forward_kinematics_poe(q_r, robot_kin)

    # Target link's center position and orientation (MRP) in world frame
    T_link_center_world = link_center_poses[link_idx]
    r_i = T_link_center_world.translation
    R_i = T_link_center_world.linear
    p_i = Rotations.params(Rotations.MRP(R_i))

    # --- Calculate Jacobians ---
    J_w = @view J_pose[4:6, :] # Angular velocity Jacobian part

    # Position Jacobian J_r (Rows 1-3) & Angular Jacobian J_w (Rows 4-6)
    for j = 1:NJ # Iterate through joints affecting the pose
        k_j = world_joint_axes[j]    # World axis vector for joint j
        pos_j = world_joint_positions[j] # World position of joint j origin

        # Check if prismatic or revolute (based on twist w component)
        # Assuming revolute for now based on Yaskawa example
        # If prismatic, Jv_j = k_j, Jw_j = 0
        is_revolute = norm(k_j) > 1e-9 # Simple check

        if is_revolute
             # Ensure k_j is normalized if using it directly for magnitude
            # k_j_norm = normalize(k_j) # Use normalized axis if twist wasn't unit vector
             k_j_norm = k_j # Assume twist w was unit vector

             # Geometric Jacobian columns
             Jv_j = cross(k_j_norm, r_i - pos_j)
             Jw_j = k_j_norm # World angular velocity contribution
             J_pose[1:3, j] = Jv_j
             J_w[:, j] = Jw_j # Assign to the view
        else # Prismatic
             @warn "Prismatic joints not fully handled in Jacobian yet"
             # For simplicity, assume all are revolute matching GP4 example
             J_pose[1:3, j] = k_j
             J_w[:, j] = zeros(T, 3)
        end
    end

    # Orientation Jacobian J_p = B(p_i) * J_w (Rows 4-6)
    # This overwrites the J_w we just calculated, using it as input
    B_pi = B_matrix_mrp(SVector{3,T}(p_i))
    J_p = B_pi * J_w # Calculate J_p = B * (angular part of geometric Jacobian)
    J_pose[4:6, :] = J_p

    return J_pose
end

# --- Constraints ---

# `ineq_con_x` needs to use the new FK function
function ineq_con_x(p::NamedTuple, x::AbstractVector)
    constraints = Float64[]

    # 1. Limits (Pos, Vel, Accel) - Structure depends only on NJ
    for r = 1:NUM_ROBOTS
        offset = (r - 1) * NX_PER_ROBOT
        q = x[offset .+ (1:NJ)]
        dq = x[offset .+ (NJ + 1 : 2*NJ)]
        ddq = x[offset .+ (2*NJ + 1 : 3*NJ)]
        # Use p.q_lim which should be defined per-robot if limits differ
        q_lim_robot = p.q_lim[r] # Assuming q_lim is now a vector of tuples/vectors
        dq_lim_robot = p.dq_lim[r]
        ddq_lim_robot = p.ddq_lim[r]
        append!(constraints, q .- q_lim_robot[2]); append!(constraints, q_lim_robot[1] .- q)
        append!(constraints, dq .- dq_lim_robot[2]); append!(constraints, dq_lim_robot[1] .- dq)
        append!(constraints, ddq .- ddq_lim_robot[2]); append!(constraints, ddq_lim_robot[1] .- ddq)
    end

    # Calculate poses using PoE FK
    q1 = x[1:NJ]
    q2 = x[NX_PER_ROBOT .+ (1:NJ)]
    # Assuming p contains robot_kin structs: p.robot_kin[1], p.robot_kin[2]
    poses1_world, _, _, _ = forward_kinematics_poe(q1, p.robot_kin[1])
    poses2_world, _, _, _ = forward_kinematics_poe(q2, p.robot_kin[2])

    # Update collision primitive poses
    all_poses_world = [poses1_world, poses2_world]
    for r = 1:NUM_ROBOTS
        poses_r = all_poses_world[r]
        for i = 1:N_LINKS # N_LINKS should match NJ here
            T_link_center = poses_r[i] # This is already the center pose
            pos = T_link_center.translation
            rot = RotMatrix(T_link_center.linear)
            mrp = Rotations.MRP(rot); mrp_vec = Rotations.params(mrp)
            # p.P_links[r] should be a vector of primitives for robot r
            p.P_links[r][i].r = pos
            p.P_links[r][i].p = mrp_vec
        end
    end

    # 2. Inter-Robot Collision Avoidance
    for i = 1:N_LINKS
        for j = 1:N_LINKS
            prox, _ = dc.proximity(p.P_links[1][i], p.P_links[2][j])
            push!(constraints, p.collision_threshold - prox)
        end
    end

    # 3. Self-Collision Avoidance 
    # for r = 1:NUM_ROBOTS
    #     # Iterate through the pairs defined in params
    #     for (i, j) in p.self_collision_pairs
    #          if 1 <= i <= N_LINKS && 1 <= j <= N_LINKS
    #              Pi = p.P_links[r][i]
    #              Pj = p.P_links[r][j]
    #              prox_self, _ = dc.proximity(Pi, Pj)
    #              push!(constraints, p.collision_threshold - prox_self)
    #          else
    #             @warn "Invalid indices in self_collision_pairs: ($i, $j)"
    #          end
    #     end
    # end

    return constraints
end

# `ineq_con_x_jac` needs to use the new Jacobian function
function ineq_con_x_jac(p::NamedTuple, x::AbstractVector)
    nx = p.nx
    ncx = p.ncx # Use pre-calculated ncx for matrix sizing
    NJ = NUM_JOINTS_PER_ROBOT

    J_x = zeros(Float64, ncx, nx)
    current_row = 0

    # --- 1. Jacobians for Limits --- (Structure unchanged, ensure q_lim etc index correctly)
    I_NJ_sparse = sparse(I, NJ, NJ)
    for r = 1:NUM_ROBOTS
        offset_x_q = (r - 1) * NX_PER_ROBOT; offset_x_dq = offset_x_q + NJ; offset_x_ddq = offset_x_dq + NJ
        rows_q_upper = current_row .+ (1:NJ); rows_q_lower = rows_q_upper .+ NJ
        rows_dq_upper = rows_q_lower .+ NJ; rows_dq_lower = rows_dq_upper .+ NJ
        rows_ddq_upper = rows_dq_lower .+ NJ; rows_ddq_lower = rows_ddq_upper .+ NJ
        cols_q = offset_x_q .+ (1:NJ); cols_dq = offset_x_dq .+ (1:NJ); cols_ddq = offset_x_ddq .+ (1:NJ)
        J_x[rows_q_upper, cols_q] = I_NJ_sparse; J_x[rows_q_lower, cols_q] = -I_NJ_sparse
        J_x[rows_dq_upper, cols_dq] = I_NJ_sparse; J_x[rows_dq_lower, cols_dq] = -I_NJ_sparse
        J_x[rows_ddq_upper, cols_ddq] = I_NJ_sparse; J_x[rows_ddq_lower, cols_ddq] = -I_NJ_sparse
        current_row = rows_ddq_lower[end]
    end

    # --- Update Primitive Poses (Needed for proximity_gradient) ---
    q1 = x[1:NJ]
    q2 = x[NX_PER_ROBOT .+ (1:NJ)]
    poses1_world, _, _, _ = forward_kinematics_poe(q1, p.robot_kin[1])
    poses2_world, _, _, _ = forward_kinematics_poe(q2, p.robot_kin[2])
    all_poses_world = [poses1_world, poses2_world]
    for r = 1:NUM_ROBOTS
        for i = 1:N_LINKS
            T_link_center = all_poses_world[r][i]
            pos = T_link_center.translation
            rot = RotMatrix(T_link_center.linear)
            mrp = Rotations.MRP(rot); mrp_vec = Rotations.params(mrp)
            p.P_links[r][i].r = pos; p.P_links[r][i].p = mrp_vec
        end
    end

    # --- 2. Jacobians for Inter-Robot Collisions ---
    q1_indices = 1:NJ
    q2_indices = NX_PER_ROBOT .+ (1:NJ)
    pose1_slice = 1:6; pose2_slice = 7:12

    for i = 1:N_LINKS
        for j = 1:N_LINKS
            current_row += 1
            if current_row > ncx; @warn "Jacobian row overflow (inter)!"; break; end
            P1 = p.P_links[1][i]; P2 = p.P_links[2][j]
            prox_val, J_prox_combined = dc.proximity_gradient(P1, P2)
            J_prox_P1 = J_prox_combined[pose1_slice]
            J_prox_P2 = J_prox_combined[pose2_slice]

            # Use new analytical FK Jacobian
            J_fk1 = calculate_link_pose_jacobian_geom(q1, i, p.robot_kin[1]) # Use geom version
            J_fk2 = calculate_link_pose_jacobian_geom(q2, j, p.robot_kin[2]) # Use geom version

            J_x[current_row, q1_indices] = -J_prox_P1' * J_fk1
            J_x[current_row, q2_indices] = -J_prox_P2' * J_fk2
        end
         if current_row > ncx; break; end
    end

#     # --- 3. Jacobians for Self-Collisions --- (Skipping detail, requires pairs and careful indexing)
#     for r = 1:NUM_ROBOTS
#         q_r_indices = (r-1)*NX_PER_ROBOT .+ (1:NJ)
#         q_r = x[q_r_indices]
#         # Iterate through the pairs defined in params
#         for (i, j) in p.self_collision_pairs
#              # Check validity again just in case filtering failed or ncx mismatch
#              if 1 <= i <= N_LINKS && 1 <= j <= N_LINKS
#                    current_row += 1
#                    if current_row > ncx; @warn "Jacobian row overflow (self)!"; break; end

#                    Pi = p.P_links[r][i]; Pj = p.P_links[r][j]
#                    prox_val, J_prox_combined = dc.proximity_gradient(Pi, Pj)
#                    J_prox_Pi = J_prox_combined[pose1_slice]
#                    J_prox_Pj = J_prox_combined[pose2_slice]
#                    J_fki = calculate_link_pose_jacobian_geom(q_r, i, p.robot_kin[r])
#                    J_fkj = calculate_link_pose_jacobian_geom(q_r, j, p.robot_kin[r])
#                    # Chain rule for self-collision: both J_fki and J_fkj depend on q_r
#                    J_x[current_row, q_r_indices] = -(J_prox_Pi' * J_fki + J_prox_Pj' * J_fkj)
#              end
#         end
#         if current_row > ncx; break; end # Break outer loop if overflow
#    end

    # Final sanity check
    if current_row != ncx && ncx > 0 # Allow ncx=0 if no collision constraints were added
         num_limit_cons = NUM_ROBOTS * 2 * 3 * NJ
         num_inter_coll = NUM_ROBOTS * (NUM_ROBOTS-1) / 2 * N_LINKS * N_LINKS # Approximation
         # num_self_coll = ... depends on pairs
         expected_rows = num_limit_cons + N_LINKS^2 # Only inter-robot
         if current_row != expected_rows
            @warn "Row count mismatch! Expected $ncx (from eval), calculated $current_row, expected based on loops ~$expected_rows"
         end
    end

    return J_x
end


"Inequality constraints on control u (jerk limits)."
function ineq_con_u(p::NamedTuple, u::AbstractVector)
    if p.min_time
        # u - u_max <= 0; -u + u_min <= 0
        h = u[1]
        u_jerk = u[2:end]
        
        # Constraints for time scaling factor (must be positive)
        h_constraints = [h - p.h_max; -h + p.h_min]
        
        # Constraints for jerk controls
        jerk_constraints = vcat(u_jerk .- p.u_max_jerk, p.u_min_jerk .- u_jerk)
        return vcat(h_constraints, jerk_constraints)
    else
        # u - u_max <= 0; -u + u_min <= 0
        return vcat(u .- p.u_max, p.u_min .- u)
    end
end

"Jacobian of inequality constraints w.r.t. control u."
function ineq_con_u_jac(p::NamedTuple, u::AbstractVector)
    if p.min_time
        nu = p.nu
        nu_jerk = nu - 1
        
        # Create the Jacobian matrix
        # 2 rows for h constraints + 2*nu_jerk rows for jerk constraints
        J_u = zeros(2 + 2*nu_jerk, nu)
        
        # Derivatives for h constraints
        J_u[1, 1] = 1.0  # ∂(h - h_max)/∂h = 1
        J_u[2, 1] = -1.0 # ∂(-h + h_min)/∂h = -1
        
        # Derivatives for jerk constraints
        for i = 1:nu_jerk
            J_u[2 + i, 1 + i] = 1.0  # ∂(u_jerk_i - u_max_i)/∂u_jerk_i = 1
            J_u[2 + nu_jerk + i, 1 + i] = -1.0  # ∂(-u_jerk_i + u_min_i)/∂u_jerk_i = -1
        end
        
        return J_u
    else
        # Simple structure for box constraints
        return Array(float([I(p.nu); -I(p.nu)]))
    end
end


# --- Dynamics ---

"Triple integrator dynamics for all joints of all robots."
function dynamics(p::NamedTuple, x::AbstractVector, u::AbstractVector, k)
    # Extract derivatives and controls for Robot 1
    dq1   = x[ (NJ + 1) : (2*NJ) ]          # Velocities of Robot 1
    ddq1  = x[ (2*NJ + 1) : (3*NJ) ]       # Accelerations of Robot 1
    jerk1 = u[ 1 : NJ ]                   # Jerks (controls) for Robot 1

    # Extract derivatives and controls for Robot 2
    offset_r2_x = NX_PER_ROBOT
    offset_r2_u = NU_PER_ROBOT
    dq2   = x[ (offset_r2_x + NJ + 1) : (offset_r2_x + 2*NJ) ] # Velocities of Robot 2
    ddq2  = x[ (offset_r2_x + 2*NJ + 1) : (offset_r2_x + 3*NJ) ]# Accelerations of Robot 2
    jerk2 = u[ (offset_r2_u + 1) : (offset_r2_u + NU_PER_ROBOT) ] # Jerks (controls) for Robot 2

    # Construct and return the state derivative vector dx = [dq1; ddq1; jerk1; dq2; ddq2; jerk2]
    # Using vcat is explicit and clear.
    # The [...] syntax also works: [dq1; ddq1; jerk1; dq2; ddq2; jerk2]
    # Both construct a *new* vector without mutation.
    return vcat(dq1, ddq1, jerk1, dq2, ddq2, jerk2)
end


"Discrete-time dynamics using RK4 integration."
function discrete_dynamics(p::NamedTuple, x::AbstractVector, u::AbstractVector, k)
    if p.min_time
        # h = u[1]
        h = clamp(u[1], p.h_min, p.h_max)
        k1 = dynamics(p, x, u[2:end], k) * h
        k2 = dynamics(p, x + k1 / 2, u[2:end], k) * h
        k3 = dynamics(p, x + k2 / 2, u[2:end], k) * h
        k4 = dynamics(p, x + k3, u[2:end], k) * h
        return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    else
        k1 = p.dt * dynamics(p, x, u, k)
        k2 = p.dt * dynamics(p, x + k1 / 2, u, k)
        k3 = p.dt * dynamics(p, x + k2 / 2, u, k)
        k4 = p.dt * dynamics(p, x + k3, u, k)
        return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
end