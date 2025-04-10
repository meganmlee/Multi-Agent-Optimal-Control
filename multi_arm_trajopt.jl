using Pkg
Pkg.add(["LinearAlgebra", "StaticArrays", "ForwardDiff", "Printf", "SparseArrays", "MeshCat", "Random", "Colors", "Rotations", "CoordinateTransformations"])

import DifferentiableCollisions as dc
using LinearAlgebra
using StaticArrays
import ForwardDiff as FD
using Printf
using SparseArrays
import MeshCat as mc
import Random
using Colors
using Rotations
using CoordinateTransformations

include("min_time_altro.jl")

# --- Robot Definition ---

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
    # u - u_max <= 0; -u + u_min <= 0
    h = u[1]
    u_jerk = u[2:end]
    
    # Constraints for time scaling factor (must be positive)
    h_constraints = [h - p.h_max; -h + p.h_min]
    
    # Constraints for jerk controls
    jerk_constraints = vcat(u_jerk .- p.u_max, p.u_min .- u_jerk)
    return vcat(h_constraints, jerk_constraints)
end

"Jacobian of inequality constraints w.r.t. control u."
function ineq_con_u_jac(p::NamedTuple, u::AbstractVector)
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
    h = u[1]
    k1 = dynamics(p, x, u[2:end], k) * h
    k2 = dynamics(p, x + k1 / 2, u[2:end], k) * h
    k3 = dynamics(p, x + k2 / 2, u[2:end], k) * h
    k4 = dynamics(p, x + k3, u[2:end], k) * h
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

# --- Main Function Update ---
function main()
    let
        # --- Define Robot Kinematics (Example: Generic 6-DOF) ---
        T = Float64
        # Twists ξᵢ (Defined in base frame {0} at q=0)
        w1 = SA[T(0), T(0), T(1)]; p1 = SA[T(0), T(0), T(0)]
        w2 = SA[T(0), T(1), T(0)]; p2 = SA[T(0), T(0), T(0.33)]
        w3 = SA[T(0), T(1), T(0)]; p3 = SA[T(0), T(0), T(0.33+0.26)]
        w4 = SA[T(1), T(0), T(0)]; p4 = SA[T(0.1), T(0), T(0.33+0.26+0.015)]
        w5 = SA[T(0), T(-1), T(0)]; p5 = SA[T(0.1+0.19), T(0), T(0.33+0.26+0.015)]
        w6 = SA[T(1), T(0), T(0)]; p6 = SA[T(0.1+0.19+0.072), T(0), T(0.33+0.26+0.015)]
        twists_base = [
            vcat(-cross(w1, p1), w1), vcat(-cross(w2, p2), w2), vcat(-cross(w3, p3), w3),
            vcat(-cross(w4, p4), w4), vcat(-cross(w5, p5), w5), vcat(-cross(w6, p6), w6)
        ]
        if NUM_JOINTS_PER_ROBOT == 7; # Add 7th DOF if needed
                w7 = SA[T(0), T(1), T(0)]; p7 = p6 # Example 7th joint at same location as 6
                push!(twists_base, vcat(-cross(w7, p7), w7))
        end

        # M_frames_zero: Pose of frame {i} relative to base {0} at q=0
        # Frame {i} is attached to link i (output of joint i), often placed at origin of joint i+1
        # M_frames_zero[1] is pose of frame {0} relative to {0} -> Identity
        # M_frames_zero[i+1] is pose of frame {i} relative to {0} at q=0
        # This requires knowing the exact geometry/DH params/URDF at q=0
        # Let's construct it based on the points p used above (p_i = origin of joint i+1 relative to base 0)
        M_frames_zero = Vector{AffineMap{RotMatrix3{T}, SVector{3, T}}}(undef, NJ + 1)
        M_frames_zero[1] = AffineMap(RotMatrix{3,T}(I), SA[T(0),T(0),T(0)]) # Frame {0} rel {0}
        # Need points p0..pNJ (p0=p1, p1=p2, ..., p_{NJ}=EndEffector)
        points_p = [p1, p2, p3, p4, p5, p6] # Points defining joint i+1 origin relative to base
        if NUM_JOINTS_PER_ROBOT == 7; push!(points_p, p7); end # Add point for frame 7 origin
        # Assume frame {i} is located at origin of joint {i+1} (point points_p[i]) with same orientation as base at q=0
        # we set the orientation so that the capsule's x-axis is aligned with the z-axis of the joint frame {i}
        M_frames_zero[2] = AffineMap(RotMatrix{3,T}(I), points_p[1]) # Frame {1} rel {0}
        M_frames_zero[3] = AffineMap(RotMatrix{3,T}(I), points_p[2]) # Frame {2} rel {0}
        M_frames_zero[4] = AffineMap(RotMatrix(RotY(π/2)), points_p[3]) # Frame {3} rel {0}
        M_frames_zero[5] = AffineMap(RotMatrix(RotY(π/2)), points_p[4]) # Frame {4} rel {0}
        M_frames_zero[6] = AffineMap(RotMatrix(RotY(π/2)), points_p[5]) # Frame {5} rel {0}
        M_frames_zero[7] = AffineMap(RotMatrix(RotY(π/2)), points_p[6]) # Frame {6} rel {0}
        if NUM_JOINTS_PER_ROBOT == 7; M_frames_zero[8] = AffineMap(RotMatrix(RotY(π/2)), p7); end # Frame {7} rel {0}

        # hard code the link length values for collision capsule
        link_lengths_geom = [0.33, 0.26, 0.1, 0.19, 0.072, 0.11] # Example lengths (L1, L2, L3, L4, L5, L6)
        # define the transmoation from frame{i} to the center of the collision capsule
        T_link_centers = Vector{AffineMap{RotMatrix3{T}, SVector{3, T}}}(undef, NJ)
        for i = 1:NJ
            # Offset from frame {i} (at joint i+1) back to center of link i
            # Needs link vector direction in frame {i}. Assume frame {i} x-axis points back along link
            L_i = link_lengths_geom[i]
            T_link_centers[i] = AffineMap(RotMatrix(RotY(π/2)), SA[T(0.0), T(0.0), T(L_i/2.0)])
            # !! This assumes orientation of frame {i} aligns with link i - needs verification !!
        end


        # Radii for collision geometry
        vis_link_radii = [0.095, 0.095, 0.06, 0.06, 0.05, 0.04] # Example radii
        if NUM_JOINTS_PER_ROBOT == 7; push!(vis_link_radii, 0.005); end

        # Base transforms
        T_base1 = AffineMap(RotMatrix(RotZ(0.0)), SA[T(0.0), T(0.0), T(0)]) # red
        T_base2 = AffineMap(RotMatrix(RotZ(π)), SA[T(0.88101), T(-0.01304), T(0)]) # blue

        # Create Kinematics structs
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


        # --- Simulation Parameters ---
        N = 51 # Fewer steps initially for faster debugging
        dt = 0.1

        # --- Start and Goal States (Adjust for 6/7 DOF) ---
        q_start1 = [0.0, 0, 0.0, 0.0, -π/6, 0.0]  # Robot 1 start configuration
        q_goal1 = [0.0, π/4, 0.0,  0.0, -π/6, 0.0]  # Robot 1 goal configuration

        q_start2 = [0.0, π/4, 0.0,  0.0, -π/6, 0.0]  # Robot 2 start configuration
        q_goal2 = [0.0, 0, 0.0, 0.0, -π/6, 0.0]  # Robot 2 goal configuration

        x0 = vcat(q_start1, zeros(NJ*2), q_start2, zeros(NJ*2))
        xg = vcat(q_goal1, zeros(NJ*2), q_goal2, zeros(NJ*2))
        Xref = [deepcopy(xg) for i = 1:N]
        Uref = [[3.0; zeros(nu_jerk)] for i = 1:N-1]

        # --- Cost Matrices (Updated dimensions) ---
        Q = Diagonal(zeros(T, NX))
        Qf_q_weight = 100.0; Qf_dq_weight = 10.0; Qf_ddq_weight = 1.0
        Qf_diag_single = vcat(fill(Qf_q_weight, NJ), fill(Qf_dq_weight, NJ), fill(Qf_ddq_weight, NJ))
        Qf = Diagonal(vcat(Qf_diag_single, Qf_diag_single))
        R_jerk_weight = 0.01; R = R_jerk_weight * Diagonal(ones(T, NU+1)) # add one for time 

        # --- Constraint Limits (Updated structure) ---
        u_min_jerk = fill(T(JERK_LIMITS[1]), NU); u_max_jerk = fill(T(JERK_LIMITS[2]), NU)
        h_min = 0.1; h_max = 3.0
        # Assuming same limits for all joints, structure as vector of tuples
        q_lim_single = (fill(T(Q_LIMITS[1]), NJ), fill(T(Q_LIMITS[2]), NJ))
        dq_lim_single = (fill(T(DQ_LIMITS[1]), NJ), fill(T(DQ_LIMITS[2]), NJ))
        ddq_lim_single = (fill(T(DDQ_LIMITS[1]), NJ), fill(T(DDQ_LIMITS[2]), NJ))
        q_lim_all = [q_lim_single, q_lim_single] # For robot 1 and 2
        dq_lim_all = [dq_lim_single, dq_lim_single]
        ddq_lim_all = [ddq_lim_single, ddq_lim_single]

        # --- Calculate Constraint Dimensions ---
        temp_params_for_eval = (; robot_kin=[robot_kin1, robot_kin2], q_lim=q_lim_all, dq_lim=dq_lim_all, ddq_lim=ddq_lim_all, 
        P_links, collision_threshold = T(COLLISION_THRESHOLD), 
        nu=NU, u_min, u_max, self_collision_pairs)
        ncx = length(ineq_con_x(temp_params_for_eval, x0))
        ncu = length(ineq_con_u(temp_params_for_eval, Uref[1]))
        println("Calculated ncx = $ncx, ncu = $ncu")
        if ncx == 0; @warn("ncx is zero, check ineq_con_x implementation!"); end

        # --- Parameters Bundle ---
        params = (
            nx = NX, nu = NU, ncx = ncx, ncu = ncu, N = N,
            Q = Q, R = R, Qf = Qf,
            u_min_jerk = u_min_jerk, u_max_jerk = u_max_jerk,
            h_min = h_min, h_max = h_max,
            q_lim = q_lim_all, dq_lim = dq_lim_all, ddq_lim = ddq_lim_all, # Pass the structured limits
            Xref = Xref, Uref = Uref, dt = dt,
            P_links = P_links,
            robot_kin = [robot_kin1, robot_kin2], # Pass kinematics parameters
            collision_threshold = T(COLLISION_THRESHOLD),
            self_collision_pairs = self_collision_pairs # Pass self-collision pairs
        );

        # --- Initial Trajectory Guess ---
        X = [deepcopy(x0) + ((i-1)/(N-1))*(xg - x0) for i = 1:N]
        U = [[1.0; zeros(T, NU)] for i = 1:N-1]

        # --- iLQR Variables ---
        Xn = deepcopy(X); Un = deepcopy(U)
        P = [zeros(T, NX, NX) for i = 1:N]; p = [zeros(T, NX) for i = 1:N]
        d = [zeros(T, NU) for i = 1:N-1]; K = [zeros(T, NU, NX) for i = 1:N-1]

        # --- Run iLQR ---
        println("Starting iLQR Optimization (3D Robot)...")
        Xhist = iLQR(params, X, U, P, p, K, d, Xn, Un;
                    atol=1e-1, max_iters=3000, verbose=true, ρ=1.0, ϕ=10.0) # Adjust params for potentially harder problem

        println("iLQR Finished.")
        X_sol = Xhist[end]

        # --- Visualization ---
        println("Setting up Visualization...")
        vis = mc.Visualizer()
        mc.open(vis)
        mc.setprop!(vis["/Background"], "top_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
        mc.setprop!(vis["/Background"], "bottom_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
        mc.delete!(vis["/Grid"]); mc.delete!(vis["/Axes"])

        colors = [mc.RGBA(1.0, 0.0, 0.0, 0.8), mc.RGBA(0.0, 0.0, 1.0, 0.8)]
        link_vis_names = [[Symbol("R$(r)_L$(l)") for l=1:N_LINKS] for r=1:NUM_ROBOTS]

        # Build visual primitives using DCOL helper
        println("Building visual primitives...")
        for r = 1:NUM_ROBOTS
            for l = 1:N_LINKS
                primitive_obj = params.P_links[r][l]
                vis_name = link_vis_names[r][l]
                vis_color = colors[r]
                # print the primitive
                dc.build_primitive!(vis, primitive_obj, vis_name; color = vis_color)
            end
             # Optionally add base visualization
             dc.build_primitive!(vis, dc.SphereMRP(0.01), Symbol("Base$r"); color=colors[r]) # Small sphere at base
             base_prim = dc.SphereMRP(0.01)
             base_prim.r = params.robot_kin[r].T_base.translation
             dc.update_pose!(vis[Symbol("Base$r")], base_prim)
        end

        # Animate using DCOL helper
        anim = mc.Animation(floor(Int, 1 / dt))
        println("Creating Animation...")
        for k = 1:N
            mc.atframe(anim, k) do
                x_k = X_sol[k]
                q1_k = x_k[1:NJ]
                q2_k = x_k[NX_PER_ROBOT .+ (1:NJ)]
                all_q_k = [q1_k, q2_k]

                for r = 1:NUM_ROBOTS
                    # Calculate FK for current robot
                    poses_world_k, _, _, _ = forward_kinematics_poe(all_q_k[r], params.robot_kin[r])

                    for l = 1:N_LINKS
                        primitive_obj = params.P_links[r][l]
                        T_link_center = poses_world_k[l]
                        pos = T_link_center.translation
                        rot = RotMatrix(T_link_center.linear)

                        # Update DCOL primitive state
                        primitive_obj.r = pos
                        primitive_obj.p = Rotations.params(Rotations.MRP(rot))

                        # Update Meshcat visualization
                        vis_name = link_vis_names[r][l]
                        dc.update_pose!(vis[vis_name], primitive_obj)
                    end
                end
            end
        end
        println("Setting Animation...")
        mc.setanimation!(vis, anim)
        println("Done. Check MeshCat visualizer.")
    end # End let
end # End main

main()