#!/usr/bin/env julia

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.add(["LinearAlgebra", "StaticArrays", "ForwardDiff", "FiniteDiff", "Printf", "SparseArrays", 
         "MeshCat", "Random", "Colors", "Rotations", "CoordinateTransformations", 
         "Ipopt", "MathOptInterface"])

import DifferentiableCollisions as dc
using Statistics
using LinearAlgebra
using StaticArrays
import ForwardDiff as FD
import FiniteDiff
using Printf
using SparseArrays
import MeshCat as mc
import Random
using Colors
using Rotations
using CoordinateTransformations
import MathOptInterface as MOI
import Ipopt
include(joinpath(@__DIR__,"fmincon.jl"))

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
const NU = NUM_ROBOTS * NU_PER_ROBOT + 1 # add one for time variable

# Dynamics & Limits (per joint) - Keep these generic for now
const Q_LIMITS = (-π, π) # Generic limits, refine per robot if needed
const DQ_LIMITS = (-2.0, 2.0)
const DDQ_LIMITS = (-5.0, 5.0)
const JERK_LIMITS = (-10.0, 10.0)

# Collision
const COLLISION_THRESHOLD = 1.0 # DCOL proximity value threshold

# Time (h rnage)
const TIME_SCALING_LIMITS = (0.3, 5.0)

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
    theta = theta_val
    v = SVector{3,T}(twist[1:3])
    w = SVector{3,T}(twist[4:6])
    w_norm = norm(w)

    # Declare R and p with more generic types
    local R
    local p

    if isapprox(w_norm, 0.0; atol=1e-12) # Pure translation
        R = RotMatrix{3,typeof(theta)}(I)
        p = v * theta
    else # General case (Screw motion)
        w_skew = skew(w)
        w_skew_sq = w_skew * w_skew
        # Rodrigues' formula for SO(3)
        R_smatrix = SMatrix{3,3,typeof(theta)}(I) + sin(w_norm * theta) / w_norm * w_skew + (1 - cos(w_norm * theta)) / (w_norm^2) * w_skew_sq
        # Use more generic type
        R = RotMatrix{3,typeof(theta)}(R_smatrix)
        # Translation part
        I3_smatrix = SMatrix{3,3,typeof(theta)}(I)
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

# Create indexing variables for optimization vector and constraints
function create_idx(nx, nu, N)
    # Z vector is [x0, u0, x1, u1, ..., xN]
    nz = (N-1) * nu + N * nx
    x = [(i - 1) * (nx + nu) .+ (1:nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ nx .+ (1:nu) for i = 1:(N-1)]
    
    # Constraints indexing for dynamics constraints
    c_dyn = [(i - 1) * nx .+ (1:nx) for i = 1:(N-1)]
    nc_dyn = (N - 1) * nx
    
    return (nx=nx, nu=nu, N=N, nz=nz, nc_dyn=nc_dyn, x=x, u=u, c_dyn=c_dyn)
end

function robot_dynamics(x, u)
    # Time scaling 
    h = u[1]

    # Robot 1
    dq1 = x[NJ+1:2*NJ]
    ddq1 = x[2*NJ+1:3*NJ]
    
    # Robot 2
    dq2 = x[NX_PER_ROBOT+NJ+1:NX_PER_ROBOT+2*NJ]
    ddq2 = x[NX_PER_ROBOT+2*NJ+1:NX_PER_ROBOT+3*NJ]
    
    # Extract control inputs (jerks)
    jerk1 = u[2:NJ+1]
    jerk2 = u[NU_PER_ROBOT+2:NU]
    
    # Triple integrator dynamics
    xdot = vcat(
        h .* dq1, h .* ddq1, h .* jerk1,  # Robot 1
        h .* dq2, h .* ddq2, h .* jerk2   # Robot 2
    )
    
    return xdot
end

# Hermite-Simpson collocation constraint
function hermite_simpson(params, x1, x2, u, dt)
    # Evaluate dynamics at the endpoints
    f1 = robot_dynamics(x1, u)
    f2 = robot_dynamics(x2, u)
    
    # Compute the midpoint state
    x_mid = 0.5 * (x1 + x2) + (dt / 8) * (f1 - f2)
    
    # Evaluate dynamics at the midpoint
    f_mid = robot_dynamics(x_mid, u)

    return x1 + (dt / 6) * (f1 + 4 * f_mid + f2) - x2
end

# Cost function for optimization
function cost(params, Z)
    idx, N, xg = params.idx, params.N, params.xg
    Q, R, Qf = params.Q, params.R, params.Qf
    time_cost_weight = params.time_cost_weight
    
    J = 0.0
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]
        h = ui[1]

        J += 0.5 * ((xi - xg)' * Q * (xi - xg))

        # Control cost for jerks
        jerks = ui[2:end]
        J += 0.5 * (jerks' * R[2:end, 2:end] * jerks)
        
        # Time cost
        J += time_cost_weight * h
    end
    
    # terminal cost 
    xn = Z[idx.x[N]]
    J += 0.5 * (xn - xg)' * Qf * (xn - xg)
    
    return J 
end

# Dynamics constraints
function dynamics_constraints(params, Z)
    idx, N, dt = params.idx, params.N, params.dt
    
    # Create dynamics constraints using Hermite-Simpson
    c = zeros(eltype(Z), idx.nc_dyn)
    
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]] 
        xip1 = Z[idx.x[i+1]]
        
        c[idx.c_dyn[i]] = hermite_simpson(params, xi, xip1, ui, dt)
    end
    
    return c 
end

# All equality constraints
function equality_constraint(params, Z)
    N, idx, x0, xg = params.N, params.idx, params.x0, params.xg 
    
    # Dynamics constraints
    c_dyn = dynamics_constraints(params, Z)
    
    # Initial state constraint
    c_init = Z[idx.x[1]] - x0
    
    # Final state constraint for position only, velocity and acceleration are free
    c_final_pos1 = Z[idx.x[N]][1:NJ] - xg[1:NJ]
    c_final_pos2 = Z[idx.x[N]][NX_PER_ROBOT+1:NX_PER_ROBOT+NJ] - xg[NX_PER_ROBOT+1:NX_PER_ROBOT+NJ]
    
    # Stack all equality constraints
    return [c_dyn; c_init; c_final_pos1; c_final_pos2]
end

# Collision and state limits inequality constraints
function inequality_constraint(params, Z)
    N, idx = params.N, params.idx
    robot_kin1, robot_kin2 = params.robot_kin[1], params.robot_kin[2]
    P_links = params.P_links
    collision_threshold = params.collision_threshold
    
    # Collision constraints between robots
    n_collision_constraints = N * N_LINKS * N_LINKS
    
    # Total inequality constraints
    n_ineq = n_collision_constraints
    
    # Initialize constraint vector with correct size
    c = zeros(eltype(Z), n_ineq)
    constraint_idx = 1 
         
    # 1. Collision Avoidance Constraints
    for i = 1:N
        x_i = Z[idx.x[i]]
        
        # Extract states for both robots
        q1 = x_i[1:NJ]
        q2 = x_i[NX_PER_ROBOT .+ (1:NJ)]
        
        # Get forward kinematics for both robots
        poses1_world, _, _, _ = forward_kinematics_poe(q1, robot_kin1)
        poses2_world, _, _, _ = forward_kinematics_poe(q2, robot_kin2)
        
        # Update primitive poses for collision checking
        all_poses_world = [poses1_world, poses2_world]
        for r = 1:NUM_ROBOTS
            poses_r = all_poses_world[r]
            for j = 1:N_LINKS
                T_link_center = poses_r[j]
                pos = T_link_center.translation
                rot = RotMatrix(T_link_center.linear)
                mrp = Rotations.MRP(rot)
                mrp_vec = Rotations.params(mrp)
                
                # Update the primitives
                P_links[r][j].r = pos
                P_links[r][j].p = mrp_vec
            end
        end
        
        # Check for collisions between robots
        for link1 = 1:N_LINKS
            for link2 = 1:N_LINKS
                # Calculate proximity (negative means collision)
                prox, _ = dc.proximity(P_links[1][link1], P_links[2][link2])
                
                # Constraint: prox - collision_threshold > 0
                # We flip the sign to make it: collision_threshold - prox < 0
                c[constraint_idx] = collision_threshold - prox
                constraint_idx += 1
            end
        end
    end
    
    @assert constraint_idx - 1 == n_ineq "Expected $n_ineq constraints total, but used $(constraint_idx-1)"
    
    return c
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
    J_eq = ForwardDiff.jacobian(z_ -> equality_constraint(params, z_), Z)

    # Recalculate n_ineq to be sure
    dummy_ineq_cons = inequality_constraint(params, Z) # Evaluate to get size
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

        poses1_world, _, _, _ = forward_kinematics_poe(q1, params.robot_kin[1])
        poses2_world, _, _, _ = forward_kinematics_poe(q2, params.robot_kin[2])
        
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

                J_fk1_q1 = calculate_link_pose_jacobian_geom(q1, link1_idx, params.robot_kin[1])
                J_fk2_q2 = calculate_link_pose_jacobian_geom(q2, link2_idx, params.robot_kin[2])
                
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

# Main function to set up and solve the optimization problem
function solve_trajectory_dircol(robot_kin1, robot_kin2, P_links, q_start1, q_goal1, q_start2, q_goal2)
    # Problem dimensions
    N = 11           # Number of knot points
    dt = 0.1          # Time step
    # tf = (N-1) * dt   # Final time
    
    # Create indexing
    idx = create_idx(NX, NU, N)
    
    # Initial and goal states
    x0 = vcat(q_start1, zeros(NJ*2), q_start2, zeros(NJ*2))
    xg = vcat(q_goal1, zeros(NJ*2), q_goal2, zeros(NJ*2))
    
    # Cost matrices
    Q = Diagonal(zeros(NX))
    Qf_q_weight = 100.0
    Qf_dq_weight = 10.0
    Qf_ddq_weight = 1.0
    Qf_diag_single = vcat(fill(Qf_q_weight, NJ), fill(Qf_dq_weight, NJ), fill(Qf_ddq_weight, NJ))
    Qf = Diagonal(vcat(Qf_diag_single, Qf_diag_single))
    R_jerk_weight = 0.01
    R_time_weight = 0.001  # Small weight for time variable
    R = Diagonal(vcat([R_time_weight], fill(R_jerk_weight, NU-1)))

    time_cost_weight = 2.0
    
    # Constraint limits
    # Joint position limits (same for both robots)
    q_lim_single = (fill(Q_LIMITS[1], NJ), fill(Q_LIMITS[2], NJ))
    dq_lim_single = (fill(DQ_LIMITS[1], NJ), fill(DQ_LIMITS[2], NJ))
    ddq_lim_single = (fill(DDQ_LIMITS[1], NJ), fill(DDQ_LIMITS[2], NJ))
    
    # Control limits (time scaling and jerk)
    u_min = vcat([TIME_SCALING_LIMITS[1]], fill(JERK_LIMITS[1], NU-1))
    u_max = vcat([TIME_SCALING_LIMITS[2]], fill(JERK_LIMITS[2], NU-1))
    
    # Parameters bundle
    params = (
        idx = idx,
        N = N,
        dt = dt,
        x0 = x0,
        xg = xg,
        Q = Q,
        R = R,
        Qf = Qf,
        time_cost_weight = time_cost_weight,
        q_lim = q_lim_all,
        dq_lim = dq_lim_all,
        ddq_lim = ddq_lim_all,
        u_min = u_min,
        u_max = u_max,
        robot_kin = [robot_kin1, robot_kin2],
        P_links = P_links,
        collision_threshold = COLLISION_THRESHOLD
    )

    # Collision constraints
    n_collision_constraints = N * N_LINKS * N_LINKS
    
    # Total inequality constraints
    n_ineq = n_collision_constraints
    
    println("Number of inequality constraints: $n_ineq")
    println("- Collision constraints: $n_collision_constraints")
    
    # Initial guess - linear interpolation
    z0 = zeros(idx.nz)
    for i = 1:N
        x_idx = idx.x[i]
        alpha = (i-1) / (N-1)
        # Linearly interpolate from x0 to xg for joint positions
        q1_i = (1-alpha) * q_start1 + alpha * q_goal1
        q2_i = (1-alpha) * q_start2 + alpha * q_goal2
        
        # Set initial guess for states
        z0[x_idx[1:NJ]] = q1_i
        z0[x_idx[NX_PER_ROBOT+1:NX_PER_ROBOT+NJ]] = q2_i
        
        # Zero velocities and accelerations
        z0[x_idx[NJ+1:NJ*3]] = zeros(2*NJ)
        z0[x_idx[NX_PER_ROBOT+NJ+1:NX]] = zeros(2*NJ)
    end
    
    # Set initial control inputs to zero
    for i = 1:N-1
        u_idx = idx.u[i]
        z0[u_idx] = vcat([1.0], zeros(NU-1))
    end
    
    # Bounds for decision variables
    x_l = -Inf * ones(idx.nz)
    x_u = Inf * ones(idx.nz)
    
    # Apply control bounds
    for i = 1:N-1
        u_idx = idx.u[i]
        x_l[u_idx] = u_min
        x_u[u_idx] = u_max
    end
    
    # Apply state bounds at each knot point
    for i = 1:N
        x_idx = idx.x[i]
        
        # Robot 1 limits
        x_l[x_idx[1:NJ]] = q_lim_single[1]
        x_u[x_idx[1:NJ]] = q_lim_single[2]
        x_l[x_idx[NJ+1:2*NJ]] = dq_lim_single[1]
        x_u[x_idx[NJ+1:2*NJ]] = dq_lim_single[2]
        x_l[x_idx[2*NJ+1:3*NJ]] = ddq_lim_single[1]
        x_u[x_idx[2*NJ+1:3*NJ]] = ddq_lim_single[2]
        
        # Robot 2 limits
        x_l[x_idx[NX_PER_ROBOT+1:NX_PER_ROBOT+NJ]] = q_lim_single[1]
        x_u[x_idx[NX_PER_ROBOT+1:NX_PER_ROBOT+NJ]] = q_lim_single[2]
        x_l[x_idx[NX_PER_ROBOT+NJ+1:NX_PER_ROBOT+2*NJ]] = dq_lim_single[1]
        x_u[x_idx[NX_PER_ROBOT+NJ+1:NX_PER_ROBOT+2*NJ]] = dq_lim_single[2]
        x_l[x_idx[NX_PER_ROBOT+2*NJ+1:NX]] = ddq_lim_single[1]
        x_u[x_idx[NX_PER_ROBOT+2*NJ+1:NX]] = ddq_lim_single[2]
    end
    
    # Fix initial state
    x_l[idx.x[1]] = x0
    x_u[idx.x[1]] = x0

    # Fix goal state
    x_l[idx.x[N]] = xg
    x_u[idx.x[N]] = xg
    
    # Inequality constraint bounds
    c_ineq_l = -Inf * ones(n_ineq)
    c_ineq_u = zeros(n_ineq)  # All constraints are of form g(x) <= 0
    
    # Test the inequality constraint function with our initial guess
    println("Testing inequality constraint function...")
    c_test = inequality_constraint(params, z0)
    println("Length of constraint vector: $(length(c_test))")
    
    println("Starting trajectory optimization using direct collocation...")
    
    # Solve the optimization problem
    diff_type = :finite  # Use ForwardDiff for derivatives
    Z = fmincon(point_cost, equality_constraint, inequality_constraint,
                x_l, x_u, c_ineq_l, c_ineq_u, z0, params, diff_type;
                analytical_constraint_jacobian=analytical_combined_constraints_jacobian,
                tol=1e-2, c_tol=1e-2, max_iters=3000, verbose=true)
    
    println("Optimization complete!")
    
    # Extract optimal trajectories
    X = [Z[idx.x[i]] for i = 1:N]
    U = [Z[idx.u[i]] for i = 1:N-1]

    check_solution_constraints(X, U, x0, xg, 
        q_lim_single, dq_lim_single, ddq_lim_single, u_min, u_max)

    # Extract and print time scaling factors
    time_scaling = [U[i][1] for i = 1:N-1]
    println("Time scaling factors: ", time_scaling)
    println("Average time scaling: ", mean(time_scaling))
    
    # Calculate actual time points based on scaling
    time_points = zeros(N)
    for i = 2:N
        time_points[i] = time_points[i-1] + dt * U[i-1][1]
    end
    println("Total trajectory time: ", time_points[end])
    
    return X, U, time_points
end

function check_solution_constraints(X_sol, U_sol, x0_target, xg_target, 
                                    q_lims_tuple, dq_lims_tuple, ddq_lims_tuple, 
                                    u_min_vec, u_max_vec; tol=1e-6)
    
    N_states = length(X_sol)
    N_controls = length(U_sol)
    
    if N_states == 0
        println("X_sol is empty. Cannot check constraints.")
        return false
    end

    violations_found = false
    println("\n--- Checking Solution Constraints (Tolerance: $tol) ---")

    # Initialize min/max trackers for each joint of each robot
    # Dimensions: [robot_idx, joint_idx]
    min_q_obs = fill(Inf, NUM_ROBOTS, NJ)
    max_q_obs = fill(-Inf, NUM_ROBOTS, NJ)
    min_dq_obs = fill(Inf, NUM_ROBOTS, NJ)
    max_dq_obs = fill(-Inf, NUM_ROBOTS, NJ)
    min_ddq_obs = fill(Inf, NUM_ROBOTS, NJ)
    max_ddq_obs = fill(-Inf, NUM_ROBOTS, NJ)
    min_jerk_obs = fill(Inf, NUM_ROBOTS, NJ)
    max_jerk_obs = fill(-Inf, NUM_ROBOTS, NJ)

    # 1. Initial State Constraint
    if !isapprox(X_sol[1], x0_target; atol=tol)
        println("VIOLATION: Initial state constraint not met.")
        println("  Expected X[1]: $x0_target")
        println("  Got X[1]:      $(X_sol[1])")
        println("  Difference:    $(X_sol[1] - x0_target)")
        violations_found = true
    else
        println("Initial state constraint: SATISFIED")
    end

    # 2. Goal State Constraint
    # Note: The problem setup in solve_trajectory_dircol fixes the entire final state X_sol[N] to xg.
    if !isapprox(X_sol[end], xg_target; atol=tol)
        println("VIOLATION: Goal state constraint not met.")
        println("  Expected X[end]: $xg_target")
        println("  Got X[end]:      $(X_sol[end])")
        println("  Difference:      $(X_sol[end] - xg_target)")
        violations_found = true
    else
        println("Goal state constraint: SATISFIED")
    end
    
    println("\nChecking state and control limits over trajectory:")

    # 3. State Limits (q, dq, ddq)
    for k = 1:N_states
        x_k = X_sol[k]
        
        q_vals = [x_k[1:NJ], x_k[NX_PER_ROBOT .+ (1:NJ)]]
        dq_vals = [x_k[NJ+1 : 2*NJ], x_k[NX_PER_ROBOT .+ (NJ+1 : 2*NJ)]]
        ddq_vals = [x_k[2*NJ+1 : 3*NJ], x_k[NX_PER_ROBOT .+ (2*NJ+1 : 3*NJ)]]

        for r = 1:NUM_ROBOTS
            for j = 1:NJ
                # Position
                q_val = q_vals[r][j]
                min_q_obs[r,j] = min(min_q_obs[r,j], q_val)
                max_q_obs[r,j] = max(max_q_obs[r,j], q_val)
                if q_val < q_lims_tuple[1][j] - tol || q_val > q_lims_tuple[2][j] + tol
                    println("VIOLATION: Robot $r, Joint $j, q limit @ k=$k. Value: $q_val, Limits: [$(q_lims_tuple[1][j]), $(q_lims_tuple[2][j])]")
                    violations_found = true
                end

                # Velocity
                dq_val = dq_vals[r][j]
                min_dq_obs[r,j] = min(min_dq_obs[r,j], dq_val)
                max_dq_obs[r,j] = max(max_dq_obs[r,j], dq_val)
                if dq_val < dq_lims_tuple[1][j] - tol || dq_val > dq_lims_tuple[2][j] + tol
                    println("VIOLATION: Robot $r, Joint $j, dq limit @ k=$k. Value: $dq_val, Limits: [$(dq_lims_tuple[1][j]), $(dq_lims_tuple[2][j])]")
                    violations_found = true
                end

                # Acceleration
                ddq_val = ddq_vals[r][j]
                min_ddq_obs[r,j] = min(min_ddq_obs[r,j], ddq_val)
                max_ddq_obs[r,j] = max(max_ddq_obs[r,j], ddq_val)
                if ddq_val < ddq_lims_tuple[1][j] - tol || ddq_val > ddq_lims_tuple[2][j] + tol
                    println("VIOLATION: Robot $r, Joint $j, ddq limit @ k=$k. Value: $ddq_val, Limits: [$(ddq_lims_tuple[1][j]), $(ddq_lims_tuple[2][j])]")
                    violations_found = true
                end
            end
        end
    end

    # 4. Control (Jerk) Limits
    if N_controls > 0 && N_controls == N_states - 1
        for k = 1:N_controls
            u_k = U_sol[k]
            jerk_vals_r1 = u_k[1:NJ]
            jerk_vals_r2 = u_k[NU_PER_ROBOT .+ (1:NJ)] # Or u_k[NJ+1 : NU]
            
            all_jerk_vals = [jerk_vals_r1, jerk_vals_r2]

            for r = 1:NUM_ROBOTS
                u_offset = (r-1)*NU_PER_ROBOT
                for j = 1:NJ
                    jerk_val = all_jerk_vals[r][j]
                    min_jerk_obs[r,j] = min(min_jerk_obs[r,j], jerk_val)
                    max_jerk_obs[r,j] = max(max_jerk_obs[r,j], jerk_val)
                    
                    # u_min_vec and u_max_vec are for the combined control vector u
                    current_u_idx = u_offset + j
                    if jerk_val < u_min_vec[current_u_idx] - tol || jerk_val > u_max_vec[current_u_idx] + tol
                        println("VIOLATION: Robot $r, Joint $j, Jerk limit @ k=$k. Value: $jerk_val, Limits: [$(u_min_vec[current_u_idx]), $(u_max_vec[current_u_idx])]")
                        violations_found = true
                    end
                end
            end
        end
    elseif N_controls > 0
         println("WARNING: Number of control inputs ($(N_controls)) does not match N_states-1 ($(N_states-1)). Skipping detailed control limit checks.")
    end

    println("\n--- Summary of Observed Min/Max Values ---")
    for r = 1:NUM_ROBOTS
        println("\nRobot $r:")
        for j = 1:NJ
            println("  Joint $j:")
            println("    Position (q): MinObs=$(round(min_q_obs[r,j], digits=4)), MaxObs=$(round(max_q_obs[r,j], digits=4)), Limits=[$(q_lims_tuple[1][j]), $(q_lims_tuple[2][j])]")
            println("    Velocity (dq): MinObs=$(round(min_dq_obs[r,j], digits=4)), MaxObs=$(round(max_dq_obs[r,j], digits=4)), Limits=[$(dq_lims_tuple[1][j]), $(dq_lims_tuple[2][j])]")
            println("    Accel (ddq): MinObs=$(round(min_ddq_obs[r,j], digits=4)), MaxObs=$(round(max_ddq_obs[r,j], digits=4)), Limits=[$(ddq_lims_tuple[1][j]), $(ddq_lims_tuple[2][j])]")
            if N_controls > 0
                u_idx_for_jerk_limit = (r-1)*NU_PER_ROBOT + j
                println("    Jerk (u):     MinObs=$(round(min_jerk_obs[r,j], digits=4)), MaxObs=$(round(max_jerk_obs[r,j], digits=4)), Limits=[$(u_min_vec[u_idx_for_jerk_limit]), $(u_max_vec[u_idx_for_jerk_limit])]")
            end
        end
    end

    if !violations_found
        println("\nAll checked constraints appear to be SATISFIED within tolerance $tol.")
    else
        println("\nSome constraints were VIOLATED (see messages above).")
    end
    
    return !violations_found
end

function main()
    let
        # --- Define Robot Kinematics (Example: Generic 6-DOF) ---
        T = Float64
        # Twists ξᵢ (Defined in base frame {0} at q=0)
        w1 = SA[T(0), T(0), T(1)]; p1 = SA[T(0), T(0), T(0)]
        w2 = SA[T(0), T(1), T(0)]; p2 = SA[T(0), T(0), T(0.33)]
        w3 = SA[T(0), T(-1), T(0)]; p3 = SA[T(0), T(0), T(0.33+0.26)]
        w4 = SA[T(-1), T(0), T(0)]; p4 = SA[T(0.1), T(0), T(0.33+0.26+0.015)]
        w5 = SA[T(0), T(-1), T(0)]; p5 = SA[T(0.1+0.19), T(0), T(0.33+0.26+0.015)]
        w6 = SA[T(-1), T(0), T(0)]; p6 = SA[T(0.1+0.19+0.072), T(0), T(0.33+0.26+0.015)]
        twists_base = [
            vcat(-cross(w1, p1), w1), vcat(-cross(w2, p2), w2), vcat(-cross(w3, p3), w3),
            vcat(-cross(w4, p4), w4), vcat(-cross(w5, p5), w5), vcat(-cross(w6, p6), w6)
        ]

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

        # Hard code the link length values for collision capsule
        link_lengths_geom = [0.33, 0.26, 0.1, 0.19, 0.072, 0.11]
        T_link_centers = Vector{AffineMap{RotMatrix3{T}, SVector{3, T}}}(undef, NJ)
        for i = 1:NJ
            L_i = link_lengths_geom[i]
            T_link_centers[i] = AffineMap(RotMatrix(RotY(π/2)), SA[T(0.0), T(0.0), T(L_i/2.0)])
        end

        # Radii for collision geometry
        vis_link_radii = [0.095, 0.095, 0.06, 0.06, 0.05, 0.04]

        # Base transforms
        T_base1 = AffineMap(RotMatrix(RotZ(0.0)), SA[T(0.0), T(0.0), T(0)])
        T_base2 = AffineMap(RotMatrix(RotZ(π)), SA[T(0.88101), T(-0.01304), T(0)])

        # Create Kinematics structs
        robot_kin1 = RobotKinematics(twists_base, T_base1, M_frames_zero, T_link_centers, vis_link_radii)
        robot_kin2 = RobotKinematics(twists_base, T_base2, M_frames_zero, T_link_centers, vis_link_radii)

        # Create Collision Primitives
        P_links = [[], []]
        primitive_shape = ["capsule", "capsule", "capsule", "cylinder", "capsule", "cylinder"]
        for r = 1:NUM_ROBOTS
            current_robot_kin = (r == 1) ? robot_kin1 : robot_kin2
            for i = 1:N_LINKS
                len = link_lengths_geom[i]
                rad = T(current_robot_kin.link_radius[i])
                if primitive_shape[i] == "capsule"
                    link_prim = dc.CapsuleMRP(rad, len)
                elseif primitive_shape[i] == "cylinder"
                    link_prim = dc.CylinderMRP(rad, len)
                else
                    error("Unknown primitive shape: $primitive_shape[i]")
                end
                link_prim.r_offset = SA[T(0), T(0), T(0)]
                link_prim.Q_offset = SMatrix{3,3,T,9}(I)
                push!(P_links[r], link_prim)
            end
        end

        # Start and Goal States
        # q_start1 = [0.0, 0, 0.0, 0.0, -π/2, 0.0]
        # q_goal1 = [0.0, π/5, -π/5, 0, -π/4, 0.0]
        # q_start2 = [0.0, 0, 0.0, 0.0, -π/2, 0.0]
        # q_goal2 = [0.0, π/5, π/5, 0, π/2, 0.0]
        q_start1 = [-π/4, 0, 0.0, 0.0, 0.0, 0.0]
        q_goal1 = [π/4, 0, 0.0, 0.0, 0.0, 0.0]
        q_start2 = [-π/4, 0, 0.0, 0.0, 0.0, 0.0]
        q_goal2 = [π/4, 0, 0.0, 0.0, 0.0, 0.0]

        # Solve using direct collocation
        X_sol, U_sol, time_points = solve_trajectory_dircol(robot_kin1, robot_kin2, P_links, 
                                             q_start1, q_goal1, q_start2, q_goal2)

        # Print time information
        println("Time scaling factors: ", [U_sol[i][1] for i = 1:length(U_sol)])
        println("Time points: ", time_points)
        println("Total trajectory time: ", time_points[end])

        # Visualization setup
        println("Setting up Visualization...")
        vis = mc.Visualizer()
        mc.open(vis)
        mc.setprop!(vis["/Background"], "top_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
        mc.setprop!(vis["/Background"], "bottom_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
        mc.delete!(vis["/Grid"]); mc.delete!(vis["/Axes"])

        colors = [mc.RGBA(1.0, 0.0, 0.0, 0.8), mc.RGBA(0.0, 0.0, 1.0, 0.8)]
        link_vis_names = [[Symbol("R$(r)_L$(l)") for l=1:N_LINKS] for r=1:NUM_ROBOTS]

        # Build visual primitives
        println("Building visual primitives...")
        for r = 1:NUM_ROBOTS
            for l = 1:N_LINKS
                primitive_obj = P_links[r][l]
                vis_name = link_vis_names[r][l]
                vis_color = colors[r]
                dc.build_primitive!(vis, primitive_obj, vis_name; color = vis_color)
            end
            # Base visualization
            dc.build_primitive!(vis, dc.SphereMRP(0.01), Symbol("Base$r"); color=colors[r])
            base_prim = dc.SphereMRP(0.01)
            if r == 1
                base_prim.r = T_base1.translation
            else
                base_prim.r = T_base2.translation
            end
            dc.update_pose!(vis[Symbol("Base$r")], base_prim)
        end

        # Animate the solution
        N = length(X_sol)
        
        # Calculate display dt based on final time
        final_time = time_points[end]
        display_fps = 60
        display_dt = 1/display_fps
        num_frames = ceil(Int, final_time * display_fps)
        
        anim = mc.Animation(Int(display_fps))
        println("Creating Animation with $num_frames frames...")
        
        # Create interpolation function for states
        function interpolate_state(t)
            # Find the right segment
            i = 1
            while i < N && time_points[i+1] < t
                i += 1
            end
            
            # If at or past the end, return the last state
            if i >= N || t >= time_points[end]
                return X_sol[end]
            end
            
            # Linear interpolation within segment
            t1 = time_points[i]
            t2 = time_points[i+1]
            s = (t - t1) / (t2 - t1)  # Interpolation factor
            
            return (1-s) * X_sol[i] + s * X_sol[i+1]
        end
        
        # Create frames at regular time intervals
        for frame = 1:num_frames
            t = (frame - 1) * display_dt
            mc.atframe(anim, frame) do
                x_t = interpolate_state(t)
                q1_t = x_t[1:NJ]
                q2_t = x_t[NX_PER_ROBOT .+ (1:NJ)]
                all_q_t = [q1_t, q2_t]
                
                for r = 1:NUM_ROBOTS
                    # Calculate FK for current robot
                    robot_kin_r = (r == 1) ? robot_kin1 : robot_kin2
                    poses_world_t, _, _, _ = forward_kinematics_poe(all_q_t[r], robot_kin_r)
                    
                    for l = 1:N_LINKS
                        primitive_obj = P_links[r][l]
                        T_link_center = poses_world_t[l]
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
        end
    end

# Run the main function
main()