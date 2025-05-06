#!/usr/bin/env julia

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.add(["LinearAlgebra", "StaticArrays", "ForwardDiff", "FiniteDiff", "Printf", "SparseArrays", 
         "MeshCat", "Random", "Colors", "Rotations", "CoordinateTransformations", 
         "Ipopt", "MathOptInterface"])

import DifferentiableCollisions as dc
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
Pkg.add(url="https://github.com/kevin-tracy/lazy_nlp_qd.jl.git")

import lazy_nlp_qd
using SparseArrays

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
    # Robot 1
    dq1 = x[NJ+1:2*NJ]
    ddq1 = x[2*NJ+1:3*NJ]
    
    # Robot 2
    dq2 = x[NX_PER_ROBOT+NJ+1:NX_PER_ROBOT+2*NJ]
    ddq2 = x[NX_PER_ROBOT+2*NJ+1:NX_PER_ROBOT+3*NJ]
    
    # Extract control inputs (jerks)
    jerk1 = u[1:NJ]
    jerk2 = u[NU_PER_ROBOT+1:NU]
    
    # Triple integrator dynamics
    xdot = vcat(
        dq1, ddq1, jerk1, # Robot 1
        dq2, ddq2, jerk2  # Robot 2
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
function sparse_point_cost(params, Z)
    idx, N, xg = params.idx, params.N, params.xg
    Q, R, Qf = params.Q, params.R, params.Qf
    
    J = 0.0
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]
       
        J += 0.5 * ((xi - xg)' * Q * (xi - xg)) + 0.5 * (ui' * R * ui)
    end
    
    # terminal cost 
    xn = Z[idx.x[N]]
    J += 0.5 * (xn - xg)' * Qf * (xn - xg)
    
    return J 
end

# Cost gradient function (required for sparse interface)
function sparse_cost_gradient!(params, grad, Z)
    idx, N, xg = params.idx, params.N, params.xg
    Q, R, Qf = params.Q, params.R, params.Qf
    
    # Zero out the gradient
    fill!(grad, 0.0)
    
    # Gradient for each state and control
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]
        
        # Add state contribution to gradient
        grad[idx.x[i]] .+= Q * (xi - xg)
        
        # Add control contribution to gradient
        grad[idx.u[i]] .+= R * ui
    end
    
    # Terminal state contribution
    xn = Z[idx.x[N]]
    grad[idx.x[N]] .+= Qf * (xn - xg)
    
    return nothing
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

# Combined constraint function for all constraints
function sparse_constraint!(params, cval, Z)
    N, idx = params.N, params.idx
    robot_kin1, robot_kin2 = params.robot_kin[1], params.robot_kin[2]
    P_links = params.P_links
    
    constraint_idx = 1
    
    # 1. Dynamics constraints (Hermite-Simpson)
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]] 
        xip1 = Z[idx.x[i+1]]
        
        # Calculate Hermite-Simpson constraint
        c_dyn_i = hermite_simpson(params, xi, xip1, ui, params.dt)
        
        # Store in constraint vector
        cval[constraint_idx:constraint_idx+idx.nx-1] = c_dyn_i
        constraint_idx += idx.nx
    end
    
    # 2. Initial state constraint
    cval[constraint_idx:constraint_idx+idx.nx-1] = Z[idx.x[1]] - params.x0
    constraint_idx += idx.nx
    
    # 3. Final state constraints (position only)
    # Robot 1 final position
    cval[constraint_idx:constraint_idx+NJ-1] = Z[idx.x[N]][1:NJ] - params.xg[1:NJ]
    constraint_idx += NJ
    
    # Robot 2 final position
    cval[constraint_idx:constraint_idx+NJ-1] = Z[idx.x[N]][NX_PER_ROBOT+1:NX_PER_ROBOT+NJ] - params.xg[NX_PER_ROBOT+1:NX_PER_ROBOT+NJ]
    constraint_idx += NJ
    
    # 4. State limit constraints
    for i = 1:N
        x_i = Z[idx.x[i]]
        
        for r = 1:NUM_ROBOTS
            offset = (r - 1) * NX_PER_ROBOT
            
            # Position limits
            q = x_i[offset .+ (1:NJ)]
            for j = 1:NJ
                # Upper bound: q[j] - q_max <= 0
                cval[constraint_idx] = q[j] - params.q_lim[r][2][j]
                constraint_idx += 1
                
                # Lower bound: q_min - q[j] <= 0
                cval[constraint_idx] = params.q_lim[r][1][j] - q[j]
                constraint_idx += 1
            end
            
            # Velocity limits
            dq = x_i[offset .+ (NJ+1:2*NJ)]
            for j = 1:NJ
                cval[constraint_idx] = dq[j] - params.dq_lim[r][2][j]
                constraint_idx += 1
                cval[constraint_idx] = params.dq_lim[r][1][j] - dq[j]
                constraint_idx += 1
            end
            
            # Acceleration limits
            ddq = x_i[offset .+ (2*NJ+1:3*NJ)]
            for j = 1:NJ
                cval[constraint_idx] = ddq[j] - params.ddq_lim[r][2][j]
                constraint_idx += 1
                cval[constraint_idx] = params.ddq_lim[r][1][j] - ddq[j]
                constraint_idx += 1
            end
        end
    end
    
    # 5. Collision constraints
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
                
                # Constraint: collision_threshold - prox <= 0
                cval[constraint_idx] = params.collision_threshold - prox
                constraint_idx += 1
            end
        end
    end
    
    return nothing
end

# Constraint Jacobian function that provides the sparsity pattern
function sparse_constraint_jacobian!(params, jac_values, Z)
    # In the lazy_nlp_qd interface, we only need to assign the non-zero values
    # since the sparsity pattern is already provided
    
    # For demonstration, we'll implement a simple finite difference approximation
    # In practice, you would calculate this analytically for better performance
    
    # Get current constraint values
    c_current = zeros(size(jac_values, 1))
    sparse_constraint!(params, c_current, Z)
    
    # Finite difference step size
    h = 1e-8
    
    # For each entry in the sparse Jacobian
    I, J, _ = findnz(jac_values)
    for idx = 1:length(I)
        i, j = I[idx], J[idx]
        
        # Perturb Z[j]
        Z_perturbed = copy(Z)
        Z_perturbed[j] += h
        
        # Calculate perturbed constraint
        c_perturbed = zeros(size(jac_values, 1))
        sparse_constraint!(params, c_perturbed, Z_perturbed)
        
        # Finite difference approximation
        jac_values[i, j] = (c_perturbed[i] - c_current[i]) / h
    end
    
    return nothing
end

# Function to create the Jacobian sparsity pattern
function create_constraint_jacobian_pattern(params)
    N, idx = params.N, params.idx
    nz = idx.nz
    
    # Calculate the number of constraints
    n_dyn = (N-1) * idx.nx         # Dynamics constraints
    n_init = idx.nx                # Initial state
    n_final = 2 * NJ               # Final position constraints
    n_eq = n_dyn + n_init + n_final # Total equality constraints
    
    n_state_limits = NUM_ROBOTS * N * 3 * NJ * 2  # State limits
    n_collision = N * N_LINKS * N_LINKS           # Collision constraints
    n_ineq = n_state_limits + n_collision         # Total inequality constraints
    
    n_total = n_eq + n_ineq                       # Total constraints
    
    # Initialize I, J vectors for sparse triplet format
    I = Int[]
    J = Int[]
    V = Float64[]
    
    # 1. Dynamics constraints
    con_idx = 0
    for i = 1:(N-1)
        for d = 1:idx.nx  # Each state dimension
            # Current state dependency
            for s = 1:idx.nx
                push!(I, con_idx + d)
                push!(J, idx.x[i][s])
                push!(V, 1.0)  # Placeholder value
            end
            
            # Current control dependency
            for u = 1:idx.nu
                push!(I, con_idx + d)
                push!(J, idx.u[i][u])
                push!(V, 1.0)  # Placeholder value
            end
            
            # Next state dependency
            for s = 1:idx.nx
                push!(I, con_idx + d)
                push!(J, idx.x[i+1][s])
                push!(V, 1.0)  # Placeholder value
            end
        end
        con_idx += idx.nx
    end
    
    # 2. Initial state constraint
    for d = 1:idx.nx
        push!(I, con_idx + d)
        push!(J, idx.x[1][d])
        push!(V, 1.0)
    end
    con_idx += idx.nx
    
    # 3. Final position constraints
    for j = 1:NJ
        # Robot 1 final position
        push!(I, con_idx + j)
        push!(J, idx.x[N][j])
        push!(V, 1.0)
    end
    con_idx += NJ
    
    for j = 1:NJ
        # Robot 2 final position
        push!(I, con_idx + j)
        push!(J, idx.x[N][NX_PER_ROBOT + j])
        push!(V, 1.0)
    end
    con_idx += NJ
    
    # 4. State limits
    for i = 1:N
        for r = 1:NUM_ROBOTS
            offset = (r - 1) * NX_PER_ROBOT
            
            # Position limits
            for j = 1:NJ
                # Upper bound
                push!(I, con_idx + 1)
                push!(J, idx.x[i][offset + j])
                push!(V, 1.0)
                con_idx += 1
                
                # Lower bound
                push!(I, con_idx + 1)
                push!(J, idx.x[i][offset + j])
                push!(V, 1.0)
                con_idx += 1
            end
            
            # Velocity limits
            for j = 1:NJ
                push!(I, con_idx + 1)
                push!(J, idx.x[i][offset + NJ + j])
                push!(V, 1.0)
                con_idx += 1
                
                push!(I, con_idx + 1)
                push!(J, idx.x[i][offset + NJ + j])
                push!(V, 1.0)
                con_idx += 1
            end
            
            # Acceleration limits
            for j = 1:NJ
                push!(I, con_idx + 1)
                push!(J, idx.x[i][offset + 2*NJ + j])
                push!(V, 1.0)
                con_idx += 1
                
                push!(I, con_idx + 1)
                push!(J, idx.x[i][offset + 2*NJ + j])
                push!(V, 1.0)
                con_idx += 1
            end
        end
    end
    
    # 5. Collision constraints
    for i = 1:N
        for link1 = 1:N_LINKS
            for link2 = 1:N_LINKS
                # Dependencies on robot 1 joint positions
                for j = 1:NJ
                    push!(I, con_idx + 1)
                    push!(J, idx.x[i][j])
                    push!(V, 1.0)
                end
                
                # Dependencies on robot 2 joint positions
                for j = 1:NJ
                    push!(I, con_idx + 1)
                    push!(J, idx.x[i][NX_PER_ROBOT + j])
                    push!(V, 1.0)
                end
                
                con_idx += 1
            end
        end
    end
    
    # Create the sparse matrix with the pattern
    return sparse(I, J, V, n_total, nz)
end

# Modified solve function to use lazy_nlp_qd.sparse_fmincon
function solve_trajectory_dircol_sparse(robot_kin1, robot_kin2, P_links, q_start1, q_goal1, q_start2, q_goal2)
    # Problem dimensions
    N = 51            # Number of knot points
    dt = 0.1          # Time step
    tf = (N-1) * dt   # Final time
    
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
    R = R_jerk_weight * Diagonal(ones(NU))
    
    # Constraint limits
    q_lim_single = (fill(Q_LIMITS[1], NJ), fill(Q_LIMITS[2], NJ))
    dq_lim_single = (fill(DQ_LIMITS[1], NJ), fill(DQ_LIMITS[2], NJ))
    ddq_lim_single = (fill(DDQ_LIMITS[1], NJ), fill(DDQ_LIMITS[2], NJ))
    q_lim_all = [q_lim_single, q_lim_single]
    dq_lim_all = [dq_lim_single, dq_lim_single]
    ddq_lim_all = [ddq_lim_single, ddq_lim_single]
    
    # Control limits (jerk)
    u_min = fill(JERK_LIMITS[1], NU)
    u_max = fill(JERK_LIMITS[2], NU)
    
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
        q_lim = q_lim_all,
        dq_lim = dq_lim_all,
        ddq_lim = ddq_lim_all,
        u_min = u_min,
        u_max = u_max,
        robot_kin = [robot_kin1, robot_kin2],
        P_links = P_links,
        collision_threshold = COLLISION_THRESHOLD
    )
    
    # Initial guess - linear interpolation (same as before)
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
        z0[u_idx] = zeros(NU)
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
    
    # Create Jacobian sparsity pattern
    jac_pattern = create_constraint_jacobian_pattern(params)
    
    # Calculate number of constraints
    n_dyn = (N-1) * idx.nx  # Dynamics
    n_init = idx.nx         # Initial state
    n_final = 2 * NJ        # Final position
    n_eq = n_dyn + n_init + n_final
    
    n_state_limits = NUM_ROBOTS * N * 3 * NJ * 2
    n_collision = N * N_LINKS * N_LINKS
    n_ineq = n_state_limits + n_collision
    
    n_total = n_eq + n_ineq
    
    # Constraint bounds
    c_l = zeros(n_total)
    c_u = zeros(n_total)
    
    # Equality constraints (all zeros)
    c_l[1:n_eq] .= 0.0
    c_u[1:n_eq] .= 0.0
    
    # Inequality constraints for state limits
    c_l[n_eq+1:n_eq+n_state_limits] .= -Inf  # g(x) <= 0 form
    c_u[n_eq+1:n_eq+n_state_limits] .= 0.0
    
    # Inequality constraints for collision avoidance
    c_l[n_eq+n_state_limits+1:n_total] .= -Inf  # collision_threshold - prox <= 0
    c_u[n_eq+n_state_limits+1:n_total] .= 0.0
    
    println("Starting trajectory optimization using sparse direct collocation...")
    println("Number of constraints: $n_total (Equality: $n_eq, Inequality: $n_ineq)")
    println("Number of non-zeros in Jacobian: $(nnz(jac_pattern))")
    
    # Solve using lazy_nlp_qd.sparse_fmincon
    Z = lazy_nlp_qd.sparse_fmincon(
        sparse_point_cost,
        sparse_cost_gradient!,
        sparse_constraint!,
        sparse_constraint_jacobian!,
        jac_pattern,
        x_l,
        x_u,
        c_l,
        c_u,
        z0,
        params;
        tol = 1e-2,
        c_tol = 1e-2,
        max_iters = 3000,
        print_level = 5
    )
    
    println("Optimization complete!")
    
    # Extract optimal trajectories
    X = [Z[idx.x[i]] for i = 1:N]
    U = [Z[idx.u[i]] for i = 1:N-1]
    
    return X, U
end

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
        q_start1 = [0.0, 0, 0.0, 0.0, -π/2, 0.0]
        q_goal1 = [0.0, π/5, -π/5, 0, -π/4, 0.0]
        q_start2 = [0.0, 0, 0.0, 0.0, -π/2, 0.0]
        q_goal2 = [0.0, π/5, π/5, 0, π/2, 0.0]

        # Solve using direct collocation
        X_sol, U_sol = solve_trajectory_dircol(robot_kin1, robot_kin2, P_links, 
                                             q_start1, q_goal1, q_start2, q_goal2)

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
        dt = 0.1
        N = length(X_sol)
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
                    robot_kin_r = (r == 1) ? robot_kin1 : robot_kin2
                    poses_world_k, _, _, _ = forward_kinematics_poe(all_q_k[r], robot_kin_r)

                    for l = 1:N_LINKS
                        primitive_obj = P_links[r][l]
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
    end
end

# Run the main function
main()