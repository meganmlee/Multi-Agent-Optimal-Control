# --- Preamble ---
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
using Rotations        # For rotations
using CoordinateTransformations # For composing transformations

# Assuming simple_altro.jl contains the iLQR implementation from your example
include("simple_altro.jl") # Make sure this file is in the same directory or provide the correct path

# --- Robot and Environment Parameters ---

const NUM_ROBOTS = 2
const NUM_JOINTS_PER_ROBOT = 3
const NJ = NUM_JOINTS_PER_ROBOT # Alias for convenience
const N_LINKS = NUM_JOINTS_PER_ROBOT # Assuming one link per joint for simplicity

# Kinematics
const LINK_LENGTHS = [1.0, 0.8, 0.6] # Length of each link (same for both arms for now)
const BASE_POSITIONS = [SA[0.0, -1.5, 0.0], SA[0.0, 1.5, 0.0]] # Base positions for arm 1 and arm 2
const LINK_RADIUS = 0.1 # Radius for collision capsules/cylinders

# Dynamics & Limits (per joint)
const Q_LIMITS = (-π, π)  # Joint angle limits
const DQ_LIMITS = (-2.0, 2.0) # Joint velocity limits
const DDQ_LIMITS = (-5.0, 5.0) # Joint acceleration limits
const JERK_LIMITS = (-10.0, 10.0) # Joint jerk limits (control limits)

# Collision
const COLLISION_THRESHOLD = 1 # Minimum inflation threshold allowed between links (m) (>1 no collision, <1 has collision)

# --- State and Control Dimensions ---

# State: [q1; dq1; ddq1; q2; dq2; ddq2] where q1, dq1, ddq1 are vectors of size NJ
const NX_PER_ROBOT = 3 * NJ
const NX = NUM_ROBOTS * NX_PER_ROBOT # Total state dimension (pos, vel, accel for all joints)

# Control: [jerk1; jerk2] where jerk1, jerk2 are vectors of size NJ
const NU_PER_ROBOT = NJ
const NU = NUM_ROBOTS * NU_PER_ROBOT # Total control dimension (jerk for all joints)

# --- Helper Functions ---

"Calculates the pose of each link for a single arm given joint angles q."
function forward_kinematics(q::AbstractVector, base_pos::SVector{3,T}, link_lengths::Vector{F}) where {T,F}
    @assert length(q) == length(link_lengths) "Mismatch in joint angles and link lengths"
    num_joints = length(q)
    link_poses = Vector{Tuple{SVector{3,T}, RotMatrix3{T}}}(undef, num_joints) # Store (position, orientation)

    T_world_joint = AffineMap(RotMatrix{3,T}(I), SVector{3,T}(base_pos))

    for i = 1:num_joints
        # 1. Relative rotation for joint i about the local Z-axis
        #    This rotation occurs *at* the current joint's origin.
        R_relative = LinearMap(RotZ(T(q[i]))) # Ensure type T

        # 2. Transform from the current joint frame (at link i's start)
        #    to the frame at the *midpoint* of link i.
        #    This involves rotating by q[i] and then translating half the link length
        #    along the *new* local x-axis.
        T_joint_to_midpoint = R_relative ∘ Translation(T(link_lengths[i]/2), T(0.0), T(0.0))

        # 3. Compose with the transform TO the current joint origin to get the
        #    world pose of the link's midpoint.
        #    T_world_midpoint = T_world_joint( T_joint_to_midpoint( point_in_midpoint_frame ) )
        T_world_midpoint = T_world_joint ∘ T_joint_to_midpoint

        # Extract the world position and orientation of the link center
        link_midpoint_position = T_world_midpoint.translation
        link_orientation = RotMatrix(T_world_midpoint.linear) # Rotation part accumulates correctly

        link_poses[i] = (link_midpoint_position, link_orientation)

        # 4. Calculate the transform from the current joint frame to the
        #    *end* of the current link (which is the origin of the *next* joint).
        T_joint_to_end = R_relative ∘ Translation(T(link_lengths[i]), T(0.0), T(0.0))

        # 5. Update the transform for the origin of the *next* joint (i+1).
        #    This becomes the starting point for the next iteration.
        T_world_joint = T_world_joint ∘ T_joint_to_end
    end
    return link_poses
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
    k1 = p.dt * dynamics(p, x, u, k)
    k2 = p.dt * dynamics(p, x + k1 / 2, u, k)
    k3 = p.dt * dynamics(p, x + k2 / 2, u, k)
    k4 = p.dt * dynamics(p, x + k3, u, k)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

# --- Helper Functions for Analytical Jacobian ---

"""
skew(v)

Computes the skew-symmetric matrix for a 3-vector v.
"""
@inline function skew(v::AbstractVector{T}) where T
    # Ensure it returns SMatrix for performance with StaticArrays input
    return SMatrix{3, 3, T, 9}(
         0,  v[3], -v[2],
       -v[3],    0,  v[1],
        v[2], -v[1],    0
    )
end

"""
B_matrix_mrp(p)

Computes the B(p) matrix relating MRP rates to angular velocity: dp/dt = B(p)*omega.
Using the convention from Schaub & Junkins / Tanygin.
B(p) = 0.25 * ( (1 - p'p)I + 2*skew(p) + 2*p*p' )
"""
@inline function B_matrix_mrp(p::AbstractVector{T}) where T
    p_sq_norm = dot(p, p)
    # Use StaticArrays for efficiency with small matrices
    I3 = SMatrix{3,3,T,9}(1.0I)
    p_outer = p * p'
    p_skew = skew(p)
    # Ensure consistent types if p might be Vector instead of SVector
    B = T(0.25) * ((T(1.0) - p_sq_norm) * I3 + T(2.0) * p_skew + T(2.0) * p_outer)
    return B
end


"""
calculate_link_pose_jacobian(q_r, link_idx, robot_idx, p_global)

Calculates the analytical Jacobian (6xNJ) of a link's pose [r; p]
with respect to the robot's joint angles q_r.
"""
function calculate_link_pose_jacobian(q_r::AbstractVector{T}, link_idx::Int, robot_idx::Int, p_global::NamedTuple) where T
    NJ = length(q_r)
    nx_pose = 6 # Size of [r; p]
    J_pose = zeros(T, nx_pose, NJ) # Initialize Jacobian (will be dense up to link_idx)

    base_pos = p_global.base_positions[robot_idx]
    link_lengths = p_global.link_lengths

    # --- Forward Kinematics Pass (Optimized for Jacobian calculation) ---
    # We need:
    # - World positions of joint origins (axes): joint_axis_positions[j]
    # - World position of target link center: r_i
    # - World orientation (Rotation Matrix) of target link: R_i

    joint_axis_positions = Vector{SVector{3,T}}(undef, NJ + 1) # joint 0 is base
    link_center_pos = zero(SVector{3,T}) # Will hold target link pos
    link_rot = RotMatrix{3,T}(I)         # Will hold target link rot

    current_trans = AffineMap(RotMatrix{3,T}(I), base_pos) # Use AffineMap for transformations
    joint_axis_positions[1] = base_pos
    current_angle = T(0.0)
    world_z_axis = SA[T(0.0), T(0.0), T(1.0)] # World Z axis

    for i = 1:NJ
        # Rotation for this joint
        joint_rot = RotZ(q_r[i])
        # Full transform up to the END of link i (origin of joint i+1)
        link_end_transform = current_trans ∘ LinearMap(joint_rot) ∘ Translation(link_lengths[i], 0, 0)
        joint_axis_positions[i+1] = link_end_transform.translation # Store position of joint i+1 origin

        if i == link_idx
            # Calculate pose for the target link's center
            link_center_transform = current_trans ∘ LinearMap(joint_rot) ∘ Translation(link_lengths[i]/2, 0, 0)
            link_center_pos = link_center_transform.translation
            link_rot = RotMatrix(link_center_transform.linear)
        end
        # Update transform for the next iteration (base of next link is end of current link)
        current_trans = link_end_transform
    end

    r_i = link_center_pos
    R_i = link_rot
    p_i = Rotations.params(Rotations.MRP(R_i)) # MRP parameters

    # --- Calculate Jacobians ---

    # Position Jacobian J_r (Rows 1-3)
    for j = 1:link_idx # Only joints up to link_idx affect its position
        pos_j = joint_axis_positions[j] # World position of joint j's axis
        col_j = cross(world_z_axis, r_i - pos_j)
        J_pose[1:3, j] = col_j
    end
    # Columns j > link_idx remain zero

    # Angular Velocity Jacobian J_w (Implicitly used)
    # J_w columns are world_z_axis for j <= link_idx, zero otherwise

    # Orientation Jacobian J_p = B(p_i) * J_w (Rows 4-6)
    B_pi = B_matrix_mrp(SVector{3,T}(p_i)) # Ensure SVector for B_matrix_mrp efficiency
    B_pi_times_z = B_pi * world_z_axis # Precompute B(p)*k_j since k_j is constant

    for j = 1:link_idx
        J_pose[4:6, j] = B_pi_times_z
    end
    # Columns j > link_idx remain zero

    return J_pose
end


# --- Constraints ---  

# ineq_con_x remains the same as the previous version (with self-collisions)
function ineq_con_x(p::NamedTuple, x::AbstractVector)
    constraints = Float64[] # Use a dynamic array, convert at the end if needed

    # 1. Limits (Pos, Vel, Accel) for each robot (same as before)
    for r = 1:NUM_ROBOTS
        offset = (r - 1) * NX_PER_ROBOT
        q = x[offset .+ (1:NJ)]
        dq = x[offset .+ (NJ + 1 : 2*NJ)]
        ddq = x[offset .+ (2*NJ + 1 : 3*NJ)]
        append!(constraints, q .- p.q_lim[2]); append!(constraints, p.q_lim[1] .- q)
        append!(constraints, dq .- p.dq_lim[2]); append!(constraints, p.dq_lim[1] .- dq)
        append!(constraints, ddq .- p.ddq_lim[2]); append!(constraints, p.ddq_lim[1] .- ddq)
    end

    # Calculate poses needed for collision checks
    q1 = x[1:NJ]
    q2 = x[NX_PER_ROBOT .+ (1:NJ)]
    poses1 = forward_kinematics(q1, p.base_positions[1], p.link_lengths)
    poses2 = forward_kinematics(q2, p.base_positions[2], p.link_lengths)

    # Update collision primitive poses (Crucial!) - Must happen before proximity checks
    all_poses = [poses1, poses2]
    for r = 1:NUM_ROBOTS
        poses_r = all_poses[r]
        for i = 1:N_LINKS
            pos, rot = poses_r[i]
            mrp = Rotations.MRP(rot); mrp_vec = Rotations.params(mrp)
            p.P_links[r][i].r = pos; p.P_links[r][i].p = mrp_vec
        end
    end

    # 2. Inter-Robot Collision Avoidance
    for i = 1:N_LINKS # Link index on arm 1
        for j = 1:N_LINKS # Link index on arm 2
            prox, _ = dc.proximity(p.P_links[1][i], p.P_links[2][j])
            push!(constraints, p.collision_threshold - prox)
        end
    end

    # # 3. Self-Collision Avoidance (within each arm)
    # for r = 1:NUM_ROBOTS
    #     for i = 1:(N_LINKS - 1)
    #         for j = (i + 2):N_LINKS # j > i + 1
    #             prox_self, _ = dc.proximity(p.P_links[r][i], p.P_links[r][j])
    #             push!(constraints, p.collision_threshold - prox_self)
    #         end
    #     end
    # end
    return constraints
end

"Jacobian of inequality constraints w.r.t. state x (Analytical + Strategic FD)."
function ineq_con_x_jac(p::NamedTuple, x::AbstractVector)
    nx = p.nx
    ncx = p.ncx # Total number of constraints from ineq_con_x
    NJ = NUM_JOINTS_PER_ROBOT # Number of joints per robot

    # Initialize Jacobian matrix (consider sparse later if performance critical)
    J_x = zeros(Float64, ncx, nx)
    current_row = 0

    # --- 1. Jacobians for Limits (Pos, Vel, Accel) ---
    I_NJ = sparse(I, NJ, NJ) # Identity matrix for one robot's joints

    for r = 1:NUM_ROBOTS
        offset_x_q = (r - 1) * NX_PER_ROBOT # Start column for q_r
        offset_x_dq = offset_x_q + NJ      # Start column for dq_r
        offset_x_ddq = offset_x_dq + NJ     # Start column for ddq_r

        # Row indices for this robot's limits
        rows_q_upper = current_row .+ (1:NJ)
        rows_q_lower = rows_q_upper .+ NJ
        rows_dq_upper = rows_q_lower .+ NJ
        rows_dq_lower = rows_dq_upper .+ NJ
        rows_ddq_upper = rows_dq_lower .+ NJ
        rows_ddq_lower = rows_ddq_upper .+ NJ

        # Column blocks for this robot's state
        cols_q = offset_x_q .+ (1:NJ)
        cols_dq = offset_x_dq .+ (1:NJ)
        cols_ddq = offset_x_ddq .+ (1:NJ)

        # Fill Jacobian blocks for limits
        J_x[rows_q_upper, cols_q] = I_NJ    # ∂(q - q_max)/∂q = I
        J_x[rows_q_lower, cols_q] = -I_NJ   # ∂(q_min - q)/∂q = -I
        J_x[rows_dq_upper, cols_dq] = I_NJ    # ∂(dq - dq_max)/∂dq = I
        J_x[rows_dq_lower, cols_dq] = -I_NJ   # ∂(dq_min - dq)/∂dq = -I
        J_x[rows_ddq_upper, cols_ddq] = I_NJ    # ∂(ddq - ddq_max)/∂ddq = I
        J_x[rows_ddq_lower, cols_ddq] = -I_NJ   # ∂(ddq_min - ddq)/∂ddq = -I

        current_row = rows_ddq_lower[end] # Update row counter
    end

    # --- Update Primitive Poses (Needed for proximity_gradient) ---
    # This is crucial: proximity_gradient needs the *current* poses set.
    q1 = x[1:NJ]
    q2 = x[NX_PER_ROBOT .+ (1:NJ)]
    poses1 = forward_kinematics(q1, p.base_positions[1], p.link_lengths)
    poses2 = forward_kinematics(q2, p.base_positions[2], p.link_lengths)
    all_poses = [poses1, poses2]
    for r = 1:NUM_ROBOTS
        poses_r = all_poses[r]
        for i = 1:N_LINKS
            pos, rot = poses_r[i]
            mrp = Rotations.MRP(rot); mrp_vec = Rotations.params(mrp)
            p.P_links[r][i].r = pos
            p.P_links[r][i].p = mrp_vec
        end
    end

    # --- 2. Jacobians for Inter-Robot Collisions ---
    q1_indices = 1:NJ
    q2_indices = NX_PER_ROBOT .+ (1:NJ)
    pose1_slice = 1:6 # Indices for [r1; p1] in the combined gradient
    pose2_slice = 7:12 # Indices for [r2; p2] in the combined gradient

    for i = 1:N_LINKS # Link index on arm 1
        for j = 1:N_LINKS # Link index on arm 2
            current_row += 1
            P1 = p.P_links[1][i]
            P2 = p.P_links[2][j]

            # Get proximity value and the combined gradient w.r.t. [r1; p1; r2; p2]
            prox_val, J_prox_combined = dc.proximity_gradient(P1, P2) # Correct return values

            # Extract gradient parts w.r.t pose1 and pose2
            J_prox_P1 = J_prox_combined[pose1_slice] # Gradient w.r.t [r1;p1] (column vector)
            J_prox_P2 = J_prox_combined[pose2_slice] # Gradient w.r.t [r2;p2] (column vector)


            # Get Jacobian of pose parameters w.r.t joint angles q
            # J_fk1 = ∂(pose1)/∂(q1), size (6, NJ)
            # J_fk2 = ∂(pose2)/∂(q2), size (6, NJ)
            # Calculate analytical FK Jacobians
            J_fk1 = calculate_link_pose_jacobian(q1, i, 1, p) # Analytical ∂(pose1)/∂(q1) (6xNJ)
            J_fk2 = calculate_link_pose_jacobian(q2, j, 2, p) # Analytical ∂(pose2)/∂(q2) (6xNJ)

            # Apply chain rule: ∂h/∂x = - ∂(prox)/∂pose * ∂pose/∂q
            # Place result in the correct columns corresponding to q1 and q2
            J_x[current_row, q1_indices] = -J_prox_P1' * J_fk1
            J_x[current_row, q2_indices] = -J_prox_P2' * J_fk2
        end
    end

    # # --- 3. Jacobians for Self-Collisions ---
    # for r = 1:NUM_ROBOTS
    #     q_r_indices = (r - 1) * NX_PER_ROBOT .+ (1:NJ)
    #     q_r = x[q_r_indices] # Extract joint angles for this robot

    #     for i = 1:(N_LINKS - 1)
    #         for j = (i + 2):N_LINKS # Check pairs (i, j) where j > i + 1
    #             current_row += 1
    #             Pi = p.P_links[r][i]
    #             Pj = p.P_links[r][j]

    #             # Get proximity value and the combined gradient w.r.t. [ri; pi; rj; pj]
    #             prox_val_self, J_prox_combined_self = dc.proximity_gradient(Pi, Pj) # Correct return values

    #             # Extract gradient parts w.r.t pose_i and pose_j
    #             J_prox_Pi = J_prox_combined_self[pose1_slice] # Gradient w.r.t [ri;pi]
    #             J_prox_Pj = J_prox_combined_self[pose2_slice] # Gradient w.r.t [rj;pj]

    #             # Calculate analytical FK Jacobians
    #             J_fki = calculate_link_pose_jacobian(q_r, i, r, p) # Analytical ∂(pose_i)/∂(q_r)
    #             J_fkj = calculate_link_pose_jacobian(q_r, j, r, p) # Analytical ∂(pose_j)/∂(q_r)

    #             # Apply chain rule: ∂h/∂x = - (∂(prox)/∂pose_i * ∂pose_i/∂q_r + ∂(prox)/∂pose_j * ∂pose_j/∂q_r)
    #             # Place result in the correct columns corresponding to q_r
    #             J_x[current_row, q_r_indices] = - (J_prox_Pi' * J_fki + J_prox_Pj' * J_fkj)
    #         end
    #     end
    # end

    # --- Sanity Check ---
    if current_row != ncx
        @warn "Row count mismatch in Jacobian calculation! Expected $ncx, got $current_row"
        # This might happen if the manual calculation of ncx differs from the actual number
        # of constraints generated dynamically. Recalculating ncx inside might be safer.
    end


    return J_x
end

"Inequality constraints on control u (jerk limits)."
function ineq_con_u(p::NamedTuple, u::AbstractVector)
    # u - u_max <= 0; -u + u_min <= 0
    return vcat(u .- p.u_max, p.u_min .- u)
end

"Jacobian of inequality constraints w.r.t. control u."
function ineq_con_u_jac(p::NamedTuple, u::AbstractVector)
    # Simple structure for box constraints
    return Array(float([I(p.nu); -I(p.nu)]))
end

# --- Cost Function ---

function stage_cost(p::NamedTuple, x::AbstractVector, u::AbstractVector, k)
    # Penalize control effort (jerk)
    J_control = 0.5 * u' * p.R * u

    # Optionally penalize deviation from a reference trajectory (if provided)
    # dx = x - p.Xref[k]
    # J_state = 0.5 * dx' * p.Q * dx
    J_state = 0.0 # No state tracking cost for now

    return J_state + J_control
end

function term_cost(p::NamedTuple, x::AbstractVector)
    # Penalize deviation from the final goal state (pos, vel, accel)
    dx = x - p.Xref[p.N] # Xref[N] should be the goal state xg
    return 0.5 * dx' * p.Qf * dx
end

function stage_cost_expansion(p::NamedTuple, x::AbstractVector, u::AbstractVector, k)
    # dx = x - p.Xref[k]
    du = u # Assuming Uref is zero

    # Jxx = p.Q
    # Jx = p.Q * dx
    Jxx = spzeros(p.nx, p.nx) # No quadratic state cost in stage cost for now
    Jx = zeros(p.nx)        # No linear state cost in stage cost for now
    Juu = p.R
    Ju = p.R * du

    return Jxx, Jx, Juu, Ju
end

function term_cost_expansion(p::NamedTuple, x::AbstractVector)
    dx = x - p.Xref[p.N]
    Jxx = p.Qf
    Jx = p.Qf * dx
    return Jxx, Jx
end


# --- Main Optimization Setup ---
function main()
    let
        # --- Simulation Parameters ---
        N = 101   # Number of time steps (controls N-1)
        dt = 0.05 # Time step duration (s) -> 20 Hz

        # --- Start and Goal States ---
        # Define start and goal joint angles for each robot
        q_start1 = [0.0, -π/4, π/4]
        q_goal1 = [π/2, π/2, π/4]

        q_start2 = [0.0, π/4, -π/4]
        q_goal2 = [-π/2, -π/2, -π/4]

        # Construct full start and goal states (q, dq=0, ddq=0)
        x0 = vcat(q_start1, zeros(NJ), zeros(NJ), q_start2, zeros(NJ), zeros(NJ))
        xg = vcat(q_goal1, zeros(NJ), zeros(NJ), q_goal2, zeros(NJ), zeros(NJ))
        Xref = [deepcopy(xg) for i = 1:N] # Reference trajectory is just the goal state
        Uref = [zeros(NU) for i = 1:N-1]  # Reference control is zero

        # --- Cost Matrices ---
        # State cost (mostly for terminal state)
        Q = Diagonal(zeros(NX)) # No running state cost initially
        Qf_q_weight = 100.0
        Qf_dq_weight = 10.0
        Qf_ddq_weight = 1.0
        Qf_diag = vcat(
            fill(Qf_q_weight, NJ), fill(Qf_dq_weight, NJ), fill(Qf_ddq_weight, NJ), # Robot 1
            fill(Qf_q_weight, NJ), fill(Qf_dq_weight, NJ), fill(Qf_ddq_weight, NJ)  # Robot 2
        )
        Qf = Diagonal(Qf_diag)

        # Control cost (jerk)
        R_jerk_weight = 0.01 # Make jerk relatively cheap to allow movement
        R = R_jerk_weight * Diagonal(ones(NU))

        # --- Constraint Limits ---
        u_min = fill(JERK_LIMITS[1], NU)
        u_max = fill(JERK_LIMITS[2], NU)

        # State limits used in ineq_con_x
        q_lim = [fill(Q_LIMITS[1], NJ), fill(Q_LIMITS[2], NJ)]
        dq_lim = [fill(DQ_LIMITS[1], NJ), fill(DQ_LIMITS[2], NJ)]
        ddq_lim = [fill(DDQ_LIMITS[1], NJ), fill(DDQ_LIMITS[2], NJ)]

        # --- Create Collision Primitives ---
        # Use capsules (approximated by cylinder in dc for now)
        P_links = [[], []] # P_links[robot_idx][link_idx]
        for r = 1:NUM_ROBOTS
            for i = 1:N_LINKS
                len = Float64(LINK_LENGTHS[i]) # Ensure Float64
                rad = Float64(LINK_RADIUS)   # Ensure Float64
                # Use CapsuleMRP constructor: dc.CapsuleMRP(Radius, Length)
                link_prim = dc.CapsuleMRP(rad, len)
                # Initialize offset to zero (assuming FK gives center pose)
                link_prim.r_offset = SA[0.0, 0.0, 0.0]
                link_prim.Q_offset = SMatrix{3,3,Float64}(I) # No local rotation offset initially
                push!(P_links[r], link_prim)
            end
        end


        # --- Calculate Constraint Dimensions ---
        # Get dimensions by evaluating constraints once
        dummy_params_for_eval = (; q_lim, dq_lim, ddq_lim, collision_threshold = COLLISION_THRESHOLD, P_links, base_positions=BASE_POSITIONS, link_lengths=LINK_LENGTHS, nu=NU, u_min, u_max) # Include all needed keys for evaluation
        ncx = length(ineq_con_x(dummy_params_for_eval, x0))
        ncu = length(ineq_con_u(dummy_params_for_eval, Uref[1]))
        println("Recalculated ncx = $ncx (limits + inter-robot + self-collision)")
        println("Recalculated ncu = $ncu (control limits)")


        # --- Parameters Bundle ---
        params = (
            nx = NX,
            nu = NU,
            ncx = ncx,
            ncu = ncu,
            N = N,
            Q = Q,
            R = R,
            Qf = Qf,
            u_min = u_min,
            u_max = u_max,
            q_lim = q_lim,     # Used within ineq_con_x
            dq_lim = dq_lim,    # Used within ineq_con_x
            ddq_lim = ddq_lim,  # Used within ineq_con_x
            Xref = Xref,
            Uref = Uref,
            dt = dt,
            P_links = P_links, # Collision primitives
            base_positions = BASE_POSITIONS,
            link_lengths = LINK_LENGTHS,
            collision_threshold = COLLISION_THRESHOLD,
        );

        # --- Initial Trajectory Guess ---
        X = [deepcopy(x0) + ((i-1)/(N-1))*(xg - x0) for i = 1:N] # Linear interpolation between x0 and xg
        U = [zeros(NU) for i = 1:N-1] # Start with zero jerk

        # --- iLQR Variables ---
        Xn = deepcopy(X)
        Un = deepcopy(U)
        P = [zeros(NX, NX) for i = 1:N]   # Cost-to-go quadratic term
        p = [zeros(NX) for i = 1:N]      # Cost-to-go linear term
        d = [zeros(NU) for i = 1:N-1]    # Feedforward control
        K = [zeros(NU, NX) for i = 1:N-1] # Feedback gain

        # --- Run iLQR ---
        println("Starting iLQR Optimization...")
        Xhist = iLQR(params, X, U, P, p, K, d, Xn, Un;
                    atol=1e-1, max_iters=3000, verbose=true, ρ=1.0, ϕ=10.0)

        println("iLQR Finished.")
        X_sol = Xhist[end] # Get the final trajectory

        # --- Visualization ---
        println("Setting up Visualization...")
        vis = mc.Visualizer()
        mc.open(vis)
        mc.setprop!(vis["/Background"], "top_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
        mc.setprop!(vis["/Background"], "bottom_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
        mc.delete!(vis["/Grid"])
        mc.delete!(vis["/Axes"])

        # Colors for arms
        colors = [mc.RGBA(1.0, 0.0, 0.0, 0.8), mc.RGBA(0.0, 0.0, 1.0, 0.8)]

        # Create MeshCat objects for each link
        link_vis_names = [[Symbol("R$(r)_L$(l)") for l=1:N_LINKS] for r=1:NUM_ROBOTS]

            # --- Build Visual Primitives using dc.build_primitive! ---
        # This creates the geometry in MeshCat once.
        println("Building visual primitives...")
        for r = 1:NUM_ROBOTS
            for l = 1:N_LINKS
                # Get the DCOL primitive object (CapsuleMRP)
                primitive_obj = params.P_links[r][l]
                # Use the corresponding name and color
                vis_name = link_vis_names[r][l]
                vis_color = colors[r]
                # Call the DCOL helper function
                dc.build_primitive!(vis, primitive_obj, vis_name; color = vis_color)
                # Note: dc.build_primitive! for CapsuleMRP creates sub-objects like :cyl, :spha, :sphb
                # under the main vis_name node. dc.update_pose! will handle transforming the main node.
            end
        end

        # --- Add target visualization (simple sphere - unchanged) ---
        # mc.setobject!(vis[:goal1], mc.Sphere(Point3f(0), 0.05f), mc.MeshPhongMaterial(color=mc.RGBA(0,1,0,0.5)))
        # mc.setobject!(vis[:goal2], mc.Sphere(Point3f(0), 0.05f), mc.MeshPhongMaterial(color=mc.RGBA(0,1,0,0.5)))
        # goal_poses1 = forward_kinematics(xg[1:NJ], params.base_positions[1], params.link_lengths)
        # goal_poses2 = forward_kinematics(xg[NX_PER_ROBOT .+ (1:NJ)], params.base_positions[2], params.link_lengths)
        # # Calculate goal end-effector positions (unchanged)
        # goal_ee_pos1 = (Translation(goal_poses1[end][1]) ∘ LinearMap(goal_poses1[end][2]) ∘ Translation(LINK_LENGTHS[end]/2, 0, 0)).translation
        # goal_ee_pos2 = (Translation(goal_poses2[end][1]) ∘ LinearMap(goal_poses2[end][2]) ∘ Translation(LINK_LENGTHS[end]/2, 0, 0)).translation
        # mc.settransform!(vis[:goal1], Translation(goal_ee_pos1))
        # mc.settransform!(vis[:goal2], Translation(goal_ee_pos2))


        # --- Animate the result using dc.update_pose! ---
        anim = mc.Animation(floor(Int, 1 / dt))
        println("Creating Animation...")
        for k = 1:N
            # debug print the current q1_k and q2_k
            println("Frame $k:")
            println("q1_k: ", X_sol[k][1:NJ])
            println("q2_k: ", X_sol[k][NX_PER_ROBOT .+ (1:NJ)])
            mc.atframe(anim, k) do
                x_k = X_sol[k]
                q1_k = x_k[1:NJ]
                q2_k = x_k[NX_PER_ROBOT .+ (1:NJ)]

                # --- Calculate Forward Kinematics for this time step ---
                poses1_k = forward_kinematics(q1_k, params.base_positions[1], params.link_lengths)
                poses2_k = forward_kinematics(q2_k, params.base_positions[2], params.link_lengths)
                all_poses_k = [poses1_k, poses2_k]

                # --- Update DCOL Primitive Poses and call dc.update_pose! ---
                for r = 1:NUM_ROBOTS
                    for l = 1:N_LINKS
                        # Get the DCOL primitive object
                        primitive_obj = params.P_links[r][l]
                        # Get the calculated pose for this link at this time step
                        pos, rot = all_poses_k[r][l]

                        # --- Update the internal state of the DCOL primitive ---
                        # This is necessary because dc.update_pose! reads from these fields.
                        primitive_obj.r = pos
                        primitive_obj.p = Rotations.params(Rotations.MRP(rot)) # Convert RotMatrix to MRP vector

                        # Get the corresponding MeshCat visualizer node name
                        vis_name = link_vis_names[r][l]

                        # --- Call the DCOL helper to update the MeshCat transform ---
                        dc.update_pose!(vis[vis_name], primitive_obj)
                    end # end link loop
                end # end robot loop
            end # end mc.atframe
        end # end animation loop

        println("Setting Animation...")
        mc.setanimation!(vis, anim)
        println("Done. Check MeshCat visualizer.")
    end
end # end main function

main()