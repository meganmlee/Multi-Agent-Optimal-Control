#!/usr/bin/env julia

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
import MathOptInterface as MOI
import Ipopt 
import FiniteDiff
import ForwardDiff
import Convex as cvx 
import ECOS
using LinearAlgebra
using Plots
using Random
using JLD2
using Test
using Statistics
include(joinpath(@__DIR__,"fmincon.jl"))

function single_point_dynamics(params, x, u)
    # Simple 2D point dynamics with jerk control
    # State: [px, py, vx, vy, ax, ay]
    # Control: [jx, jy] (jerk)
    
    px, py, vx, vy, ax, ay = x
    jx, jy = u
    
    xdot = [
        vx,
        vy,
        ax,
        ay,
        jx,
        jy
    ]
    
    return xdot
end

function combined_dynamics(params, x, u)
    # dynamics for three 2D points, assuming the state is stacked
    # x = [x1;x2;x3] where each xi = [px, py, vx, vy, ax, ay]
    
    # point 1 
    x1 = x[1:6]
    u1 = u[1:2]
    xdot1 = single_point_dynamics(params, x1, u1)
    
    # point 2
    x2 = x[(1:6) .+ 6]
    u2 = u[(1:2) .+ 2]
    xdot2 = single_point_dynamics(params, x2, u2)
    
    # point 3
    x3 = x[(1:6) .+ 12]
    u3 = u[(1:2) .+ 4]
    xdot3 = single_point_dynamics(params, x3, u3)
    
    # return stacked dynamics 
    return [xdot1; xdot2; xdot3]
end

function create_idx(nx, nu, N)
    # This function creates useful indexing tools for Z 
    # x_i = Z[idx.x[i]]
    # u_i = Z[idx.u[i]]
    
    # our Z vector is [x0, u0, x1, u1, …, xN]
    nz = (N-1) * nu + N * nx # length of Z 
    x = [(i - 1) * (nx + nu) .+ (1 : nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ ((nx + 1):(nx + nu)) for i = 1:(N - 1)]
    
    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c = [(i - 1) * (nx) .+ (1 : nx) for i = 1:(N - 1)]
    nc = (N - 1) * nx
    
    return (nx=nx, nu=nu, N=N, nz=nz, nc=nc, x=x, u=u, c=c)
end

function hermite_simpson(params::NamedTuple, x1::Vector, 
            x2::Vector, u, dt::Real)::Vector
    # Hermite simpson implicit integrator residual
    f1 = combined_dynamics(params, x1, u)
    f2 = combined_dynamics(params, x2, u)

    x_HS = 0.5 * (x1 + x2) + (dt / 8) * (f1 - f2)

    f_HS = combined_dynamics(params, x_HS, u)
    
    return x1 + (dt / 6) * (f1 + 4 * f_HS + f2) - x2
end

function point_cost(params::NamedTuple, Z::Vector)::Real
    idx, N, xg = params.idx, params.N, params.xg
    Q, R, Qf = params.Q, params.R, params.Qf
    
    J = 0
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]
       
        J += (0.5 * ((xi - xg)' * Q * (xi - xg))) +
                           (0.5 * (ui' * R * ui))
    end
    
    # terminal cost 
    xn = Z[idx.x[N]]
    J += 0.5 * (xn - xg)' * Qf * (xn - xg)
    
    return J 
end

function point_dynamics_constraints(params::NamedTuple, Z::Vector)::Vector
    idx, N, dt = params.idx, params.N, params.dt
    
    # create dynamics constraints using hermite simpson 
    c = zeros(eltype(Z), idx.nc)
    
    for i = 1:(N-1)
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]] 
        xip1 = Z[idx.x[i+1]]
        
        c[idx.c[i]] = hermite_simpson(params, xi, xip1, ui, dt)
    end
    return c 
end

function point_equality_constraint(params::NamedTuple, Z::Vector)::Vector
    N, idx, xic, xg = params.N, params.idx, params.xic, params.xg 
    
    # Return all equality constraints (dynamics + initial condition)
    c = zeros(eltype(Z), idx.nc + idx.nx)
    
    c[1:idx.nc] = point_dynamics_constraints(params, Z)
    c[idx.nc+1:idx.nc+idx.nx] = Z[idx.x[1]] - xic
    
    return c
end

function inequality_constraint(params::NamedTuple, Z::Vector)::Vector
    N, idx, xg = params.N, params.idx, params.xg
    
    # Create inequality constraints for minimum distance between points
    # and terminal goal constraint
    c = zeros(eltype(Z), 3*N + 1)

    c_idx = 1
    for i = 1:N
        xi = Z[idx.x[i]]
        
        # Extract positions for each point
        p1 = xi[1:2]
        p2 = xi[7:8]
        p3 = xi[13:14]
        
        # Check minimum distance between points
        # c > 0 ensures distance is greater than min_dist
        c[c_idx] = norm(p1 - p2)^2 - params.min_dist^2
        c_idx += 1
        c[c_idx] = norm(p1 - p3)^2 - params.min_dist^2
        c_idx += 1
        c[c_idx] = norm(p2 - p3)^2 - params.min_dist^2
        c_idx += 1
    end

    # Check goal state constraint
    # c < 0 ensures final state is within goal_tol of target
    xN = Z[idx.x[N]]
    c[c_idx] = norm(xg - xN)^2 - params.goal_tol^2
    
    return c 
end

"""
    point_trajectory_optimization

Function for returning collision-free trajectories for 3 points in 2D.

Outputs:
    x1::Vector{Vector}  # state trajectory for point 1 
    x2::Vector{Vector}  # state trajectory for point 2 
    x3::Vector{Vector}  # state trajectory for point 3 
    u1::Vector{Vector}  # control trajectory for point 1 
    u2::Vector{Vector}  # control trajectory for point 2 
    u3::Vector{Vector}  # control trajectory for point 3 
    t_vec::Vector
    params::NamedTuple

The resulting trajectories have dt=0.2, tf = 5.0, N = 26
where all the x's are length 26, and the u's are length 25.

Each trajectory for point k starts at `xkic`, and finishes near 
`xkg`. The distances between each point are greater than min_dist at 
every knot point in the trajectory.
"""
function point_trajectory_optimization(;verbose=true)
    
    # problem size 
    nx = 18  # 6 states per point × 3 points
    nu = 6   # 2 control inputs per point × 3 points
    dt = 0.2
    tf = 5.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    
    # indexing 
    idx = create_idx(nx, nu, N)

    # LQR cost 
    Q = diagm(ones(nx))
    R = 0.1*diagm(ones(nu))
    Qf = 10*diagm(ones(nx))
    
    # initial conditions and goal states
    # Each point has state [px, py, vx, vy, ax, ay] 
    lo = 0.5 
    mid = 2 
    hi = 3.5 
    
    x1ic = [-2, lo, 0, 0, 0, 0]      # ic for point 1 
    x2ic = [-2, mid, 0, 0, 0, 0]     # ic for point 2 
    x3ic = [-2, hi, 0, 0, 0, 0]      # ic for point 3 
    xic = vcat(x1ic, x2ic, x3ic)

    x1g = [2, mid, 0, 0, 0, 0]       # goal for point 1 
    x2g = [2, hi, 0, 0, 0, 0]        # goal for point 2 
    x3g = [2, lo, 0, 0, 0, 0]        # goal for point 3 
    xg = vcat(x1g, x2g, x3g)
    
    # Parameters
    params = (
        Q = Q,
        R = R,
        Qf = Qf,
        x1ic = x1ic,
        x2ic = x2ic,
        x3ic = x3ic,
        xic = xic,
        x1g = x1g,
        x2g = x2g,
        x3g = x3g,
        xg = xg,
        dt = dt,
        N = N,
        idx = idx,
        min_dist = 0.8,  # minimum distance between points
        goal_tol = 0.2   # tolerance for reaching goal
    )
    
    # Solve for the three collision-free trajectories

    # Primal bounds
    x_l = -Inf*ones(idx.nz)
    x_u = Inf*ones(idx.nz)

    # Inequality constraint bounds
    c_l = zeros(N*3 + 1)
    c_u = Inf*ones(N*3 + 1)
    c_l[N*3 + 1] = -Inf
    c_u[N*3 + 1] = 0

    # Initial guess - linear interpolation between initial and goal states
    init_traj = range(xic, xg, length = N)
    z0 = zeros(idx.nz)
    for i = 1:N
        z0[idx.x[i]] = init_traj[i]
    end

    # diff type
    diff_type = :auto 

    # DIRCOL
    Z = fmincon(point_cost, point_equality_constraint,
                inequality_constraint,
                x_l, x_u, c_l, c_u, z0, params, diff_type;
                tol = 1e-6, c_tol = 1e-6, max_iters = 10_000,
                verbose = verbose)
    
    # Extract the trajectories
    x1 = [Z[idx.x[i]][1:6] for i = 1:N]
    x2 = [Z[idx.x[i]][7:12] for i = 1:N]
    x3 = [Z[idx.x[i]][13:18] for i = 1:N]
    u1 = [Z[idx.u[i]][1:2] for i = 1:(N-1)]
    u2 = [Z[idx.u[i]][3:4] for i = 1:(N-1)]
    u3 = [Z[idx.u[i]][5:6] for i = 1:(N-1)]
    
    # Calculate distances between points for plotting
    distances = []
    for i = 1:N
        p1 = x1[i][1:2]
        p2 = x2[i][1:2]
        p3 = x3[i][1:2]
        push!(distances, [norm(p1-p2), norm(p1-p3), norm(p2-p3)])
    end
    
    # Print information about results
    println("Optimization complete!")
    println("Initial positions:")
    println("Point 1: $(x1[1][1:2])")
    println("Point 2: $(x2[1][1:2])")
    println("Point 3: $(x3[1][1:2])")
    println("\nFinal positions:")
    println("Point 1: $(x1[end][1:2])")
    println("Point 2: $(x2[end][1:2])")
    println("Point 3: $(x3[end][1:2])")
    println("\nMinimum distances between points:")
    min_dist12 = minimum([d[1] for d in distances])
    min_dist13 = minimum([d[2] for d in distances])
    min_dist23 = minimum([d[3] for d in distances])
    println("Min dist 1-2: $min_dist12")
    println("Min dist 1-3: $min_dist13")
    println("Min dist 2-3: $min_dist23")
        
    return x1, x2, x3, u1, u2, u3, t_vec, params, distances 
end

function plot_results(X1, X2, X3, t_vec, params, distances)
    # Convert trajectory arrays to matrices for easier plotting
    X1m = hcat(X1...)
    X2m = hcat(X2...)
    X3m = hcat(X3...)
    
    # Create separate plots and save them individually
    
    # Plot distances
    p1 = plot(t_vec, params.min_dist*ones(params.N), ls = :dash, color = :red, 
        label = "minimum distance", xlabel = "time (s)", ylabel = "distance (m)", 
        title = "Distance between Points")
    plot!(p1, t_vec, hcat(distances...)', label = ["|p₁ - p₂|" "|p₁ - p₃|" "|p₂ - p₃|"])
    savefig(p1, "output/distances.png")
    println("Saved distances plot to 'distances.png'")
    
    # Plot positions
    p2 = plot(X1m[1,:], X1m[2,:], color = :red, title = "Point Trajectories", 
        label = "point 1", xlabel = "x position", ylabel = "y position")
    plot!(p2, X2m[1,:], X2m[2,:], color = :green, label = "point 2")
    plot!(p2, X3m[1,:], X3m[2,:], color = :blue, label = "point 3")
    
    # Mark initial and goal positions
    scatter!(p2, [X1m[1,1]], [X1m[2,1]], color = :red, marker = :circle, label = "start")
    scatter!(p2, [X2m[1,1]], [X2m[2,1]], color = :green, marker = :circle, label = "")
    scatter!(p2, [X3m[1,1]], [X3m[2,1]], color = :blue, marker = :circle, label = "")
    scatter!(p2, [X1m[1,end]], [X1m[2,end]], color = :red, marker = :star, label = "goal")
    scatter!(p2, [X2m[1,end]], [X2m[2,end]], color = :green, marker = :star, label = "")
    scatter!(p2, [X3m[1,end]], [X3m[2,end]], color = :blue, marker = :star, label = "")
    savefig(p2, "output/trajectories.png")
    println("Saved trajectory plot to 'trajectories.png'")
    
    # Plot velocities
    pv = plot(t_vec, X1m[3,:], color = :red, title = "X Velocities", 
        label = "point 1", xlabel = "time (s)", ylabel = "vx")
    plot!(pv, t_vec, X2m[3,:], color = :green, label = "point 2")
    plot!(pv, t_vec, X3m[3,:], color = :blue, label = "point 3")
    
    pvy = plot(t_vec, X1m[4,:], color = :red, title = "Y Velocities", 
        label = "point 1", xlabel = "time (s)", ylabel = "vy")
    plot!(pvy, t_vec, X2m[4,:], color = :green, label = "point 2")
    plot!(pvy, t_vec, X3m[4,:], color = :blue, label = "point 3")
    pvel = plot(pv, pvy, layout=(1,2), size=(900, 400))
    savefig(pvel, "output/velocities.png")
    println("Saved velocity plots to 'velocities.png'")
    
    # Plot accelerations
    pa = plot(t_vec, X1m[5,:], color = :red, title = "X Accelerations", 
        label = "point 1", xlabel = "time (s)", ylabel = "ax")
    plot!(pa, t_vec, X2m[5,:], color = :green, label = "point 2")
    plot!(pa, t_vec, X3m[5,:], color = :blue, label = "point 3")
    
    pay = plot(t_vec, X1m[6,:], color = :red, title = "Y Accelerations", 
        label = "point 1", xlabel = "time (s)", ylabel = "ay")
    plot!(pay, t_vec, X2m[6,:], color = :green, label = "point 2")
    plot!(pay, t_vec, X3m[6,:], color = :blue, label = "point 3")
    pacc = plot(pa, pay, layout=(1,2), size=(900, 400))
    savefig(pacc, "output/accelerations.png")
    println("Saved acceleration plots to 'accelerations.png'")
    
    # For display in the Julia environment, return just the trajectory plot
    return p2
end

# Function to create animation of the trajectories
function animate_trajectories(X1, X2, X3, t_vec, params, distances)
    # Convert trajectory arrays to matrices for easier plotting
    X1m = hcat(X1...)
    X2m = hcat(X2...)
    X3m = hcat(X3...)
    
    # Determine axis limits for consistent view
    x_min = min(minimum(X1m[1,:]), minimum(X2m[1,:]), minimum(X3m[1,:]))
    x_max = max(maximum(X1m[1,:]), maximum(X2m[1,:]), maximum(X3m[1,:]))
    y_min = min(minimum(X1m[2,:]), minimum(X2m[2,:]), minimum(X3m[2,:]))
    y_max = max(maximum(X1m[2,:]), maximum(X2m[2,:]), maximum(X3m[2,:]))
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    # Create animation
    println("Creating animation...")
    anim = @animate for i in 1:params.N
        # Plot full trajectories as faded lines
        plot(X1m[1,1:i], X1m[2,1:i], color = :red, alpha = 0.5,
             label = "", linewidth = 2)
        plot!(X2m[1,1:i], X2m[2,1:i], color = :green, alpha = 0.5,
              label = "", linewidth = 2)
        plot!(X3m[1,1:i], X3m[2,1:i], color = :blue, alpha = 0.5,
              label = "", linewidth = 2)
        
        # Plot current positions as markers
        scatter!([X1m[1,i]], [X1m[2,i]], color = :red, markersize = 8, label = "Point 1")
        scatter!([X2m[1,i]], [X2m[2,i]], color = :green, markersize = 8, label = "Point 2")
        scatter!([X3m[1,i]], [X3m[2,i]], color = :blue, markersize = 8, label = "Point 3")
        
        # Mark goal positions with stars
        if i == 1
            scatter!([X1m[1,end]], [X1m[2,end]], color = :red, marker = :star, 
                     markersize = 8, label = "Goal 1")
            scatter!([X2m[1,end]], [X2m[2,end]], color = :green, marker = :star, 
                     markersize = 8, label = "Goal 2")
            scatter!([X3m[1,end]], [X3m[2,end]], color = :blue, marker = :star, 
                     markersize = 8, label = "Goal 3")
        end
        
        # Add time indicator
        title!("Time: $(round(t_vec[i], digits=2)) s")
        xlabel!("X Position")
        ylabel!("Y Position")
        
        # Set consistent axis limits
        xlims!(x_min, x_max)
        ylims!(y_min, y_max)
    end
    
    # Save animation as gif
    gif(anim, "output/points_animation.gif", fps = 10)
    println("Animation saved to 'points_animation.gif'")
end

# Function to create animation showing before/after optimization
function generate_before_after_animation(params)
    # Create the "before" trajectory - naive straight lines
    N = params.N
    xic, xg = params.xic, params.xg
    
    # Linear interpolation between start and goal
    before_traj = [range(xic[i], xg[i], length=N) for i in 1:length(xic)]
    
    # Extract individual point trajectories
    X1_before = [[before_traj[j][i] for j in 1:6] for i in 1:N]
    X2_before = [[before_traj[j][i] for j in 6+1:12] for i in 1:N]
    X3_before = [[before_traj[j][i] for j in 12+1:18] for i in 1:N]
    
    # Calculate distances for before case
    distances_before = []
    for i = 1:N
        p1 = X1_before[i][1:2]
        p2 = X2_before[i][1:2]
        p3 = X3_before[i][1:2]
        push!(distances_before, [norm(p1-p2), norm(p1-p3), norm(p2-p3)])
    end
    
    # Create animation for the before case
    println("Creating 'before optimization' animation...")
    X1m_before = hcat(X1_before...)
    X2m_before = hcat(X2_before...)
    X3m_before = hcat(X3_before...)
    
    # Determine axis limits for consistent view across both animations
    x_min = min(minimum(X1m_before[1,:]), minimum(X2m_before[1,:]), minimum(X3m_before[1,:]))
    x_max = max(maximum(X1m_before[1,:]), maximum(X2m_before[1,:]), maximum(X3m_before[1,:]))
    y_min = min(minimum(X1m_before[2,:]), minimum(X2m_before[2,:]), minimum(X3m_before[2,:]))
    y_max = max(maximum(X1m_before[2,:]), maximum(X2m_before[2,:]), maximum(X3m_before[2,:]))
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    anim_before = @animate for i in 1:N
        plot(X1m_before[1,1:i], X1m_before[2,1:i], color = :red, alpha = 0.5,
             label = "", linewidth = 2, title = "Before Optimization - Time: $(round(t_vec[i], digits=2)) s")
        plot!(X2m_before[1,1:i], X2m_before[2,1:i], color = :green, alpha = 0.5,
              label = "", linewidth = 2)
        plot!(X3m_before[1,1:i], X3m_before[2,1:i], color = :blue, alpha = 0.5,
              label = "", linewidth = 2)
        
        scatter!([X1m_before[1,i]], [X1m_before[2,i]], color = :red, markersize = 8, label = "Point 1")
        scatter!([X2m_before[1,i]], [X2m_before[2,i]], color = :green, markersize = 8, label = "Point 2")
        scatter!([X3m_before[1,i]], [X3m_before[2,i]], color = :blue, markersize = 8, label = "Point 3")
        
        # Check for collisions in this frame
        collisions = []
        if distances_before[i][1] < params.min_dist
            push!(collisions, "1-2")
        end
        if distances_before[i][2] < params.min_dist
            push!(collisions, "1-3")
        end
        if distances_before[i][3] < params.min_dist
            push!(collisions, "2-3")
        end
        
        if !isempty(collisions)
            annotate!(x_min + 0.1*x_range, y_max - 0.1*y_range, 
                     "COLLISION: " * join(collisions, ", "), 
                     color = :red, fontsize = 10)
        end
        
        xlabel!("X Position")
        ylabel!("Y Position")
        xlims!(x_min, x_max)
        ylims!(y_min, y_max)
    end
    
    # Save animation
    gif(anim_before, "output/before_optimization.gif", fps = 10)
    println("'Before optimization' animation saved to 'before_optimization.gif'")
    
    return X1_before, X2_before, X3_before, distances_before
end

# Run the optimization
X1, X2, X3, U1, U2, U3, t_vec, params, distances = point_trajectory_optimization(verbose=true)

# Generate the "before" optimization trajectories and animation
X1_before, X2_before, X3_before, distances_before = generate_before_after_animation(params)

# Animate the optimized trajectories
animate_trajectories(X1, X2, X3, t_vec, params, distances)

# Plot static results for comparison
p = plot_results(X1, X2, X3, t_vec, params, distances)
display(p)   # Make sure to display the plot