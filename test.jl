# Required Packages
# If not installed, run:
# using Pkg
# Pkg.add("DataStructures")
# Pkg.add("Plots")
# Pkg.add("Measures") # For plot margins

using DataStructures # For PriorityQueue
using Plots
using Measures       # For plot margins
using Random         # For random obstacle/start/goal generation
using TickTock # Simple timing


# --- Parameters ---

MAP_HEIGHT = 10
MAP_WIDTH = 10
NUM_ROBOTS = 3       # Keep this low (e.g., 2-4) for joint A* performance
OBSTACLE_DENSITY = 0.15 # Fraction of non-start/goal cells that are obstacles

# --- Environment Definition ---

struct Environment
    dims::Tuple{Int, Int}         # (height, width)
    obstacles::Set{Tuple{Int, Int}} # Set of (row, col) obstacle locations
    starts::Vector{Tuple{Int, Int}} # Start locations for each robot
    goals::Vector{Tuple{Int, Int}}  # Goal locations for each robot
end

# --- Helper Functions ---

# Check if a position is within bounds and not an obstacle
function is_valid(pos::Tuple{Int, Int}, env::Environment)
    r, c = pos
    height, width = env.dims
    return 1 <= r <= height && 1 <= c <= width && pos ∉ env.obstacles
end

# Manhattan distance heuristic for a single robot
function manhattan_distance(p1::Tuple{Int, Int}, p2::Tuple{Int, Int})
    return abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
end

# Heuristic for the joint state: sum of Manhattan distances for each robot
function heuristic(state::Vector{Tuple{Int, Int}}, goals::Vector{Tuple{Int, Int}})
    h = 0
    for i in 1:length(state)
        h += manhattan_distance(state[i], goals[i])
    end
    return h
end

# --- Joint-Space A* Implementation ---

# State representation: Vector{Tuple{Int, Int}} - position of each robot
# Node in the search: (state, g_cost, h_cost, parent_state) - simplified here

function get_neighbors(current_state::Vector{Tuple{Int, Int}}, env::Environment)
    num_robots = length(current_state)
    neighbor_states = Vector{Vector{Tuple{Int, Int}}}()

    # Possible moves for each robot: Wait, Up, Down, Left, Right
    moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    move_indices = 1:length(moves)

    # Generate all combinations of moves for all robots
    # This uses an iterator over the Cartesian product of move indices
    for joint_move_indices in Iterators.product(ntuple(_ -> move_indices, num_robots)...)
        next_state = Vector{Tuple{Int, Int}}(undef, num_robots)
        valid_joint_move = true

        # 1. Calculate next positions and check basic validity (bounds, obstacles)
        for i in 1:num_robots
            move = moves[joint_move_indices[i]]
            current_pos = current_state[i]
            next_pos = (current_pos[1] + move[1], current_pos[2] + move[2])

            if !is_valid(next_pos, env)
                valid_joint_move = false
                break # This robot's move is invalid, so the joint move is invalid
            end
            next_state[i] = next_pos
        end

        !valid_joint_move && continue # Skip to the next joint move combination

        # 2. Check for collisions (vertex and edge)
        collided = false
        for i in 1:num_robots
            for j in (i + 1):num_robots
                # Vertex collision: two robots in the same cell at the next timestep
                if next_state[i] == next_state[j]
                    collided = true
                    break
                end
                # Edge collision (swap): robot i moves to j's current pos, and j moves to i's current pos
                if next_state[i] == current_state[j] && next_state[j] == current_state[i]
                    collided = true
                    break
                end
            end
            collided && break
        end

        if !collided
            push!(neighbor_states, next_state)
        end
    end

    return neighbor_states
end

function joint_a_star(env::Environment)
    start_state = env.starts
    goal_state = env.goals
    num_robots = length(start_state)

    if start_state == goal_state
        return [start_state] # Path is just the starting state
    end

    open_set = PriorityQueue{Vector{Tuple{Int, Int}}, Int}()
    enqueue!(open_set, start_state => heuristic(start_state, goal_state))

    came_from = Dict{Vector{Tuple{Int, Int}}, Vector{Tuple{Int, Int}}}()

    g_score = Dict{Vector{Tuple{Int, Int}}, Int}()
    g_score[start_state] = 0

    f_score = Dict{Vector{Tuple{Int, Int}}, Int}()
    f_score[start_state] = heuristic(start_state, goal_state)

    println("Starting A*...")
    search_steps = 0
    max_search_steps = 100000 # Limit search space exploration

    while !isempty(open_set) && search_steps < max_search_steps
        search_steps += 1
        if search_steps % 1000 == 0
            println("Search steps: $search_steps, Open set size: $(length(open_set))")
        end

        current_state = dequeue!(open_set)

        if current_state == goal_state
            println("Goal reached! Reconstructing path...")
            path = Vector{Vector{Tuple{Int, Int}}}()
            temp = current_state
            while haskey(came_from, temp)
                push!(path, temp)
                temp = came_from[temp]
            end
            push!(path, start_state)
            reverse!(path)
            println("Path found with length $(length(path) - 1) steps.")
            return path
        end

        neighbors = get_neighbors(current_state, env)

        for neighbor_state in neighbors
            # Cost to move is always 1 time step
            tentative_g_score = get(g_score, current_state, typemax(Int)) + 1

            if tentative_g_score < get(g_score, neighbor_state, typemax(Int))
                # This path to neighbor is better than any previous one. Record it.
                came_from[neighbor_state] = current_state
                g_score[neighbor_state] = tentative_g_score
                h = heuristic(neighbor_state, goal_state)
                f = tentative_g_score + h
                f_score[neighbor_state] = f

                # Use neighbor_state as key directly. If it's already there, update priority.
                open_set[neighbor_state] = f
            end
        end
    end

    if search_steps >= max_search_steps
         println("Search limit reached.")
    end
    println("No path found.")
    return nothing # No path found
end

# --- Visualization ---

function visualize_path(env::Environment, path::Union{Nothing, Vector{Vector{Tuple{Int, Int}}}})
    if path === nothing
        println("Cannot visualize, no path found.")
        # Optionally plot just the static environment
        plot_static_environment(env)
        return
    end

    height, width = env.dims
    num_robots = length(env.starts)
    colors = distinguishable_colors(num_robots, [RGB(1,1,1), RGB(0,0,0)], lchoices=range(40, stop=80, length=15)) # Generate distinct colors

    println("Generating animation...")

    anim = @animate for t in 1:length(path)
        state = path[t]
        timestep = t - 1 # 0-indexed time

        # Base plot: grid, obstacles, starts, goals
        p = plot(
            xlims=(0.5, width + 0.5), ylims=(0.5, height + 0.5), aspect_ratio=:equal,
            title="Multi-Robot Path (Time: $timestep)", xlabel="X", ylabel="Y",
            size=(600, 600), legend=false,
            xticks=1:width, yticks=1:height, yflip=true, # Y axis inverted for typical matrix layout
            grid=true, gridalpha=0.3, framestyle=:box
        )

        # Plot obstacles (gray squares)
        for obs in env.obstacles
            plot!(p, [obs[2]-0.5, obs[2]+0.5, obs[2]+0.5, obs[2]-0.5, obs[2]-0.5],
                     [obs[1]-0.5, obs[1]-0.5, obs[1]+0.5, obs[1]+0.5, obs[1]-0.5],
                     seriestype=:shape, fillcolor=:darkgray, linecolor=:black)
        end

        # Plot start positions (squares) and goal positions (stars)
        for i in 1:num_robots
            # Start - Square marker
            scatter!(p, [env.starts[i][2]], [env.starts[i][1]], marker=:square, markersize=8, color=colors[i], markerstrokecolor=:black, label="Robot $i Start")
            # Goal - Star marker
            scatter!(p, [env.goals[i][2]], [env.goals[i][1]], marker=:star5, markersize=10, color=colors[i], markerstrokecolor=:black, label="Robot $i Goal")
        end

        # Plot current robot positions (circles)
        for i in 1:num_robots
             # Add small offset for better visibility if multiple robots are close
             offset_x = (rand() - 0.5) * 0.1
             offset_y = (rand() - 0.5) * 0.1
             scatter!(p, [state[i][2] + offset_x], [state[i][1] + offset_y], marker=:circle, markersize=8, color=colors[i], markerstrokecolor=colors[i], label="Robot $i (t=$timestep)")
             # Add robot number inside the circle (adjust position slightly)
             # annotate!(p, state[i][2], state[i][1] + 0.1, text("$i", :white, :center, 8))
        end

        # Add margin to prevent title overlap
        plot!(p, bottom_margin=10mm, left_margin=5mm)

    end # end animation loop

    # Save the animation
    gif_filename = "mapf_animation_$(num_robots)robots_$(height)x$(width).gif"
    gif(anim, gif_filename, fps=2)
    println("Animation saved to $gif_filename")

    # Display the first frame statically if needed (e.g., in Pluto or VS Code)
     # display(anim[1]) # Uncomment if you want to see the first frame immediately

end

function plot_static_environment(env::Environment)
    height, width = env.dims
    num_robots = length(env.starts)
    colors = distinguishable_colors(num_robots, [RGB(1,1,1), RGB(0,0,0)], lchoices=range(40, stop=80, length=15))

    p = plot(
        xlims=(0.5, width + 0.5), ylims=(0.5, height + 0.5), aspect_ratio=:equal,
        title="Environment Setup", xlabel="X", ylabel="Y",
        size=(600, 600), legend=:outertopright,
        xticks=1:width, yticks=1:height, yflip=true,
        grid=true, gridalpha=0.3, framestyle=:box
    )

    # Plot obstacles
    for obs in env.obstacles
        plot!(p, [obs[2]-0.5, obs[2]+0.5, obs[2]+0.5, obs[2]-0.5, obs[2]-0.5],
                 [obs[1]-0.5, obs[1]-0.5, obs[1]+0.5, obs[1]+0.5, obs[1]-0.5],
                 seriestype=:shape, fillcolor=:darkgray, linecolor=:black, label= (obs == first(env.obstacles) ? "Obstacle" : ""))
    end

    # Plot starts and goals
    for i in 1:num_robots
        scatter!(p, [env.starts[i][2]], [env.starts[i][1]], marker=:square, markersize=8, color=colors[i], markerstrokecolor=:black, label="Robot $i Start")
        scatter!(p, [env.goals[i][2]], [env.goals[i][1]], marker=:star5, markersize=10, color=colors[i], markerstrokecolor=:black, label="Robot $i Goal")
    end
     plot!(p, bottom_margin=10mm, left_margin=5mm)
    display(p) # Display the static plot
end


# --- Main Execution ---

function main()
    Random.seed!(1234) # For reproducible random generation

    # Generate Environment
    height, width = MAP_HEIGHT, MAP_WIDTH
    num_robots = NUM_ROBOTS

    # Generate valid, non-overlapping start and goal positions
    possible_positions = Set((r, c) for r in 1:height for c in 1:width)
    starts = Vector{Tuple{Int, Int}}(undef, num_robots)
    goals = Vector{Tuple{Int, Int}}(undef, num_robots)
    used_positions = Set{Tuple{Int, Int}}()

    for i in 1:num_robots
        # Generate unique start position
        while true
            start_pos = rand(collect(possible_positions))
            if start_pos ∉ used_positions
                starts[i] = start_pos
                push!(used_positions, start_pos)
                break
            end
        end
         # Generate unique goal position (different from start and other goals/starts)
         while true
             goal_pos = rand(collect(possible_positions))
             if goal_pos ∉ used_positions
                 goals[i] = goal_pos
                 push!(used_positions, goal_pos)
                 break
             end
         end
    end

    # Generate obstacles, avoiding start/goal positions
    available_for_obstacles = setdiff(possible_positions, used_positions)
    num_obstacles = floor(Int, OBSTACLE_DENSITY * (height * width - 2 * num_robots))
    obstacles = Set(rand(collect(available_for_obstacles), min(num_obstacles, length(available_for_obstacles))))

    env = Environment((height, width), obstacles, starts, goals)

    println("Environment:")
    println("  Dimensions: $height x $width")
    println("  Num Robots: $num_robots")
    println("  Starts: $starts")
    println("  Goals: $goals")
    println("  Num Obstacles: $(length(obstacles))")

    # Plot the static environment setup first
    plot_static_environment(env)

    # Run A*
    println("\nRunning Joint A* Search...")
    tick()
    path = joint_a_star(env)
    tock() # Prints elapsed time

    # Visualize the result
    if path !== nothing
        visualize_path(env, path)
    else
        println("A* search failed to find a path.")
    end
end

# Execute the main function
main()