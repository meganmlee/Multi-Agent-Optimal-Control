# visualization_utils.jl
import MeshCat as mc
using StaticArrays
using LinearAlgebra
using Rotations
using CoordinateTransformations
import DifferentiableCollisions as dc

# Assumes RobotKinematics struct is available
# Assumes NJ, NUM_ROBOTS, N_LINKS constants are available

"""
Sets up the basic MeshCat environment.
"""
function setup_visualization_environment(vis::mc.Visualizer)
    mc.setprop!(vis["/Background"], "top_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
    mc.setprop!(vis["/Background"], "bottom_color", mc.RGBA(1.0, 1.0, 1.0, 1.0))
    mc.delete!(vis["/Grid"])
    mc.delete!(vis["/Axes"])
end

"""
Builds the robot link geometries in MeshCat using DCOL primitives.
"""
function build_robot_visualization(vis::mc.Visualizer, P_links, link_radii1, link_radii2, link_lengths_geom)
     colors = [mc.RGBA(1.0, 0.0, 0.0, 0.8), mc.RGBA(0.0, 0.0, 1.0, 0.8)]
     link_vis_names = [[Symbol("R$(r)_L$(l)") for l=1:N_LINKS] for r=1:NUM_ROBOTS]
     println("Building visual primitives...")
     for r = 1:NUM_ROBOTS
         for l = 1:N_LINKS
             # Use the actual primitive object from P_links to ensure type match
             primitive_obj = P_links[r][l]
             vis_name = link_vis_names[r][l]
             vis_color = colors[r]
             # Build based on the primitive type stored in P_links
             dc.build_primitive!(vis, primitive_obj, vis_name; color = vis_color)
         end
     end
     return link_vis_names
end


"""
Animates a given state trajectory X_sol in MeshCat.
"""
function animate_trajectory(vis::mc.Visualizer, X_sol::Vector{Vector{T}}, robot_kin, P_links, link_vis_names, fps::Float64) where T
    if X_sol === nothing || isempty(X_sol)
        println("Cannot animate empty trajectory.")
        return
    end
    N = length(X_sol)
    NJ = Int(length(X_sol[1]) / 6) # Infer NJ
    NX_PER_ROBOT = 3 * NJ

    anim = mc.Animation(floor(Int, fps))
    println("Creating Animation...")
    for k = 1:N
        mc.atframe(anim, k) do
            x_k = X_sol[k]
            q1_k = x_k[1:NJ]
            q2_k = x_k[NX_PER_ROBOT .+ (1:NJ)]
            all_q_k = [q1_k, q2_k]

            for r = 1:NUM_ROBOTS
                # Calculate FK for current robot
                # Ensure forward_kinematics_poe is accessible here
                poses_world_k, _, _, _ = forward_kinematics_poe(all_q_k[r], robot_kin[r])

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
        end # end atframe
    end # end loop k
    println("Setting Animation...")
    mc.setanimation!(vis, anim)
end