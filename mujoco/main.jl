using MuJoCo

# Load a model file (XML format)
# Since you're now inside the mujoco folder, you can use a direct reference
# or use "./" to indicate the current directory
model = MuJoCo.mj_loadXML("./two_arm_vention_table.xml", "", Cstring(C_NULL))

# Create a data instance for the model
data = MuJoCo.mj_makeData(model)

# Step the simulation
MuJoCo.mj_step(model, data)

# Access state information
position = data.qpos
velocity = data.qvel