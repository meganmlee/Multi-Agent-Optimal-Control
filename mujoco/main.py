import mujoco as mj
from mujoco import viewer
import numpy as np
from one_arm_pid_control import SingleArmPIDControl
import time

if __name__ == "__main__":

    # Set the XML filepath
    xml_filepath = "/home/megan/magic/ros2_ws/src/control/kinova_gen3_mujoco/two_arm_vention_table.xml"

    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)
    model.opt.timestep = 0.001
    mj.mj_forward(model, data)

    ################ PID Control Testing ######################

    controller = SingleArmPIDControl(model, data)
    controller.go_to_home(model, data)

    # Launch the simulate viewer
    viewer.launch(model, data)