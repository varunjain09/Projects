import pybullet as p
import pybullet_data

def setup_simulation(slope_angle=0):
    """
    Initializes PyBullet simulation with gravity and a flat plane.
    Returns the simulation plane ID.
    """
    if p.getConnectionInfo()["isConnected"]:
        print("PyBullet already connected. Reusing existing connection.")
    else:
        p.connect(p.GUI)  # Only connect if not already connected
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    plane_id = p.loadURDF("plane.urdf")
    return plane_id


def load_robot(xml_path):
    """
    Loads the robot from an XML/URDF file.
    :param xml_path: Path to the robot XML file.
    :return: Robot ID.
    """
    robot_id = p.loadURDF(xml_path)
    return robot_id

def apply_torque(robot_id, left_wheel_id, right_wheel_id, torque):
    """
    Applies torque to the robot's wheels.
    """
    try:


    # Apply torque to left wheel
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-torque
        )

        # Apply torque to right wheel
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=torque
        )
    except Exception as e:
        print(f"Error applying torque:{e}")



def add_target_point(position, color=(1, 0, 0, 1), radius=0.5):
    """
    Adds a target point (sphere) to the simulation.
    
    :param position: List [x, y, z] coordinates of the target point.
    :param color: Tuple specifying RGBA color of the sphere.
    :param radius: Radius of the sphere.
    :return: Target point ID.
    """
    # Create a visual sphere for the target
    marker_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color  # Default is red (1, 0, 0, 1)
    )
    
    # Create the multi-body for the target point
    target_point = p.createMultiBody(
        baseVisualShapeIndex=marker_visual,
        basePosition=position
    )
    return target_point
