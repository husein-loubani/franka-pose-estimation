import logging
import os
from types import SimpleNamespace

import deepdish as dd
import numpy as np
import pybullet as pb
import pybullet_data
import pybulletX as pX
import tacto
import yaml
from robot import Robot

import cv2

### tacto headless rendering
### pip install git+https://github.com/mmatl/pyopengl.git@76d1261adee2d3fd99b418e75b0416bb7d2865e6
## osmesa cpu rendering
## egl gpu rendering
# os.environ["PYOPENGL_PLATFORM"] = "egl" 

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs.log", level=logging.DEBUG)


class Camera:
    def __init__(self, cameraResolution=[320, 240], visualize_gui=True):
        self.cameraResolution = cameraResolution
        self.visualize_gui = visualize_gui

        camTargetPos = [0.5, 0, 0.05]
        camDistance = 0.4
        upAxisIndex = 2

        yaw = 90
        pitch = -30.0
        roll = 0

        fov = 60
        self.nearPlane = 0.05
        self.farPlane = 20

        self.viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = cameraResolution[0] / cameraResolution[1]

        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov, aspect, self.nearPlane, self.farPlane
        )

    def get_image(self):
        img_arr = pb.getCameraImage(
            self.cameraResolution[0],
            self.cameraResolution[1],
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]  # color data RGB
        
        dep_buffer = img_arr[3]  # depth data
        dep = self.farPlane * self.nearPlane / (self.farPlane - (self.farPlane - self.nearPlane) * dep_buffer) 
        dep /= self.farPlane
        
        msk = img_arr[4]  # mask data
        
        return rgb, dep, msk
    
    def _depth_to_color(self, depth):
        gray = (np.clip(np.ones(depth.shape).astype(float) - depth, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    
    def updateGUI(self, color, depth, mask):
        """
        Update images for visualization
        """
        if not self.visualize_gui:
            return
        
        masks = []
        for l in np.arange(np.min(mask), np.max(mask) + 1):
            l_imag = np.expand_dims(np.where(mask == l, 255, 0).astype(np.uint8), axis=-1).repeat(4, axis=-1)
            masks.append(l_imag)

        # concatenate the resulting two images horizontal (axis=1)
        color_n_depth = np.concatenate([color, self._depth_to_color(depth)] + masks, axis=1)

        cv2.imshow(
            "Camera", cv2.cvtColor(color_n_depth, cv2.COLOR_RGB2BGR)
        )

        cv2.waitKey(1)
        

def get_forces(bodyA=None, bodyB=None, linkIndexA=None, linkIndexB=None):
    """
    get contact forces

    :return: normal force, lateral force
    """
    kwargs = {
        "bodyA": bodyA,
        "bodyB": bodyB,
        "linkIndexA": linkIndexA,
        "linkIndexB": linkIndexB,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    pts = pb.getContactPoints(**kwargs)

    totalNormalForce = 0
    totalLateralFrictionForce = [0, 0, 0]

    for pt in pts:
        totalNormalForce += pt[9]

        totalLateralFrictionForce[0] += pt[11][0] * pt[10] + pt[13][0] * pt[12]
        totalLateralFrictionForce[1] += pt[11][1] * pt[10] + pt[13][1] * pt[12]
        totalLateralFrictionForce[2] += pt[11][2] * pt[10] + pt[13][2] * pt[12]

    return totalNormalForce, totalLateralFrictionForce


class Log:
    def __init__(self, dirName, id=0):
        self.dirName = dirName
        self.id = id
        os.makedirs(dirName, exist_ok=True)

    def save(
        self,
        tactileColorL,
        tactileColorR,
        tactileDepthL,
        tactileDepthR,
        visionColor,
        visionDepth,
        visionMask,
        gripForce,
        normalForce,
        label,
        objPose,
        endEffectorPose,
        gripperPose,
    ):
        data = {
            "tactileColorL": tactileColorL,
            "tactileColorR": tactileColorR,
            "tactileDepthL": tactileDepthL,
            "tactileDepthR": tactileDepthR,
            "visionColor": visionColor,
            "visionDepth": visionDepth,
            "visionMask": visionMask,
            "gripForce": gripForce,
            "normalForce": normalForce,
            "objPose": objPose,
            "label": label,
            "endEffectorPose": endEffectorPose,
            "gripperPose": gripperPose,
        }

        id_str = "{:07d}".format(self.id)
        outputDir = os.path.join(self.dirName, id_str)
        os.makedirs(outputDir, exist_ok=True)

        newData = {k: [] for k in data.keys()}
        for k in data.keys():
            if isinstance(data[k], list):
                newData[k] = data[k]
                
        newData["label"] = list(np.ones((len(data["visionColor"]))).astype(float) * data["label"])
        newData["gripForce"] = list(np.ones((len(data["visionColor"]))).astype(float) * data["gripForce"])

        for k in data.keys():
            fn_k = "{}_{}.h5".format(id_str, k)
            outputFn = os.path.join(outputDir, fn_k)
            dd.io.save(outputFn, newData[k])

        self.id += 1


def load_config(filepath: str = "/home/hussein-lobs/franka-pose-estimation/object_pose_estimation/config.yaml"):
    with open(filepath, "r") as f:
        config = SimpleNamespace(**yaml.load(f, Loader=yaml.FullLoader))
    return config


class Env:
    def __init__(self, conf):
        self.conf = conf

        self.log = Log("data/pose_estimation")

        # Initialize World
        logging.info("Initializing world")
        if conf.physicsClientType == "gui":
            physicsClient = pb.connect(pb.GUI)
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            # pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING,0)
        else:
            if conf.device == "cpu":
                physicsClient = pb.connect(pb.DIRECT)
            elif conf.device == "gpu":
                physicsClient = pb.connect(pb.SHARED_MEMORY_SERVER)
                os.environ["PYOPENGL_PLATFORM"] = "egl" 
                
                import pkgutil
                egl = pkgutil.get_loader('eglRenderer')
                
                plugin = pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                print("plugin=", plugin)

                pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
                pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            else:
                raise ValueError("Invalid device")
            
            

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

        # Initialize digits
        self.digits = tacto.Sensor(
            width=240, height=320, visualize_gui=conf.digitsVisualizeGui)

        pb.resetDebugVisualizerCamera(
            cameraDistance=0.6,
            cameraYaw=15,
            cameraPitch=-20,
            # cameraTargetPosition=[-1.20, 0.69, -0.77],
            cameraTargetPosition=[0.5, 0, 0.08],
        )

        # Create plane
        planeId = pb.loadURDF("plane.urdf")

        # Initilize robot
        robotURDF = "/home/hussein-lobs/franka-pose-estimation/object_pose_estimation/setup/franka_description/robots/fr3/fr3.urdf"
        self.robotID = pb.loadURDF(robotURDF, useFixedBase=True)
        self.rob = Robot(self.robotID)

        # Add camera to robot
        self.cam = Camera(visualize_gui=conf.digitsVisualizeGui)

        # Move robot to initial position
        self.rob.go(self.rob.pos, wait=True)

        # Add tactile sensors to robot
        sensorLinks = self.rob.get_id_by_name(
            ["joint_finger_tip_right", "joint_finger_tip_left"]
        )  # [21, 24]
        self.digits.add_camera(self.robotID, sensorLinks)

        nbJoint = pb.getNumJoints(self.robotID)

        # Add object to pybullet and tacto simulator
        urdfObj = "setup/objects/011_banana/banana.urdf"
        globalScaling = 0.6
        self.objStartPos = [0.50, 0, 0.05]
        self.objStartOrientation = pb.getQuaternionFromEuler([0, 0, np.pi / 2])

        body = pX.Body(urdfObj, self.objStartPos, self.objStartOrientation, global_scaling=globalScaling)
        self.objID = body.id
        self.digits.add_body(body)

        self.sensorID = self.rob.get_id_by_name(
            ["joint_finger_tip_right", "joint_finger_tip_left"])

        self.dz = 0.003
        posList = [
            [0.50, 0, 0.205],
            [0.50, 0, 0.213],
            [0.50, 0.03, 0.205],
            [0.50, 0.03, 0.213],
        ]
        posID = 0
        self.pos = posList[posID].copy()

        self.t = 0
        self.gripForce = 20

        self.tactileColorDef, self.tactileDepthDef = self.digits.render()

        self.tactileColorLList = []
        self.tactileColorRList = []
        self.tactileDepthLList = []
        self.tactileDepthRList = []
        self.visionColorList = []
        self.visionDepthList = []
        self.visionMaskList = []
        self.normalForceList = []
        self.objPosList = []
        self.ee_pose_list = []
        self.gripper_width_list = []

    def get_object_pose(self):
        res = pb.getBasePositionAndOrientation(self.objID)

        world_positions = res[0]
        world_orientations = res[1]

        # Check if object is out of reach
        if (world_positions[0] ** 2 + world_positions[1] ** 2) > 0.8 ** 2:
            pb.resetBasePositionAndOrientation(
                self.objID, self.objStartPos, self.objStartOrientation)
            return self.objStartPos, self.objStartOrientation

        world_positions = np.array(world_positions)
        world_orientations = np.array(world_orientations)

        return (world_positions, world_orientations)

    def record_sensor_states(self):
        visionColor, visionDepth, visionMask = self.cam.get_image()

        normalForce0, lateralForce0 = get_forces(
            self.robotID, self.objID, self.sensorID[0], -1)
        normalForce1, lateralForce1 = get_forces(
            self.robotID, self.objID, self.sensorID[1], -1)

        if normalForce0 > 0 or normalForce1 > 0:
            tactileColor, tactileDepth = self.digits.render()
        else:
            tactileColor, tactileDepth = self.tactileColorDef, self.tactileDepthDef

        self.digits.updateGUI(tactileColor, tactileDepth)
        self.cam.updateGUI(visionColor, visionDepth, visionMask)

        objPos, objOri = self.get_object_pose()

        # Record sensor states
        if self.t % self.conf.save_each == 0:
            tactileColorL, tactileColorR = tactileColor[0], tactileColor[1]
            tactileDepthL, tactileDepthR = tactileDepth[0], tactileDepth[1]
            self.tactileColorLList.append(tactileColorL)
            self.tactileColorRList.append(tactileColorR)
            self.tactileDepthLList.append(tactileDepthL)
            self.tactileDepthRList.append(tactileDepthR)

            self.visionColorList.append(visionColor)
            self.visionDepthList.append(visionDepth)
            self.visionMaskList.append(visionMask)

            self.normalForceList.append([normalForce0, normalForce1])

            self.objPosList.append([objPos, objOri])
            self.ee_pose_list.append(self.rob.get_ee_pose())
            self.gripper_width_list.append(self.rob.get_gripper_width())

    def pick_and_place(self):
        moving = True
        self.t += 1
        if self.t <= 60:
            # Reaching
            self.rob.go(self.pos, width=0.11)
        elif self.t < 140:
            # Grasping
            self.rob.go(self.pos, width=0.03, gripForce=self.gripForce)
        elif self.t == 140:
            # Record the object pose
            self.objPos0, _ = self.get_object_pose()
        elif self.t > 140 and self.t <= 200:
            # Lift
            self.pos[-1] += self.dz
            self.rob.go(self.pos, wait=False)
        elif self.t > 240:
            # Save the data few frames after the object is lifted
            objPos, _ = self.get_object_pose()

            if objPos[2] - self.objPos0[2] < 60 * self.dz * 0.8:
                # Fail
                label = 0
            else:
                # Success
                label = 1

            self.log.save(
                self.tactileColorLList,
                self.tactileColorRList,
                self.tactileDepthLList,
                self.tactileDepthRList,
                self.visionColorList,
                self.visionDepthList,
                self.visionMaskList,
                self.gripForce,
                self.normalForceList,
                label,
                self.objPosList,
                self.ee_pose_list,
                self.gripper_width_list,
            )
            print("\rsample {}".format(self.log.id), end="")

            if self.log.id > self.conf.num_frames:
                moving = False

            # Reset
            self.t = 0

            self.tactileColorLList.clear()
            self.tactileColorRList.clear()
            self.tactileDepthLList.clear()
            self.tactileDepthRList.clear()
            self.visionColorList.clear()
            self.visionDepthList.clear()
            self.visionMaskList.clear()
            self.normalForceList.clear()
            self.objPosList.clear()
            self.ee_pose_list.clear()
            self.gripper_width_list.clear()

            self.rob.reset_robot()

            # Reset randomly object pose
            objRestartPos = [
                0.50 + 0.1 * np.random.random(),
                -0.15 + 0.3 * np.random.random(),
                0.05,
            ]
            objRestartOrientation = pb.getQuaternionFromEuler(
                [0, 0, 2 * np.pi * np.random.random()]
            )

            # Reset randomly robot pose
            self.pos = [
                objRestartPos[0] + np.random.uniform(-0.02, 0.02),
                objRestartPos[1] + np.random.uniform(-0.02, 0.02),
                objRestartPos[2] * (1 + np.random.random() * 0.5) + 0.14,
            ]
            ori = [0, np.pi, 2 * np.pi * np.random.random()]

            # Reset randomly gripper force
            self.gripForce = 5 + np.random.random() * 15

            # Reset robot and object
            self.rob.go(self.pos + np.array([0, 0, 0.1]), ori=ori, width=0.11)
            pb.resetBasePositionAndOrientation(
                self.objID, objRestartPos, objRestartOrientation)
            for i in range(100):
                pb.stepSimulation()

        return moving


print("\n")

env = Env(load_config())
while True:
    if not env.pick_and_place():
        break
    env.record_sensor_states()

    pb.stepSimulation()

pb.disconnect()  # Close PyBullet
