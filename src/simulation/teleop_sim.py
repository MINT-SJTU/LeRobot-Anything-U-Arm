import serial
import time 
import numpy as np
import re
import gymnasium as gym
import mani_skill.envs  # Must import to register all env/agent
from threading import Event, Thread, Lock
from queue import Queue, Empty
import torch
import sapien
import argparse
from mani_skill.utils import sapien_utils


class ServoTeleoperatorSim: 
    """Robot arm teleoperation simulation system
    
    Supports reading servo angles through serial port and mapping to different types of robot arm simulation environments.
    Supported robot arm types: arx-x5, so100, xarm6_robotiq, panda, x_fetch, unitree_h1
    """
    
    def __init__(self, scene: str, robot_uids: str, serial_port: str = '/dev/ttyUSB0'):
        """Initialize teleoperation system
        
        Args:
            scene: Simulation scene name
            robot_uids: Robot arm type identifier
            serial_port: Serial port device path
        """
        # Serial port configuration
        self.SERIAL_PORT = serial_port
        self.BAUDRATE = 115200
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.01)

        # System configuration
        self.scene = scene
        self.robot_uids = robot_uids
        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7  # Initial calibration angles for servos
        self.sim_init_angles = [0.0] * 7  # Simulation initial angles
        self.stop_event = Event()
        self.rate = 50.0  # Control frequency
        
        # Initialize servos and calibrate zero position
        self._init_servos()

        # Thread-safe data exchange queue
        self.arm_pos_queue: "Queue[list]" = Queue(maxsize=1)

        # Select control mode based on robot type
        if robot_uids == "x_fetch": 
            self.control_mode = "pd_joint_pos_dual_arm"
        elif robot_uids == "unitree_h1":
            self.control_mode = "pd_joint_pos"
        else: 
            self.control_mode = "pd_joint_pos"

        # Create simulation environment
        self.env = gym.make(
            scene,
            robot_uids=robot_uids,
            render_mode="human",
            control_mode=self.control_mode,
            sensor_configs=dict(shader_pack="rt-fast"),
            human_render_camera_configs=dict(shader_pack="rt-fast"),
            viewer_camera_configs=dict(shader_pack="rt-fast"),
            sim_config=dict(
                default_materials_config=dict(
                    static_friction=10.0,  # Static friction
                    dynamic_friction=10.0, # Dynamic friction
                    restitution=0.0       # Restitution coefficient
                )
            ),
        )
        obs, _ = self.env.reset(seed=0)
        print("Action space:", self.env.action_space)
        
        # Set initial standing pose for H1
        if robot_uids == "unitree_h1":
            self._setup_h1_standing_pose()

        # Create producer thread (read servo angles)
        self.produce_thread = Thread(
            target=self.angle_stream_loop, 
            args=(self.default_sender,), 
            daemon=True
        )

        # Create consumer thread (control simulation)
        self.consume_thread = Thread(
            target=self.pose_consumer_loop, 
            args=(self.teleop_sim_handler,), 
            daemon=True
        )

        self._setup_camera_pose()


    def _setup_camera_pose(self):
        agent = getattr(self.env.unwrapped, "agent", None)
        pose = sapien.Pose()
        if agent is not None:
            pose = agent.robot.get_pose()  # Returns sapien.Pose
            print(f"Robot initial position: {pose}")
        camera_pose = sapien_utils.look_at(
            [0.0, -1.5, 1.7], pose.p
        )
        camera_viewer = getattr(self.env.unwrapped, "viewer", None)
        if camera_viewer is not None:
            print(camera_pose)
            camera_pose_arr = camera_pose.raw_pose.squeeze().cpu().numpy()
            camera_position = camera_pose_arr[:3]
            camera_quaternion = camera_pose_arr[3:]
            camera_viewer.set_camera_pose(sapien.Pose(camera_position, camera_quaternion))

    def _setup_h1_standing_pose(self):
        """Set initial standing pose for H1 robot"""
        try:
            agent = getattr(self.env.unwrapped, "agent", None)
            if agent is not None:
                # Use H1 predefined standing pose
                standing_keyframe = agent.keyframes["standing"]
                
                # Check qpos dimensions
                if hasattr(standing_keyframe.qpos, '__len__') and len(standing_keyframe.qpos) >= 19:
                    agent.reset(standing_keyframe.qpos)
                    agent.robot.set_root_pose(standing_keyframe.pose)
                    print("H1 set to standing pose")
                else:
                    print("Warning: standing_keyframe.qpos dimensions incorrect, using default standing pose")
                    # Use default standing pose
                    default_standing = np.array([
                        0, 0, 0, 0, 0, 0, 0, -0.4, -0.4, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, -0.4, -0.4, 0.0, 0.0
                    ])
                    agent.reset(default_standing)
                    agent.robot.set_root_pose(standing_keyframe.pose)
                    print("H1 set to default standing pose")
        except Exception as e:
            print(f"Failed to set H1 standing pose: {e}")
    
    def _init_servos(self):
        """Initialize servos and calibrate zero position angles"""
        self.send_command('#000PVER!')
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip(), i)
            self.zero_angles[i] = angle if angle is not None else 0.0
        print("[INFO] Servo initial angle calibration completed")

    def send_command(self, cmd: str) -> str:
        """Send serial command and read response
        
        Args:
            cmd: Command string to send
            
        Returns:
            Response string, returns empty string if no response
        """
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        response = self.ser.read_all()
        return response.decode('ascii', errors='ignore') if response else ""
    
    def pwm_to_angle(response_str, servo_num, pwm_min=500, pwm_max=2500, angle_range=270):
        """Convert PWM response to angle
        
        Args:
            response_str: Servo response string
            pwm_min: PWM minimum value
            pwm_max: PWM maximum value
            angle_range: Angle range (degrees)
            
        Returns:
            Angle value, returns None if parsing fails
        """
        pattern = f"#{servo_num:03d}P(\\d{{4}})"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle
    
    def publish_arm_pos(self, arm_pos: list):
        """Publish latest arm position to queue, overwriting old values"""
        try:
            # Clear old data from queue
            while True:
                self.arm_pos_queue.get_nowait()
        except Empty:
            pass
        try:
            # Add new data
            self.arm_pos_queue.put_nowait(list(arm_pos))
        except Exception:
            pass
    
    def get_latest_arm_pos(self, timeout: float = 0.0):
        """Get latest arm position snapshot
        
        Args:
            timeout: Timeout time, 0 means return immediately
            
        Returns:
            Latest arm position list, returns None if queue is empty
        """
        try:
            return self.arm_pos_queue.get(timeout=timeout) if timeout and timeout > 0 else self.arm_pos_queue.get_nowait()
        except Empty:
            return None
    
    def angle_to_gripper(self, angle_rad: float, pos_min: float, pos_max: float, 
                        angle_range: float = 1.5 * np.pi) -> float:
        """Map servo angle to gripper position
        
        Args:
            angle_rad: Servo angle (radians)
            pos_min: Gripper minimum position
            pos_max: Gripper maximum position
            angle_range: Servo angle range
            
        Returns:
            Gripper position value
        """
        ratio = max(0, 1 - (angle_rad / angle_range))
        position = pos_min + (pos_max - pos_min) * ratio
        return float(np.clip(position, pos_min, pos_max))

    def convert_pose_to_action(self, pose: list) -> np.ndarray: 
        """Convert servo position to simulation action based on different robot arm types
        
        Args:
            pose: 7-dimensional servo angle list (radians)
            
        Returns:
            Corresponding robot arm action vector
        """
        action = np.array([])

        if self.robot_uids == "arx-x5":  # 6-axis robot arm + dual-finger gripper
            action = np.array(pose)
            # Handle gripper: map last dimension to gripper position
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.044)
            action = np.concatenate([action, [action[-1]]])

            action[2] = -action[2]
            action[4], action[5] = -action[5], -action[4]  # Swap joints 4 and 5
        
        elif self.robot_uids == "piper":  # 6-axis robot arm + dual-finger gripper
            action = np.array(pose)
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.04)

            action = np.concatenate([action, [action[-1]]])
            action[3], action[4] = action[4], -action[3]  # Swap joints 4 and 5

        elif self.robot_uids == "so100":  # 5-axis robot arm
            pose_copy = pose.copy()
            pose_copy.pop(5)  # Remove 6th dimension (so100 only has 5 axes)
            action = np.array(pose_copy)
            action[-1] = self.angle_to_gripper(action[-1], -1.1, 1.1)
            
            action[0] = -action[0]
            action[3] = -action[3]
            action[4] = -action[4]

        elif self.robot_uids == "xarm6_robotiq":  # 6-axis robot arm + Robotiq gripper
            action = np.array(pose)
            action[3], action[4] = action[4], -action[3]  # Swap joints 3 and 4
            # action[1] = -action[1]
            action[-1] = 0.81 - self.angle_to_gripper(action[-1], 0, 0.81)

        elif self.robot_uids == "panda":  # 7-axis robot arm
            pose_copy = pose.copy()
            pose_copy.insert(2, 0.0)  # Insert 0 at 3rd position (Panda's 3rd joint)
            action = np.array(pose_copy)
            # action[1] = -action[1]
            action[3] = -action[3]
            action[4], action[5] = action[5], action[4]  # Swap joints 4 and 5
            action[-1] = self.angle_to_gripper(action[-1], -1.0, 1.0)

        elif self.robot_uids == "x_fetch":  # Dual-arm robot + mobile base
            pose_copy = pose.copy()
            pose_copy.pop(5)  # Remove 6th dimension
            action = np.array(pose_copy)
            action[-1] = self.angle_to_gripper(action[-1], -1.1, 1.1)
            # Adjust joint directions
            action[0] = -action[0]
            action[1] = -action[1]
            action[3] = -action[3]
            action[4] = -action[4]
            # Build dual-arm action
            left_arm_action = action.copy()
            right_arm_action = left_arm_action.copy()
            right_arm_action[0] = -right_arm_action[0]
            right_arm_action[-2] = -right_arm_action[-2]
            # Combine: left arm joints + right arm joints + left/right grippers + base motion (0)
            zero_action = np.zeros(6)
            action = np.concatenate([
                left_arm_action[0:-1], 
                right_arm_action[0:-1], 
                [left_arm_action[-1], right_arm_action[-1]], 
                np.zeros(4) 
            ])

        elif self.robot_uids == "widowx250s":  # 6-axis robot arm + dual-finger gripper
            action = np.array(pose)
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.04)
            action = np.concatenate([action, [action[-1]]])
            action[3], action[4] = action[4], -action[3] 
        
        elif self.robot_uids == "unitree_h1":  # Humanoid robot
            raw = np.array(pose, dtype=np.float32)
            action = np.zeros(19, dtype=np.float32)

            # Only modify arm joint increments (relative to current state)
            # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            action[5] = raw[0]  # left_shoulder_pitch increment
            action[9] = raw[1]   # left_shoulder_roll increment
            action[13] = raw[2]  # left_shoulder_yaw increment
            action[17] = raw[3]  # left_elbow increment

            # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            action[6] = raw[4]   # right_shoulder_pitch increment
            action[10] = raw[5]  # right_shoulder_roll increment
            action[14] = raw[6]  # right_shoulder_yaw increment
            action[18] = raw[3]  # right_elbow increment (reuse 4th servo)

        else: 
            raise ValueError(f"Unsupported robot arm type: {self.robot_uids}")

        return action

    def default_sender(self, arm_pos: list): 
        """Default angle sending callback (for debugging)"""
        print(f"Servo angles (degrees): {np.degrees(arm_pos)}")

    def teleop_sim_handler(self, action: np.ndarray, dwell: float = 0.01):
        """Simulation control handler function
        
        Args:
            action: Robot arm action vector
            dwell: Delay time
        """
        if self.env is None or action is None:
            return
            
        # All robot types execute actions
        self.env.step(action)
        self.env.render()
        time.sleep(dwell)
    
    def angle_stream_loop(self, on_send):
        """Angle data producer thread: periodically read servo angles
        
        Args:
            on_send: Callback function that receives angle data list
        """
        num_joints = 7
        arm_pos = [0.0] * num_joints

        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()

        while not self.stop_event.is_set():
            # Read all joint angles
            for i in range(num_joints):
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip(), i)
                if angle is not None:
                    # Calculate angle relative to zero position
                    new_angle = angle - self.zero_angles[i]
                    arm_pos[i] = np.radians(new_angle)
                else: 
                    raise ValueError(f"Servo {i} response error: {response.strip()}")
            
            # Publish latest data and call callback
            self.publish_arm_pos(arm_pos)
            try:
                on_send(list(arm_pos))
            except Exception as e:
                print(f"Angle sending callback error: {e}")
                
            # Maintain fixed frequency
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                # If falling behind significantly, resync time
                next_time = time.monotonic()

    def pose_consumer_loop(self, on_pose):
        """Simulation control consumer thread: periodically get angle data and control simulation
        
        Args:
            on_pose: Callback function that receives action vector
        """
        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()

        # Safely get action space dimensions
        try:
            action_shape = self.env.action_space.shape[0] if self.env.action_space is not None else 0
        except (AttributeError, TypeError, IndexError):
            action_shape = 0
        print("Action Space Shape:", action_shape)
        
        while not self.stop_event.is_set():
            pose = self.get_latest_arm_pos(timeout=0.0)
            if pose is not None:
                try:
                    action = self.convert_pose_to_action(pose)
                    on_pose(action)
                except Exception as e:
                    print(f"Simulation control callback error: {e}")
                    
            # Maintain fixed frequency
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                next_time = time.monotonic()
    
    def run(self):
        """Start teleoperation system"""
        print("Starting angle reading thread...")
        self.produce_thread.start()
        print("Starting simulation control thread...")
        self.consume_thread.start()
        
        try: 
            print("System running, press Ctrl+C to stop...")
            while True: 
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Received interrupt signal, preparing to stop...")
        finally:
            self.stop_event.set()
            self.produce_thread.join(timeout=2.0)
            self.consume_thread.join(timeout=2.0)
            print("All threads stopped")
            self.env.close()
            self.ser.close()
            print("Resource cleanup completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Robot arm teleoperation simulation program',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--robot', '-r', 
        type=str, 
        default='so100',
        choices=['arx-x5', 'so100', 'xarm6_robotiq', 'panda', 'x_fetch', 'piper', 'widowx250s'],
        help='Select robot arm type to control'
    )
    parser.add_argument(
        '--scene', '-s', 
        type=str, 
        default='ReplicaCAD_SceneManipulation-v1',
        help='Simulation scene name'
    )
    parser.add_argument(
        '--rate', 
        type=float, 
        default=50.0,
        help='Control frequency (Hz)'
    )
    parser.add_argument(
        '--serial-port', 
        type=str, 
        default='/dev/ttyUSB0',
        help='Serial port device path'
    )
    
    args = parser.parse_args()
    
    # Display startup information
    print("=" * 60)
    print("    Robot Arm Teleoperation Simulation System")
    print("=" * 60)
    print(f"Robot arm type: {args.robot}")
    print(f"Simulation scene:   {args.scene}")
    print(f"Control frequency:   {args.rate} Hz")
    print(f"Serial device:   {args.serial_port}")
    print("-" * 60)
    
    # Create and run simulation instance
    try:
        sim = ServoTeleoperatorSim(scene=args.scene, robot_uids=args.robot, serial_port=args.serial_port)
        sim.rate = args.rate
        sim.run()
    except Exception as e:
        print(f"Program runtime error: {e}")
        import traceback
        traceback.print_exc()