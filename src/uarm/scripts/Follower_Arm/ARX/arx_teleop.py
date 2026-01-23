#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import serial
import time
import re
import numpy as np
import os
import sys

# ====== ARX5 Interface Path Configuration ======
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
import arx5_interface as arx5  # You should install arx5 api first, following https://github.com/real-stanford/arx5-sdk/tree/main/python/examples


# ====== Master Arm Serial Reading Class ======
class ServoReader:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.SERIAL_PORT = port
        self.BAUDRATE = baudrate
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.1)
        print(f"[ServoReader] Serial port {self.SERIAL_PORT} opened")

        self.zero_angles = [0.0] * 7
        self.current_angles = [0.0] * 7
        self.lock = threading.Lock()

        self._init_servos()

    def send_command(self, cmd):
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        return self.ser.read_all().decode('ascii', errors='ignore')

    def pwm_to_angle(response_str, servo_num, pwm_min=500, pwm_max=2500, angle_range=270):
        pattern = f"#{servo_num:03d}P(\\d+)"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle

    def _init_servos(self):
        self.send_command('#000PVER!')
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip(), i)
            self.zero_angles[i] = angle if angle is not None else 0.0
        print("[ServoReader] Servo initial angle calibration completed")

    def read_loop(self, hz=100):
        dt = 1.0 / hz
        while True:
            new_angles = [0.0] * 7
            for i in range(7):
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip(), i)
                if angle is not None:
                    new_angles[i] = angle - self.zero_angles[i]
            with self.lock:
                self.current_angles = new_angles
            time.sleep(dt)

    def get_angles(self):
        with self.lock:
            return list(self.current_angles)


# ====== Slave Arm Control Class (with velocity limiting) ======
class ArxTeleop:
    def __init__(self, model="X5", interface="can0"):
        self.ctrl = arx5.Arx5JointController(model, interface)
        self.robot_cfg = self.ctrl.get_robot_config()
        self.ctrl_cfg = self.ctrl.get_controller_config()
        self.dof = self.robot_cfg.joint_dof  # e.g. 6

        # PID gains
        gain = arx5.Gain(self.dof)
        gain.kd()[:] = 0.01
        self.ctrl.set_gain(gain)

        self.ctrl.reset_to_home()

        # Master-slave mapping parameters (assuming servo 0-5 corresponds to joints 0-5)
        self.scale = [1.0] * self.dof
        self.offset_rad = [0.0] * self.dof

        # Gripper channel
        self.gripper_index = 6
        self.gripper_min_deg = -10.0
        self.gripper_max_deg = 30

        # === Velocity limiting parameters (can be adjusted per joint) ===
        # Maximum joint velocity (rad/s), default uniform upper limit (about 69 deg/s)
        self.max_joint_vel = np.array([1.2] * self.dof, dtype=np.float64)
        self.max_joint_vel[3:] = np.array([2] * (self.dof - 3), dtype=np.float64)
        # Maximum gripper "normalized velocity" (/s), maximum change per second in 0~1 range
        self.max_gripper_vel = 2.0

        # Optional: velocity feedforward switch and ratio (refer to your example)
        self.use_vel_feedforward = False
        self.vel_ff_gain = 0.3

        # Velocity limiting state (target sent in previous cycle)
        self._inited_cmd = False
        self._last_cmd_pos = np.zeros(self.dof, dtype=np.float64)
        self._last_cmd_grip = 0.0

    def _deg_to_rad_mapped(self, master_angles_deg):
        """Master arm angle (degrees) -> Slave arm joint angle (radians)"""
        joints_rad = np.zeros(self.dof, dtype=np.float64)
        for j in range(self.dof):
            joints_rad[j] = np.deg2rad(master_angles_deg[j]) * self.scale[j] + self.offset_rad[j]
        
        joints_rad[4], joints_rad[5] = -joints_rad[5], joints_rad[4] #swap j5 and j6
        # If you find that the direction of the robot arm is opposite to what you expected when controlling it
        # you can modify it here:
        # Example:
        # joints_rad[3] = -joints_rad[3]

        return joints_rad

    def _map_gripper(self, master_angles_deg):
        grip_deg = master_angles_deg[self.gripper_index]
        grip_norm = (grip_deg - self.gripper_min_deg) / max(1e-6, self.gripper_max_deg - self.gripper_min_deg)
        return float(np.clip(grip_norm, 0.0, 1.0))

    def send_cmd(self, master_angles_deg):
        # Desired pose
        desired_pos = self._deg_to_rad_mapped(master_angles_deg)
        desired_grip = self._map_gripper(master_angles_deg)

        dt = float(self.ctrl_cfg.controller_dt)

        # Initialize: first frame directly align
        if not self._inited_cmd:
            self._last_cmd_pos[:] = desired_pos
            self._last_cmd_grip = desired_grip
            self._inited_cmd = True

        # === Joint velocity limiting: limit maximum step per cycle ===
        max_step = self.max_joint_vel * dt                     # Maximum allowed displacement per cycle
        delta = desired_pos - self._last_cmd_pos
        delta_clipped = np.clip(delta, -max_step, max_step)
        cmd_pos = self._last_cmd_pos + delta_clipped

        # === Gripper velocity limiting ===
        grip_delta = desired_grip - self._last_cmd_grip
        grip_step = self.max_gripper_vel * dt
        grip_cmd = self._last_cmd_grip + float(np.clip(grip_delta, -grip_step, grip_step))

        # Organize command
        js = arx5.JointState(self.dof)

        js.pos()[:] = cmd_pos
        js.gripper_pos = grip_cmd

        # Optional: velocity feedforward (based on displacement after velocity limiting)
        if self.use_vel_feedforward and hasattr(js, "vel"):
            safe_vel = delta_clipped / max(dt, 1e-6)
            js.vel()[:] = self.vel_ff_gain * safe_vel

        # Send command
        self.ctrl.set_joint_cmd(js)

        # Update state
        self._last_cmd_pos[:] = cmd_pos
        self._last_cmd_grip = grip_cmd


# ====== Main Program ======
if __name__ == "__main__":
    try:
        # Create master arm reader
        servo_reader = ServoReader(port="/dev/ttyUSB0", baudrate=115200)

        # Create slave arm controller (with velocity limiting)
        teleop = ArxTeleop(model="X5", interface="can0")

        # Start reading thread
        t_reader = threading.Thread(target=servo_reader.read_loop, kwargs={"hz": 100}, daemon=True)
        t_reader.start()

        print("[Main] Teleoperation started, press Ctrl+C to exit.")
        dt = teleop.ctrl_cfg.controller_dt
        while True:
            master_angles = servo_reader.get_angles()
            teleop.send_cmd(master_angles)
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted, robot arm returning to home...")
        teleop.ctrl.reset_to_home()
        print("[Main] Exited.")
