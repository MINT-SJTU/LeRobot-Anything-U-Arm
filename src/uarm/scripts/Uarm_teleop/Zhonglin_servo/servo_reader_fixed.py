#!/usr/bin/env python3
import rospy
import serial
import time
import re
import numpy as np
from std_msgs.msg import Float64MultiArray

class ServoReaderNode:
    def __init__(self):
        rospy.init_node("servo_reader_node")
        self.pub = rospy.Publisher('/servo_angles', Float64MultiArray, queue_size=10)
        self.rate = rospy.Rate(50)
        self.SERIAL_PORT = '/dev/ttyUSB0'

        self.BAUDRATE = rospy.get_param("~baudrate", 115200)
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.1)
        rospy.loginfo("Serial port opened")

        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7
        self.valid_servos = [0, 1, 2]  # Only use servos 0-2
        self._init_servos()

    def send_command(self, cmd):
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        return self.ser.read_all().decode('ascii', errors='ignore')

    def pwm_to_angle(response_str, servo_num, pwm_min=500, pwm_max=2500, angle_range=270):
        pattern = f"#{servo_num:03d}P(\\d{{4}})"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle

    def _init_servos(self):
        self.send_command('#000PVER!')
        for i in self.valid_servos:  # Only initialize valid servos
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip(), i)
            self.zero_angles[i] = angle if angle is not None else 0.0
        rospy.loginfo("Servo initial angle calibration completed")

    def run(self):
        angle_offset = [0.0] * 7  # Currently published angles
        target_angle_offset = [0.0] * 7  # Target angle for each servo
        num_interp = 5  # Interpolation steps
        step_size = 1  # Minimum change threshold

        while not rospy.is_shutdown():
            for i in self.valid_servos:  # Only read valid servos
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip(), i)
                if angle is not None:
                    new_angle = angle - self.zero_angles[i]
                    if abs(new_angle - target_angle_offset[i]) > step_size:
                        target_angle_offset[i] = new_angle
                else:
                    rospy.logwarn(f"Servo {i} response abnormal: {response.strip()}")

            # Interpolate towards target angles
            for step in range(num_interp):
                for i in range(7):
                    delta = target_angle_offset[i] - angle_offset[i]
                    angle_offset[i] += delta * 0.2  # Lazy interpolation, coefficient < 1 for adjustable smoothness
                self.pub.publish(Float64MultiArray(data=angle_offset))
                self.rate.sleep()

if __name__ == '__main__':
    try:
        node = ServoReaderNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 