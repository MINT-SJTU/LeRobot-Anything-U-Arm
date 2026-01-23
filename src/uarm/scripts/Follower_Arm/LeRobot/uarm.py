import serial
import time
import re
import numpy as np

class ServoReader:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        print("[INFO] Serial port opened:", port)
        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7
        self._init_servos()

    def send_command(self, cmd):
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        return self.ser.read_all().decode('ascii', errors='ignore')

    def pwm_to_angle(response_str, servo_num, pwm_min=500, pwm_max=2500, angle_range=270):
        """Convert PWM value to angle"""
        pattern = f"#{servo_num:03d}P(\\d{{4}})"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        angle = (pwm_val - pwm_min) / (pwm_max - pwm_min) * angle_range
        return angle

    def _init_servos(self):
        """Initialize and record the zero angle of each servo"""
        self.send_command('#000PVER!')
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip(), i)
            self.zero_angles[i] = angle if angle is not None else 0.0
        print("[INFO] Servo zero calibration completed.")
        print("Zero Angles:", np.round(self.zero_angles, 2))

    def get_action_offset(self):
        """
        Get the angle offset (action offset) of each servo relative to the zero point
        Returns: list[float], length is 7
        """
        angle_offset = [0.0] * 7
        for i in range(7):
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip(), i)
            if angle is not None:
                angle_offset[i] = angle - self.zero_angles[i]
            else:
                print(f"[WARN] Servo {i} response error: {response.strip()}")
        return angle_offset

    def run(self):
        """Loop to read and print angles (optional)"""
        angle_offset = [0.0] * 7
        target_angle_offset = [0.0] * 7
        num_interp = 5
        step_size = 1

        while True:
            for i in range(7):
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip(), i)
                if angle is not None:
                    new_angle = angle - self.zero_angles[i]
                    if abs(new_angle - target_angle_offset[i]) > step_size:
                        target_angle_offset[i] = new_angle
                else:
                    print(f"[WARN] Servo {i} response error: {response.strip()}")

            for step in range(num_interp):
                for i in range(7):
                    delta = target_angle_offset[i] - angle_offset[i]
                    angle_offset[i] += delta * 0.2
                print("Current Angles:", np.round(angle_offset, 2))
                time.sleep(0.02)


if __name__ == '__main__':
    reader = ServoReader(port='/dev/ttyUSB0', baudrate=115200)

    print("\n[EXAMPLE] read action offset：")
    offsets = reader.get_action_offset()
    print("Action Offsets:", np.round(offsets, 2))
