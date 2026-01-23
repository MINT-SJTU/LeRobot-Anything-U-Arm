import serial
import time 
import numpy as np
import re

init_qpos = np.array([14.1, -8, -24.7, 196.9, 62.3, -8.8, 0.0])
init_qpos = np.radians(init_qpos)


# Set serial port parameters
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200

def send_command(ser, cmd):
    ser.write(cmd.encode('ascii'))
    time.sleep(0.01)
    response = ser.read_all()
    # print(f"Raw response (binary): {response}")  # Print raw binary response
    return response.decode('ascii', errors='ignore')

def pwm_to_angle(response_str, servo_num, pwm_min=500, pwm_max=2500, angle_range=270):
    pattern = f"#{servo_num:03d}P(\\d{{4}})"
    match = re.search(pattern, response_str)
    if not match:
        return None
    pwm_val = int(match.group(1))
    pwm_span = pwm_max - pwm_min
    angle = (pwm_val - pwm_min) / pwm_span * angle_range
    return angle

def angle_to_gripper(angle_deg, angle_range=270, pos_min=50, pos_max=730):
    """
    Map servo angle (degrees) to gripper position.

    Parameters:
    - angle_deg: Servo angle in degrees
    - angle_range: Maximum servo angle (default 270°)
    - pos_min: Gripper closed position (default 50)
    - pos_max: Gripper open position (default 730)

    Returns:
    - gripper position (integer)
    """
    ratio = (angle_deg / angle_range) * 3
    position = pos_min + (pos_max - pos_min) * ratio
    return int(np.clip(position, pos_min, pos_max))

def main():
    index=2
    arm_pos = [0.0] * 7
    angle_pos = [0.0] * 7
    zero_angles = [0.0] * 7
    with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01) as ser:
        print("Serial port opened")

        response = send_command(ser, f'#00{index}PVER!')
        
        for i in range(7):
            cmd = f'#00{i}PULK!'
            response = send_command(ser, cmd)
            print(f"Servo {i} torque released: {response.strip()}")

        print(f"Version response: {response.strip()}")
        while True:
            for i in range(7):
                cmd = f'#00{i}PRAD!'
                response = send_command(ser, cmd)
                angle = pwm_to_angle(response.strip(), i)
                angle_pos[i] = angle
                if angle is not None:
                    angle_offset = angle - zero_angles[i]+init_qpos[i]
                    angle_rad = np.radians(angle_offset)
                    arm_pos[i] = angle_rad
            print(angle_pos)
    
if __name__ == "__main__":
    main()
