import socket
from config import ROBOT_IP, PORT

class Meca500:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ROBOT_IP, PORT))
        print("Robot connected")

    def send(self, cmd):
        self.sock.send((cmd + "\n").encode())

    def move_relative(self, dx, dy, dz):
        cmd = f"MoveLinRel({dx},{dy},{dz},0,0,0)"
        self.send(cmd)

    def close(self):
        self.sock.close()