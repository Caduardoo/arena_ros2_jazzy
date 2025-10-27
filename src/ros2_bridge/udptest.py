import socket
import time
import numpy as np

TARGET_IP = "192.168.0.133"
TARGET_PORT = 25568     

def send_udp_messages_in_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP socket
    counter = 0

    try:
        while True:
            message = np.append([1,1,1], [counter, counter]).astype(np.float32)
            message_bytes = message.tobytes()
            sock.sendto(message_bytes, (TARGET_IP, TARGET_PORT))
            print(f"Sent: {message}")
            counter += 1
            time.sleep(1) # Send a message every 1 second
    except KeyboardInterrupt:
        print("UDP sender stopped by user.")
    finally:
        sock.close()

if __name__ == "__main__":
    send_udp_messages_in_loop()