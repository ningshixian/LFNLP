import socket


def _is_available(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(("localhost", port)) == 0:
        print("Port %d is open" % port)
        return True
    return False

