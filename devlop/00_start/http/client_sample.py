import socket
import time

host = '127.0.0.1'
port = 65000
buff = 1024

def make_connection(_host, _port):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.connect((_host, _port))
            print('connected')
            return sock
        except socket.error as e:
            print('failed to connect, try reconnect')
            time.sleep(1)

socket = make_connection(host, port)

try:
    #print('send')
    #socket.send('GET'.encode("utf-8"))
    msg = socket.recv(buff)
    print('Client received: %s' % msg)
    time.sleep(1)
except socket.error as e:
    print('connection lost, try reconnect')
    socket = make_connection(host, port)
