import socket
import time

host = '127.0.0.1'
port = 65000
buff = 1024

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((host, port))

while True:
    server.listen(1)
    print('Waiting for connection')
    client, addr = server.accept()
    print('Established connection')
    print(client)
    print(addr)

    while True:
        try:
            #msg = client.recv(buff)
            #print('Received msg: %s' % msg)
            client.send('Hello,World'.encode("utf-8"))
            time.sleep(1)
        except socket.error:
            client.close()
            break
