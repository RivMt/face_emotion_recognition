import socket

import config_loader as config

server_ip = config.get_target_address()
server_port = config.get_port()
server_addr_port = (server_ip, server_port)

def send(message):
    bytes_to_send = str.encode(message)
    udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_client_socket.sendto(bytes_to_send, server_addr_port)