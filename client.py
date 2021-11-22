import socket
import sys
import json

PORT = 5050
HEADER = 64
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER,PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    msg_encoded = msg.encode(FORMAT)
    msg_length = len(msg_encoded)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER-len(send_length))
    client.send(send_length)
    client.send(msg_encoded)
    backtrace_dict_string = client.recv(2048).decode(FORMAT).replace(
        "{", "").replace("}", "").replace("\"","")
    match_intent_list_string = client.recv(2048).decode(FORMAT).replace(
        "[", "").replace("]", "").replace("\"", "")
    backtrace_dict_string = backtrace_dict_string
    
    # this is a string, u need to convert it into a list, which is a buffer for the agent to process.
    print(backtrace_dict_string)
    print("||")
    # this is a string, u need to convert it into tag_value unit in the c sharp program, and throw it into the memory.
    print(match_intent_list_string)


# here you need to replace input() by arguments from the command line
#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))
arguments = sys.argv
if len(arguments)>2:
    print("Error: Too many arguments, only one argument (a sentence string) is allowed")
    sys.exit() 
else:
    input_message = arguments[1]
    send(input_message)
# If use input() uncommment the following lines
#while True:
#    input_message = input()
#    if input_message == "exit":
#        break
#    send(input_message)

send(DISCONNECT_MESSAGE)
sys.exit()
