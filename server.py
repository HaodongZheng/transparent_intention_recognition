from semantic_parsing_services import extract_intent_and_semantic_tags_from_result, get_single_result, initialize_through_recalculating, method_evaluation_parsing_based_CKY_MED, semantic_grammar_parsing_general_idf
import socket
import threading
import json
import multiprocessing as mp

PORT = 5050
HEADER = 64
SERVER = socket.gethostbyname(socket.gethostname())
print(SERVER)
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)


def handle_client(conn, addr):
    print("[NEW CONNNECTION] {0} connected. ".format(addr))
    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False 
                break   
            print(f"[{addr}] {msg}") 
            # here what you need to add is a function that processed the msg received from the user, and send back the semantic parsing result
            # including the intent and semantic tags as well as the best match pattern to client.
            # you will need to serialize everything into json string and deserialize it in the C sharp program.
            result = semantic_grammar_parsing_general_idf(
                msg,
                lowest_grammar_idf,
                vocab_document,
                idf_calculated_from_grammar,
                grammar_pieces_dict,
                vocab,
                states,
                A,
                B,
                intent_taglist_dict,
                pos_dict_for_grammar_terminal,
                insert_discount_factor=1.0,
                verbose=True,
                beam_size=2,
                not_in_doc_factor=1.0,
                single_word_beam_size=2,
                relative_matching_threshold=0.20,
                maximum_length_difference=4,
                force_intent_matching=True,
                real_threshold=0.8,
            )

            match_intent_list, backtrace_dict = extract_intent_and_semantic_tags_from_result(
                result)

            backtrace_dict_string = json.dumps(backtrace_dict)  # data serialized
            match_intent_list_string = json.dumps(match_intent_list)
            print("Dynamic Information: ", backtrace_dict)
            print("Intent: ",match_intent_list)
            conn.send(backtrace_dict_string.encode(FORMAT))
            conn.send(match_intent_list_string.encode(FORMAT))
    conn.close()




def start():
    server.listen()
    print("[LISTENING] Server is listening on {0}".format(SERVER))
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount()-1}")

# you will need a new function here to extract all the necessary tags and intent




grammar_pieces_dict, idf_calculated_from_grammar, normalized_idf_dict, states, A, B, vocab, vocab_document, intent_taglist_dict, pos_dict_for_grammar_terminal, lowest_grammar_idf = initialize_through_recalculating()
test_sentence = 'I now really want to know the entry requirements of a program'
result = semantic_grammar_parsing_general_idf(
    test_sentence,
    lowest_grammar_idf,
    vocab_document,
    idf_calculated_from_grammar,
    grammar_pieces_dict,
    vocab,
    states,
    A,
    B,
    intent_taglist_dict,
    pos_dict_for_grammar_terminal,
    verbose=False,
    single_word_beam_size=2,
    beam_size=2,
    insert_discount_factor=1.0,
    maximum_length_difference=4,
    not_in_doc_factor=0.5,
    relative_matching_threshold=0.15,
    force_intent_matching=False,
    real_threshold=0.9,
)


print(result)
backtrace_dict, match_intent_list = extract_intent_and_semantic_tags_from_result(result)
print(backtrace_dict)
print(match_intent_list)
print(grammar_pieces_dict["[RequestInfo] "])
print("[STARTING] server is starting...")
print(idf_calculated_from_grammar["[the]"],lowest_grammar_idf)

start()
