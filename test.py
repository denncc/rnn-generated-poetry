import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.autograd import Variable
import sys

from model import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(temperature = 1):  
    ########### Hyperparameters ###########
    hidden_size = 512   # size of hidden state
    seq_len = 40       # length of LSTM sequence
    num_layers = 3      # num of layers in LSTM layer stack
    lr = 0.002          # learning rate
    epochs = 10        # max number of epochs
    op_seq_len = 1000    # total num of characters in output test sequence
    load_chk = False    # load weights from save_path directory to continue training
    save_path = "./preTrained/CharRNN_shakespeare.pth"
    data_path = "./poem_data/shakespeare.txt"
    #######################################
    # load the text file
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")
        
    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    
    # data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    print(len(char_to_ix))
    
    # model instance
    rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers).to(device)
    
    # load checkpoint if True
    rnn.load_state_dict(torch.load(save_path))
    print("Model loaded successfully !!")
    print("----------------------------------------")
    data_ptr = 0
    hidden_state = None
    
    # initialize test input
    input_seq = data[10955:10955 + 40]
    
    # compute the last hidden state of the sequence
    _, hidden_state = rnn(input_seq, hidden_state)
    
    # next element is the input to rnn
    input_seq = data[10955 + 40 : 10955 + 41]
    
    print("Original text:\n Shall I compare thee to a summer's day? \n")
    print("Predict: ")
    while True:
        # forward pass
        output, hidden_state = rnn(input_seq, hidden_state)

        # construct categorical distribution and sample a character controlled by temperature
        output = F.softmax(torch.squeeze(output) / temperature, dim=0)
        dist = Categorical(output)
        index = dist.sample()
        
        # print the sampled character
        print(ix_to_char[index.item()], end='')

        # next input is current output
        input_seq[0][0] = index.item()
        data_ptr += 1

        if data_ptr > op_seq_len:
            break

    print("\n----------------------------------------")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        test()
    if len(sys.argv) == 3 and sys.argv[1] == "-t":
        test(temperature=int(sys.argv[2]))
    else:
        raise ValueError("Invalid flag specified")