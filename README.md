# rnn-generated-poetry
Character Based LSTM model that enables poetry generation with user specifiable temperature attribute.

# To run this project

## Testing
After cloning, if you want to use the pretrained model, run directly the test.py by:

    python test.py
    
If you want to specified the temperature, the hyperparameter of LSTMs (and neural networks generally) used to control the randomness of predictions by scaling the logits before applying softmax, you can:

    python test.py -t TEMPERATURE
    
## Training
If you want to re-train the model on shakespeare data, you can run the following:

    python train.py
    
Also, if you want to run the model on the custom data, you can run the following:

    python train.py -d DATADIR
    
# Acknowledgement
I want to thank specially to nikhilbarhate99. Without his inspiration this repo is unable to finish.
