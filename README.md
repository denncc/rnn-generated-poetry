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
    
# Results
After 40 epochs of training, the cross entropy training loss is reduced to 0.55. Here are some of the test poems generations.

<p align="center">
158<br>
Whod me do I hold my distill doth her own,<br>
That leaves be repent her featered born,<br>
Compare the old virtue rehearse of thy di?.<br>
  If thy proud hath like thy love doth lies,<br>
To mak'st then fire hate on you to my name'.
</p>

# Acknowledgement
I want to thank specially to nikhilbarhate99. Without his inspiration this repo is unable to finish.
