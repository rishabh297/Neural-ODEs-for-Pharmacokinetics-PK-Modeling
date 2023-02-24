
import torch
import torch.nn as nn
from args import args
# from torch.nn.modules.rnn import GRU, LSTM, RNN
import utils

#ODEFunc is a subclass that inherits from superclass nn.Module (base class for neural networks; the main inherited method that subclasses of nn.Module need to override is forward())
class ODEFunc(nn.Module):
    
    #initializes neural network with desired dimensions
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()

        #nn.Sequential is a method that allows the creation of layers in the neural network. The method itself acts as a "container" for the layers/modules inside the network.
        self.net = nn.Sequential(
            #layers in a neural network are nothing but a series of linear transformations on our input matrix (y = xAt + b). nn.Linear forms a "linear layer" which applies learnable weights (x) and biases (b) to our input data (At). The dimensionality of data often changes hence the allowance of input_dim and hidden-dim. 
            nn.Linear(input_dim, hidden_dim),
            #nn.SeLu is an activation function. Activation functions determine the weighted "importance" of features in the input data. This adds non-linearity to our model which allows it to be more complex than simple linear regression. SELU specifically allows for self-normalizing neural nets and tackles the vanishing gradient problem.
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # this for loop interates through every linear layer (set of layers is returned by calling self.net.modulesi) in the neural network we defined above and initializes weights/biases of input feature
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                #if a module (layer) in our network is a linear layer and not an activation function, this randomly initializes our input tensor of weights of each layer (m.weights) with values sampled from a Gaussian distribution with mean 0 and SD 0.001. This mitigates the vanishing/exploding gradient problem.
                nn.init.normal_(m.weight, mean=0, std=0.001)
                # if a module is a linear layer, then the input tensor (tensor containing biases of the layer) are all initialized to 0.5
                nn.init.constant_(m.bias, val=0.5)
                
     #feeds our input data through our network (self.net)
    def forward(self, t, x):
        # print(x)
        return self.net(x)


#defines an encoder network following variational autoencoder concept; this means that the inputs are mapped to a distribution rather than a deterministic outcome. 
class Encoder(nn.Module):

    #initializes attributes of instances of encoder
    def __init__(self, input_dim, output_dim, hidden_dim, device=torch.device("cpu")):
        super(Encoder, self).__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

    #Sets up a sequential layers for network. Encoders are built by repeatedly "stacking" RNNs (in this case GRU) as each layer. Each layer (GRU) analyzes a single element of the input sequence, "retains/encodes" important info about that element, and propogates forward. 
        self.hiddens_to_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            #ReLU activation function is similar to SELU except it takes on binary values and can result in dead neurons, causing them to not be used for predicing outputs from features.
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        #utils function that serves similar purpose as for loop in ODEFunc class. However, this function initializes biases as a 0 constant while weights are sampled from gaussian dist.
        utils.init_network_weights(self.hiddens_to_output, std=0.001)

        #nn.GRU applies a "pre-built" GRU to a given input; the GRU "scans" through the time series data in reverse and encodes the relevant data into a 12-element array. This array is fed into the ODEFunc network to define the mean and standard deviation of the latent state distributions which z_t0 is sampled from. 
        # self.rnn = nn.RNN(self.input_dim, self.hidden_dim, nonlinearity="relu").to(device)
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim).to(device)

    #defines forward pass of encoder
    def forward(self, data):
        #permutes data to make necessary dimensional changes
        data = data.permute(1, 0, 2)
        #reverses data to allow GRU to scan through time series data in reverse fashion (why?)
        data = utils.reverse(data)
        #sends input data through GRU
        output_rnn, _ = self.rnn(data)
        #print(output_rnn)
        #takes in the data scanned in reverse (done by GRU) and feeds through 
        outputs = self.hiddens_to_output(output_rnn[-1])
        #print(outputs)
        
        return outputs

#initializes and defines a decoder network that takes in the sequence of z_t's outputted by the ODEFunc network. It then generates the predictions from the output of the ODE solver and the first dosing observations.
class Classifier(nn.Module):

    #init method creates another set of sequential modules with 1 fully connected layer and 32 hidden units
    def __init__(self, latent_dim, output_dim):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 20, 32),
            nn.SELU(),
            nn.Linear(32, output_dim)
        )
        
        #follows same weight and bias initialization protocol as the Encoder class
        utils.init_network_weights(self.net, std=0.001)

    #defines forward pass where z is the sequence of z_t's generated by the output of ODEFunc and cmax_time refers to the dosing information.
    def forward(self, z, cmax_time):
        #repeates dosing information along given dimensions to match up with z
        cmax_time = cmax_time.repeat(z.size(0), 1, 1)
        #joins dosing info with sequence of z_t's and feeds in as input to decoder
        z = torch.cat([z, cmax_time], 2)
        return self.net(z)
