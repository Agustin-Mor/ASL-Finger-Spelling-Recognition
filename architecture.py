class FullyDenseNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_sizes):
        """
        Initializes the FullyDenseNetwork.
        
        :param input_size: Number of features in the input.
        :param output_size: Number of output neurons (e.g., number of classes or regression output size).
        :param hidden_layer_sizes: List of integers where each integer represents the number of neurons in a hidden layer.
        """

        super(FullyDenseNetwork, self).__init__()
        
        # Create a list of layers
        layers = []
        
        # Input layer
        in_features = input_size
        
        # Add hidden layers
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, output_size))        

        # Create the sequential model
        self.network = nn.Sequential(*layers)



    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        :param x: Input tensor.
        :return: Output tensor.
        """

        result = self.network(x)

        return result