from torch import nn, optim

# Batch
BATCH_SIZE = 50

# Model
INPUT_SIZE = 63
HIDDEN_LAYERS = [126, 21]
OUTPUT_SIZE = 24

# Loss and optimizer functions
LOSS_FUNCTION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam

# Training
EPOCHS = 200
LEARNING_RATE = 0.001