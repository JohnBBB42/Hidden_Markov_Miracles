from hidden_markov_neural_network.model import Model
from hidden_markov_neural_network.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
