
import torch
import random
from memory import Memory, ReFrame
from torch import nn, optim
from network import DeepNetWork

class DDqn(object):

    path = './model.pth'

    actions = 2
    gamma = 0.99

    init_epsilon = 0.5
    final_epsilon = 0.05

    batch_size = 32
    decay_rate = 0.05
    learning_rate = 1e-6
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    save_interval = 1000
    replace_interval = 500

    def __init__(self, mode):


        self.train = mode
        self.time_step = 0
        self.memory = Memory()
        self.epsilon = self.init_epsilon
        self.build()
        self.reframe = ReFrame()

    def build(self):

        self.loss = nn.MSELoss()
        self.eval_net = DeepNetWork()
        self.target_net = DeepNetWork()
        self._load_model()
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=1e-6, weight_decay=0.9, momentum=0.95)

    def init_weights(self, model):

        if type(model) == nn.Conv2d or type(model) == nn.Linear:
            model.weight.data.normal_(0.0, 0.01)
        if type(model) == nn.Conv2d:
            model.bias.data.fill_(0.01)

    def _load_model(self):

        try:
            checkpoint = torch.load(self.path)
            self.epsilon = checkpoint['epsilon']
            self.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

            if self.train:
                self.eval_net.train()
                self.target_net.train()
            else:
                self.eval_net.eval()
                self.target_net.eval()

            self.eval_net = self.eval_net.to(self.device)
            self.target_net = self.target_net.to(self.device)
            print('successfully load trained model')

        except Exception:

            self.eval_net = self.eval_net.to(self.device)
            self.target_net = self.target_net.to(self.device)
            self.eval_net.apply(self.init_weights)
            print('could not find trained model')

    def _save_model(self, epoch):

        if epoch % self.save_interval == 0:

            torch.save({'epsilon': self.epsilon,
                        'eval_net_state_dict': self.eval_net.state_dict(),
                        'target_net_state_dict': self.target_net.state_dict()
                        }, self.path)

            print('save model')


if __name__ == '__main__':

    agent = DDqn()
