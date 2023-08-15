import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q3_schedule import LinearExploration, LinearSchedule
from q4_linear_torch import Linear
import logging


from configs.q5_nature import config


class NatureQN(Linear):
    """
    Implementing DQN that will solve MinAtar's environments.

    Model configuration can be found in the assignment PDF, section 4a.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?



        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
            3. To calculate the size of the input to the first linear layer, you
               can use online tools that calculate the output size of a
               convolutional layer (e.g. https://madebyollin.github.io/convnet-calculator/)
        """
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.num_actions()
        
        ##############################################################
        ################ YOUR CODE HERE - 20-30 lines ################
        """
        From assignment,
         • One convolution layer with 16 output channels, a kernel size of 3, stride 1, and no padding.
         • A ReLU activation.
         • A dense layer with 128 hidden units.
         • Another ReLU activation.
         • The final output layer.
        """
        conv1_output_channel = 16
        channels = n_channels*self.config.state_history
        input_shape = (img_height, img_width)
        conv1 = nn.Conv2d(in_channels=channels, out_channels=conv1_output_channel, kernel_size=3, stride=1)
        reLU1 = nn.ReLU()
        flatten1 = nn.Flatten()
        hidden1 = nn.Linear(in_features=(img_height-2)*(img_width-2)*conv1_output_channel, out_features=128)
        reLU2 = nn.ReLU()
        output1 = nn.Linear(in_features=128, out_features=num_actions)
        self.q_network = nn.Sequential(
            conv1,
            reLU1,
            flatten1,
            hidden1,
            reLU2,
            output1
        )
        self.target_network = nn.Sequential(
            conv1,
            reLU1,
            flatten1,
            hidden1,
            reLU2,
            output1
        )
        self.target_network.load_state_dict(self.q_network.state_dict())     
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        # network input shape should be : (batch_size, nchannels x config.state_history, img height, img width)
        state_permuted = state.permute(0,3,1,2)
        network_to_use = self.target_network if network == "target_network" else self.q_network
        out = network_to_use(state_permuted)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
