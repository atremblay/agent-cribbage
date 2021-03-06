from .register import register
from .value_function import ValueFunction
from gym_cribbage.envs.cribbage_env import Stack, RANKS, SUITS
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@register
class LSTM(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=52,
            hidden_size=104,
            num_layers=2,
            batch_first=True
        )

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(104+52+13, 52),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(52, 1),
        )
        # Before applying weights
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x, discarded, hand):

        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]  # Only keeps last value of sequence
        out = self.clf(torch.cat((out, discarded, hand), dim=1))
        return out

    def get_after_state(self, state, env):
        choices = np.expand_dims(np.array([c.state for c in state.hand]), axis=1)

        # If has card on the table
        if len(env.table) != 0:
            table_cards = np.expand_dims(np.array([card.state for card in env.table]), 0)
            table_cards_repeated = np.repeat(table_cards, len(state.hand), axis=0)
            choices = np.append(table_cards_repeated, choices, axis=1)

        hand = np.array([state.hand.remove(c).compact_state[1].sum(axis=1) for c in state.hand])

        # Store state for data generation.
        after_state = [choices.astype('float32'),
                       np.repeat(np.expand_dims(env.discarded.state, axis=0), len(state.hand), axis=0).astype(
                           'float32'),
                       hand]

        return after_state


@register
class ConvLstm(ValueFunction):
    def __init__(self, split_discarded=False):
        """
        Simple LSTM that only takes the cards face up. We do not include
        face down here, thereby reducing the number of possible states.
        """
        super().__init__()
        self.split_discarded = split_discarded
        if not self.split_discarded:
            self.forward_arg_size -= 1
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=15,
                kernel_size=2,
                stride=1,
                bias=False
            )
        )

        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 2, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=15,
                kernel_size=3,
                stride=1,
                bias=False
            )
        )

        self.conv4 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 3, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=15,
                kernel_size=4,
                stride=1,
                bias=False
            )
        )

        self.lstm = nn.LSTM(
            input_size=45 + 13,
            hidden_size=104,
            num_layers=2,
            batch_first=True
        )

        # Logistic Regression
        num_dim = 104  # LSTM outputs
        num_dim += 13  # Hand
        num_dim += 13  # Discarded
        if self.split_discarded:
            num_dim += 13  # Discarded cards are separated between players

        self.clf = nn.Sequential(
            nn.Linear(num_dim, num_dim),
            nn.LeakyReLU(),
            #nn.Dropout(),
            nn.Linear(num_dim, 52),
            nn.LeakyReLU(),
            #nn.Dropout(),
            nn.Linear(52, 1),
        )
        # Before applying weights
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x, hand, discarded, discarded_other=None):
        if self.split_discarded and discarded_other is None:
            raise Exception(
                "`discarded_other` should be provided since you set "
                "`split_discarded=True`"
            )

        lengths = x.sum(dim=1).sum(dim=1).cpu().numpy()
        sorted_idx = torch.tensor(np.argsort(lengths)[::-1].copy())

        if len(lengths) > 1:
            sorted_len = lengths[sorted_idx]
        else:
            sorted_len = lengths

        out2 = self.conv2(x.unsqueeze(1)).sum(dim=-1)
        out3 = self.conv3(x.unsqueeze(1)).sum(dim=-1)
        out4 = self.conv4(x.unsqueeze(1)).sum(dim=-1)
        out = torch.cat(
            [
                out2.transpose(1, 2),
                out3.transpose(1, 2),
                out4.transpose(1, 2),
                x
            ],
            dim=2
        )

        out = out[sorted_idx]
        packed_out = pack_padded_sequence(
            out,
            sorted_len,
            batch_first=True
        )

        packed_out, (hidden, cell) = self.lstm(packed_out)
        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        # Extract the outputs for the last timestep of each example
        idx = (torch.LongTensor(sorted_len) - 1).view(-1, 1)
        if x.is_cuda:
            idx = idx.cuda(device=x.device)
        # This duplicates the last time step for every batch element through
        # the hidden dimension
        idx = idx.expand(out.shape[0], out.shape[2])

        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)

        # Shape: (batch_size, rnn_hidden_dim)
        last_output = out.gather(
            time_dimension, idx
        ).squeeze(time_dimension)
        last_output = last_output[np.argsort(sorted_idx)]

        # out = out[:, -1, :][np.argsort(sorted_idx)]  # Only keeps last value of sequence
        to_cat = [last_output, hand, discarded]
        if self.split_discarded:
            to_cat.append(discarded_other)
        out = self.clf(torch.cat(to_cat, dim=1))
        return out

    @staticmethod
    def stack_to_numpy(stacks):
        if isinstance(stacks, Stack):
            stacks = [stacks]

        batch_size = len(stacks)
        x = np.zeros((batch_size, 8, 13), dtype=np.float32)

        for i, stack in enumerate(stacks):
            for j, card in enumerate(stack):
                rank = RANKS.index(card.rank)
                x[i, j, rank] = 1
        return x

    def get_after_state(self, state, env):
        choices = [env.table.add(c) for c in state.hand]

        choices = self.stack_to_numpy(choices)

        hand = np.array([state.hand.remove(c).compact_state[1].sum(axis=1) for c in state.hand])

        # Store state for data generation.
        after_state = [choices, hand]
        cards_have_no_player = [card.player is None for card in env.discarded]
        if any(cards_have_no_player) and self.split_discarded:
            raise Exception("At least one card had no player.")

        if self.split_discarded:
            discarded_player_0 = np.zeros((1, 13), dtype=np.float32)
            discarded_player_1 = np.zeros((1, 13), dtype=np.float32)
            for card in env.discarded:
                if card.player == 0:
                    discarded_player_0 += card.compact_state[1]
                else:
                    discarded_player_1 += card.compact_state[1]

            # The player playing knows what he put in the crib. Add this
            # information to the discarded stack
            for card in env.crib:
                if card.player != env.player:
                    continue
                if card.player == 0:
                    discarded_player_0 += card.compact_state[1]
                elif card.player == 1:
                    discarded_player_1 += card.compact_state[1]

            after_state.append(np.repeat(discarded_player_0, len(state.hand), axis=0))
            after_state.append(np.repeat(discarded_player_1, len(state.hand), axis=0))
        else:
            discarded = np.zeros((1, 13), dtype=np.float32)
            for card in env.discarded:
                discarded += card.compact_state[1]
            after_state.append(np.repeat(discarded, len(state.hand), axis=0))

        return after_state
