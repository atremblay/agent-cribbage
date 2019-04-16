from ..agent.value_function.conv import Conv, stack_to_numpy
from ..agent.value_function.lstm import ConvLstm
from .job import Job
from .register import register
from gym_cribbage.envs.cribbage_env import (
    Stack,
    Card,
    Deck,
    evaluate_cards,
    evaluate_table,
    RANKS,
    SUITS
)
from itertools import combinations
from poutyne.framework import Model
from poutyne.framework.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import os
import torch
import tqdm
from pathlib import Path


@register
class Preheat(Job):
    def __init__(self):
        super().__init__()
        super()._setup_job(__name__, None, None)
        os.makedirs(self['checkpoint_dir'], exist_ok=True)

    def add_argument(self):
        # Add arguments
        self.parser.add_argument("--data_dir", default=None)
        self.parser.add_argument("--epochs", default=5, type=int, help="Number of epochs to wait before stopping the training")
        self.parser.add_argument("--checkpoint_dir", default=f'.', type=Path, help="Path where to save all the files")

        self.parser.add_argument(
            "-b", "--batch_size",
            help="Number of data point to use per batch",
            default=64,
            type=int
        )

        self.parser.add_argument(
            "--patience",
            help="Number of epochs to wait before stopping the training",
            default=5,
            type=int
        )

        self.parser.add_argument(
            "--epsilon",
            help="Minimum improvement on the validation loss to keep training",
            default=0.0001,
            type=float
        )

        self.parser.add_argument(
            "--lr",
            help="Learning Rate",
            default=5e-4,
            type=float
        )

        self.parser.add_argument(
            "--with_dealer",
            help="Include dealer in the state",
            default=False,
            action='store_true'
        )

        self.parser.add_argument(
            "--model",
            help="Type of model to train",
            choices=["conv", "lstm"]
        )

    def job(self):

        if self['model'] == 'conv':
            train_conv(self.args)
        else:
            train_lstm(self.args)


def train_conv(args):
    all_comb = list(combinations(Deck().cards, 4))
    total_comb = len(all_comb)
    X_tarot = np.empty((total_comb, 4, 13, 4), dtype=np.float32)
    y = np.empty((total_comb, 1), dtype=np.float32)
    for i, cards in tqdm.tqdm(enumerate(all_comb), total=total_comb):
        stack = Stack(list(cards))
        tarot = stack_to_numpy(stack)
        X_tarot[i] = tarot
        y[i] = evaluate_cards(stack)

    train_idx, test_idx = train_test_split(
        np.arange(total_comb), train_size=0.8, test_size=0.2
    )
    X_tarot, y = torch.tensor(X_tarot), torch.tensor(y)

    X_train, y_train = X_tarot[train_idx], y[train_idx]
    X_test, y_test = X_tarot[test_idx], y[test_idx]

    conv = Conv(out_channels=15, with_dealer=args.with_dealer)
    print(conv)
    model = Model(
        conv,
        torch.optim.SGD(
            conv.parameters(),
            lr=args.lr,
            momentum=0.9
        ),
        'mse'
    )

    if args.cuda is not None:
        model.cuda(args.cuda)

    if args.with_dealer:
        dealer_train = np.random.random((X_train.shape[0], 1)) > 0.5
        dealer_train = torch.tensor(dealer_train.astype(np.float32))
        X_train = (X_train, dealer_train)

        dealer_test = np.random.random((X_test.shape[0], 1)) > 0.5
        dealer_test = torch.tensor(dealer_test.astype(np.float32))
        X_test = (X_test, dealer_test)

    model.fit(
        X_train, y_train,
        validation_x=X_test,
        validation_y=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            EarlyStopping(
                min_delta=args.epsilon,
                patience=args.patience
            )
        ]
    )

    torch.save(
        {'model_state_dict': [conv.state_dict()], 'epoch': args.epochs},
        os.path.join(args.checkpoint_dir / 'conv.pkl')
    )


def train_lstm(args):
    lstm = ConvLstm()
    model = Model(
        lstm,
        torch.optim.SGD(
            lstm.parameters(),
            lr=args.lr,
            momentum=0.9
        ),
        'mse',
        metrics=['mse']
    )

    if args.cuda is not None:
        model.cuda(args.cuda)

    base_hand1 = Stack(
        cards=[
            Card(RANKS[2], SUITS[0]), Card(RANKS[2], SUITS[1])
        ]
    )
    base_hand2 = Stack(
        cards=[
            Card(RANKS[3], SUITS[0]), Card(RANKS[3], SUITS[1])
        ]
    )

    simluation = 0
    while True:
        stacks = []
        num_samples = args.batch_size * 1000
        values = np.empty((num_samples, 1), dtype=np.float32)
        num_with_value = 0
        i = 0
        while num_with_value < 0.3 * num_samples and len(stacks) < num_samples:
        # for i in range(args.batch_size * 1000):
            deck = Deck()
            stack = Stack()
            sumz = 0
            # Get a stack of random size on the table
            for _ in range(np.random.randint(1, 9)):
                card = deck.deal()
                sumz += card.value
                if sumz <= 31:
                    stack.add_(card)
                else:
                    break

            value = evaluate_table(stack)
            if value > 0:
                num_with_value += 1
            if (num_with_value > 0.3 * num_samples and len(stacks) < num_samples) or num_with_value < 0.3 * num_samples:
                stacks.append(stack)
                values[i, 0] = value
                i += 1

        history = model.fit(
            ConvLstm.stack_to_numpy(stacks), values,
            # epochs=1,
            batch_size=args.batch_size,
            callbacks=[
                EarlyStopping(
                    min_delta=args.epsilon,
                    patience=args.patience,
                    monitor='loss'
                )
            ],
            # verbose=0
        )
        pred = model.predict(
            ConvLstm.stack_to_numpy([base_hand1, base_hand2])
        )
        print(f"Simulation {simluation}, loss {history[-1]['loss']:.6f} {base_hand1} {pred[0, 0]:.6f}  {base_hand2}  {pred[1, 0]:.6f}")
        simluation += 1
        if simluation == args.epochs:
            break

    torch.save(
        {'model_state_dict': [lstm.state_dict()], 'epoch': args.epochs},
        os.path.join(args.checkpoint_dir / 'lstm.pkl')
    )

