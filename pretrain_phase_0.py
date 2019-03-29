from gym_cribbage.envs.cribbage_env import (
    Stack,
    Card,
    Deck,
    evaluate_cards,
    RANKS,
    SUITS
)

from agent.value_function.ffw import FFW
from agent.value_function.conv import Conv, ConvEval
from itertools import combinations
import numpy as np
import torch
import tqdm
from functools import reduce


def summary(model):
    print(model)
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        num_params = reduce(lambda x, y: x * y, p.shape)
        if p.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params
    print("{} trainable parameters".format(trainable_params))
    print("{} non trainable parameters".format(non_trainable_params))


def deal_hand(deck, num_cards=6):
    hand = Stack()
    for _ in range(num_cards):
        hand.add_(deck.deal())
    return hand


def mean_value_of_hand(hand):
    deck = Deck()
    for card in hand:
        deck.remove_(card)

    values = []
    while True:
        starter = deck.deal()
        if starter is None:
            break
        values.append(evaluate_cards(hand, starter))
    return np.array(values).mean()


base_hand = Stack(
    cards=[
        Card(RANKS[3], SUITS[0]),
        Card(RANKS[4], SUITS[1]),
        Card(RANKS[4], SUITS[2]),
        Card(RANKS[10], SUITS[3])
    ]
)
base_hand_ranks, base_hand_tarot = Conv.stack_to_tensor(base_hand)

base_value = mean_value_of_hand(base_hand)
print(f"Mean value of {base_hand}  {base_value:.5f}")

##################
# Full procedure #
##################
BATCH_SIZE = 32
NUM_BATCHES = 100
SAMPLES = BATCH_SIZE * NUM_BATCHES
i = 0
value_function = Conv()
summary(value_function)

correct_choice = np.empty(SAMPLES, dtype=np.int32)
sanity_checks = []
train_ranks = np.empty((SAMPLES, 10), dtype=np.float32)
train_tarot = np.empty((SAMPLES, 4, 13, 4), dtype=np.float32)
target = np.empty(SAMPLES)
# pbar = tqdm.tqdm(total=NUM_BATCHES)
# Alternate phase of exploring environment and training
for training_session in tqdm.trange(NUM_BATCHES):
    # Need to turn the model into evaluation mode to score each combinations
    value_function.eval()

    for episode in range(BATCH_SIZE):
        # Shuffle the deck
        deck = Deck()
        hand = deal_hand(deck)
        starter = deck.deal()

        card_combinations = [list(c) for c in combinations(hand.cards, 4)]
        as_stacks = [Stack(c) for c in card_combinations]
        combinations_scores = [evaluate_cards(s, starter) for s in as_stacks]
        combinations_scores = np.array(combinations_scores)

        # There might be more than one combinations with the max points
        # Keeping track of them all
        max_score = combinations_scores.max()
        best_score_idx = np.where(combinations_scores == max_score)[0]

        # Run all combinations through the model to get values
        ranks, tarot = Conv.stack_to_tensor(as_stacks)

        outputs = value_function(ranks, tarot)
        outputs = outputs.detach().numpy().flatten()

        # Turn the output into probabilities
        probs = np.exp(outputs) / np.exp(outputs).sum()
        # Sample from probs
        idx = np.random.choice(np.arange(len(probs)), p=probs)

        train_ranks[i] = ranks[idx]
        train_tarot[i] = tarot[idx]
        target[i] = combinations_scores[idx]
        correct_choice[i] = idx in best_score_idx
        i += 1

    ##################
    # Train sequence #
    ##################

    optimizer = torch.optim.SGD(
        value_function.parameters(),
        lr=0.0005,
        momentum=0.9
    )
    criterion = torch.nn.MSELoss()

    # After BATCH_SIZE episodes, train on the whole dataset
    running_loss = 0
    train_bar = tqdm.tqdm(leave=True, total=i // BATCH_SIZE)
    for j in range(i // BATCH_SIZE):
        value_function.train(True)
        outputs = value_function(
            torch.tensor(train_ranks[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]),
            torch.tensor(train_tarot[j * BATCH_SIZE: (j + 1) * BATCH_SIZE])
        )
        optimizer.zero_grad()
        loss = criterion(outputs, torch.tensor(target[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]).float())
        loss.backward()
        # Calculate the loss
        running_loss += loss.item()
        optimizer.step()

        value_function.eval()
        sanity_check = value_function(base_hand_ranks, base_hand_tarot).item()
        sanity_checks.append(sanity_check)

        correct_choice = np.array(correct_choice)

        desc = "Loss {:.5f}, Sanity: {:.5f}, base value: {:.5f}"
        train_bar.update()
        train_bar.set_description(
            desc.format(
                running_loss / (j + 1),
                sanity_check,
                base_value,
                # correct_choice[i - 100:i].sum() / min(100, i)
            )
        )
    train_bar.close()

torch.save(
    value_function.state_dict(),
    'model.pkl'
)

model = ConvEval(model=value_function)
hand1 = Stack(
    cards=[
        Card(RANKS[3], SUITS[0]),
        Card(RANKS[4], SUITS[1]),
        Card(RANKS[4], SUITS[2]),
        Card(RANKS[10], SUITS[3])
    ]
)

hand2 = Stack(
    cards=[
        Card(RANKS[3], SUITS[0]),
        Card(RANKS[4], SUITS[1]),
        Card(RANKS[6], SUITS[2]),
        Card(RANKS[10], SUITS[3])
    ]
)

hand3 = Stack(
    cards=[
        Card(RANKS[4], SUITS[0]),
        Card(RANKS[4], SUITS[1]),
        Card(RANKS[4], SUITS[2]),
        Card(RANKS[10], SUITS[3])
    ]
)

hand4 = Stack(
    cards=[
        Card(RANKS[3], SUITS[0]),
        Card(RANKS[4], SUITS[1]),
        Card(RANKS[5], SUITS[2]),
        Card(RANKS[10], SUITS[3])
    ]
)

hand5 = Stack(
    cards=[
        Card(RANKS[3], SUITS[0]),
        Card(RANKS[4], SUITS[1]),
        Card(RANKS[5], SUITS[2]),
        Card(RANKS[6], SUITS[3])
    ]
)

hand6 = Stack(
    cards=[
        Card(RANKS[3], SUITS[0]),
        Card(RANKS[5], SUITS[1]),
        Card(RANKS[10], SUITS[2]),
        Card(RANKS[12], SUITS[3])
    ]
)

print("hand\tfunc\ttrue")
for hand in [hand1, hand2, hand3, hand4, hand5, hand6]:
    print(
        hand, "\t",
        model.predict(hand).detach().numpy()[0, 0], "\t",
        mean_value_of_hand(hand)
    )

# a = []
# for i in range(correct_choice.shape[0] - 100):
#     a.append(correct_choice[i: i + 100].sum() / 100)
# a = np.array(a)

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].plot(range(len(sanity_checks)), sanity_checks)
# ax[0].set_title("Value of base hand: {:.2f}".format(base_value))
# ax[1].plot(
#     range(len(a)),
#     a
# )
# ax[1].set_title("Frequency of correct choice since beginning")
# plt.show()
