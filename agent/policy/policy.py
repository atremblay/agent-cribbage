import numpy as np


class Policy:
    @staticmethod
    def random_choice(probability):
        return np.random.choice(np.arange(0, len(probability)), p=probability)

    def choose(self, states, value_function):
        values = value_function.evaluate(states).squeeze()
        V_s = [v.data.tolist() for v in values] if values.nelement() > 1 else [values.data.tolist()]
        return self.prob(V_s)

    def prob(self, V_s):
        """
        To implement in subclass
        :return:
        """
        pass

    def step(self):
        pass