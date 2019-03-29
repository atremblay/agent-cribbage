import numpy as np


class Policy:
    @staticmethod
    def random_choice(probability):
        return np.random.choice(np.arange(0, len(probability)), p=probability)

    def choose(self, states, value_function):
        V_s = [v.data.tolist()[0] for v in value_function.evaluate(states)]
        return self.prob(V_s)

    def prob(self, V_s):
        """
        To implement in subclass
        :return:
        """
        pass