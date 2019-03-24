import numpy as np


class Policy:
    @staticmethod
    def random_choice(probability):
        return np.random.choice(np.arange(0, len(probability)), p=probability)

    def choose(self, actions, value_function):
        V_s = [value_function.evaluate(a).data.tolist()[0] for a in actions]
        return self.prob(V_s)

    def prob(self, V_s):
        """
        To implement in subclass
        :return:
        """
        pass