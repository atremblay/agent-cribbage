from .policy import Policy
from .register import register
import logging

@register
class Human(Policy):
    def __init__(self):
        super(Human).__init__()

        def human(self, message, *args, **kws):
            if self.isEnabledFor(100):
                self._log(100, message, args, **kws)

        logging.Logger.human = human

    @property
    def custom_hash(self):
        return __name__

    def prob(self, V_s):
        pass