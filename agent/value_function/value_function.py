class ValueFunction:
    def update(self, batch):
        pass

    def evaluate(self, x):
        return self.forward(x)

    def forward(self, x):
        pass