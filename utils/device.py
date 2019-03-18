class Device:
    def __init__(self, enable_cuda=True):
        self.isCuda = enable_cuda

    def __call__(self, x):
        if self.isCuda:
            return x.cuda()
        else:
            return x

    def __str__(self):
        if self.isCuda:
            return 'Cuda'
        else:
            return 'Cpu'


# Redirection instance
device = Device()


