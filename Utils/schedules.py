class LinearScheduler(object):
    """
    Linearly increase from 0 to 1 over iters
    """
    def __init__(self, iters, maxval=1.0):
        iters = max(1, iters)
        self.val = 0#maxval / iters
        self.maxval = maxval
        self.iters = iters

    def step(self):
        self.val = min(self.maxval, self.val + self.maxval / self.iters)

    def __call__(self):
        return self.val


class ZeroLinearScheduler(object):
    """
    Linearly increase from zero to 1 from start to end 
    """
    def __init__(self, start, end, maxval=1.0):
        self.val = 0
        self.start = start
        self.end = end
        self.maxval = maxval
        self.i = 0

    def step(self):
        if self.i >= self.start:
            self.val = min(self.maxval, self.val + self.maxval/(self.end - self.start))
        self.i += 1

    def __call__(self):
        return self.val
