import numpy as np
from random import shuffle

class replay_buffer:
    """
    Data in buffer is a list of numpy array
    ith row is [states,actions,rewards,next_states,q,q_next]
    """
    def __init__(self, limit=5e5, sample_rate=0.7):
        self.limit = int(limit)
        self.sample_rate = sample_rate
        self.buffer = []

    def add_to_buffer(self,data):
##        if method == 'append':
##            self.buffer.append(data)
##        elif method == 'extend':
        self.buffer.extend(data)
##        else:
        if len(self.buffer) > self.limit:
            shuffle(self.buffer)
            self.buffer = self.buffer[-self.limit:]

    def sample(self):
        idx = np.random.permutation(len(self.buffer))\
              [:int(self.limit*self.sample_rate)]
##        print(np.array(self.buffer))
        return np.asarray(self.buffer)[idx]

    def size(self):
        return len(self.buffer)
