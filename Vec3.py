



class Vec3:



    def __init__(self, i, j, k):
        self.values = [i, j, k]

    def multiply(self, v):
        return [self.values[i]*v.values[i] for i in range(3)]

    def add(self):