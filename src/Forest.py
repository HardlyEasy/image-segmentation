class Forest:
    """Disjoint-set forest with union-by-rank heuristic
    """
    def __init__(self, node_num):
        self.num_set = node_num
        self.parent = []
        self.rank = []
        self.sizes = []
        for x in range(0, node_num):
            self.parent.append(x)
        self.rank.append(0)
        self.sizes.append(1)
        self.rank = self.rank * node_num
        self.sizes = self.sizes * node_num

    def get_size(self, x):
        return self.sizes[x]

    # Same as find_set(x) pseudocode
    def find_set(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]

    def merge(self, x, y):
        # Same as union(x, y) in pseudocode
        x = self.find_set(x)
        y = self.find_set(y)
        # Same as link(x, y) in pseudocode
        if x != y:
            if self.rank[x] > self.rank[y]:
                self.parent[y] = x
                self.sizes[x] += self.sizes[y]
            else:
                self.parent[x] = y
                self.sizes[y] += self.sizes[x]
                if self.rank[x] == self.rank[y]:
                    self.rank[y] += 1
        self.num_set -= 1
