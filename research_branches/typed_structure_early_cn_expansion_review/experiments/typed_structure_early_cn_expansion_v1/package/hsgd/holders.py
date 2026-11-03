"""Only containers needed by source-pinned sparse backbone; no training framework."""


class PlaceHolder:
    def __init__(self, X, E, y, charge=0):
        self.X, self.E, self.y, self.charge = X, E, y, charge


class SparsePlaceHolder:
    def __init__(self, node, edge_index, edge_attr, y, ptr=None, batch=None, charge=None):
        self.node, self.edge_index, self.edge_attr, self.y = node, edge_index, edge_attr, y
        self.ptr, self.batch, self.charge = ptr, batch, charge
