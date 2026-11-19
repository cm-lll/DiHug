"""Exact local edge-flip evaluation of the frozen JointLocalScore.

No learned head, new loss, graph neural network or approximate objective.
CPU numpy batches operate on existing hard auxiliary graph samples.
"""
import numpy as np


def kernel(x, y):
    d = max(x.shape[-1], 1)
    distance = (np.square(x).sum(-1)[..., :, None]
                + np.square(y).sum(-1)[..., None, :] - 2 * (x @ np.swapaxes(y, -1, -2)))
    return np.exp(-np.maximum(distance, 0) / (2 * d))


class IncrementalScore:
    def __init__(self, reference_score, bits):
        core = reference_score
        self.n, self.r = core.n, core.r
        self.types = core.types.copy()
        self.query = core.query.numpy().copy()
        self.families = core.families.numpy().copy()
        self.bits = np.asarray(bits, dtype=np.float64).copy()
        if self.bits.shape != (self.query.shape[1],) or not np.isin(self.bits, [0, 1]).all():
            raise ValueError('Incremental evaluator requires a complete hard graph in registered slot order')
        self.adjacency = np.zeros((self.n, self.n), dtype=bool)
        i, j = self.query
        self.adjacency[i, j] = self.bits.astype(bool)
        self.adjacency[j, i] = self.bits.astype(bool)
        # Once per hard graph. CPU feature calculation, not a model call.
        import torch
        with torch.no_grad():
            self.raw = core.features(torch.from_numpy(self.bits)).numpy()
        max_type = int(self.types.max())
        self.triangle_channel = np.full((max_type+1, max_type+1), -1, dtype=int)
        for k, (ta, tb) in enumerate(core.triangle_pairs):
            self.triangle_channel[ta, tb] = self.triangle_channel[tb, ta] = self.r + k
        self.cache = {}
        score = 0.
        for t, nodes in core.nodes.items():
            nodes = nodes.numpy().copy()
            mask, scales = core.masks[t].numpy().copy(), core.scales[t].numpy().copy()
            if not mask.any():
                self.cache[t] = dict(nodes=nodes, mask=mask)
                continue
            x = np.log1p(self.raw[nodes][:, mask]) / scales
            y = core.targets[t].numpy().copy()
            xx, xy = kernel(x, x), kernel(x, y)
            self.cache[t] = dict(nodes=nodes, mask=mask, scale=scales, x=x, y=y,
                xx=xx, xx_row=xx.sum(-1), xy_row=xy.sum(-1))
            score += float(xx.mean()-2*xy.mean()+core.target_self[t].item())
        self.base_score = score / len(self.cache)
        for a in [self.types, self.query, self.families, self.bits, self.adjacency, self.raw]:
            a.flags.writeable = False

    def differences(self, indices=None, batch_size=128):
        """Return D(B_-e,1)-D(B_-e,0) for the requested registered edges.

        All candidates derive from ONE immutable B. Do not sequentially edit B.
        This function computes no probability derivatives and samples no RNG.
        """
        indices = np.arange(len(self.bits)) if indices is None else np.asarray(indices, dtype=int)
        result = np.empty(len(indices), dtype=np.float64)
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start+batch_size]
            i, j = self.query[:, ix]
            rel = self.families[ix]
            sign = 1 - 2*self.bits[ix]
            b = np.arange(len(ix))
            common = self.adjacency[i] & self.adjacency[j]
            changed = common.copy()
            changed[b, i] = True
            changed[b, j] = True
            # raw is the old graph's integer-valued structure summary.
            raw_new = np.broadcast_to(self.raw, (len(ix), *self.raw.shape)).copy()
            raw_new[b, i, rel] += sign
            raw_new[b, j, rel] += sign
            bb, kk = np.nonzero(common)
            if len(bb):
                # Each newly created/deleted triangle affects ALL its three centers.
                ch_i = self.triangle_channel[self.types[j[bb]], self.types[kk]]
                ch_j = self.triangle_channel[self.types[i[bb]], self.types[kk]]
                ch_k = self.triangle_channel[self.types[i[bb]], self.types[j[bb]]]
                np.add.at(raw_new, (bb, i[bb], ch_i), sign[bb])
                np.add.at(raw_new, (bb, j[bb], ch_j), sign[bb])
                np.add.at(raw_new, (bb, kk, ch_k), sign[bb])
            if raw_new.min() < 0:
                raise ArithmeticError('Negative local count: stop; do not clip a bad increment')
            delta = np.zeros(len(ix))
            for cache in self.cache.values():
                nodes, mask = cache['nodes'], cache['mask']
                if not mask.any():
                    continue
                local = changed[:, nodes]
                counts = local.sum(-1)
                width = int(counts.max())
                if width == 0:
                    continue
                row, col = np.nonzero(local)
                offsets = np.r_[0, np.cumsum(counts)[:-1]]
                position = np.arange(len(row))-offsets[row]
                gather = np.zeros((len(ix), width), dtype=int)
                gather[row, position] = col
                valid = np.arange(width)[None, :] < counts[:, None]
                node_ids = nodes[gather]
                updated = raw_new[np.arange(len(ix))[:,None], node_ids][..., mask]
                xnew = np.log1p(updated) / cache['scale']
                xold = cache['x'][gather]
                # Cache old row sums. Correct changed-changed terms so neither
                # omitted pairs nor double-counted changed pairs alter the MMD.
                new_old = kernel(xnew, cache['x'])
                row_delta = ((new_old.sum(-1)-cache['xx_row'][gather])*valid).sum(-1)
                new_new_changed = kernel(xnew, xnew)
                new_old_changed = kernel(xnew, xold)
                old_old_changed = cache['xx'][gather[:,:,None], gather[:,None,:]]
                valid_pair = valid[:,:,None] & valid[:,None,:]
                correction = ((new_new_changed-2*new_old_changed+old_old_changed)*valid_pair).sum((-1,-2))
                cross_delta = ((kernel(xnew, cache['y']).sum(-1)-cache['xy_row'][gather])*valid).sum(-1)
                ntype = len(nodes)
                delta += (2*row_delta + correction - 2*cross_delta) / (ntype*ntype)
            # Flip delta has opposite sign when the starting graph contains e.
            result[start:start+len(ix)] = sign * delta / len(self.cache)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite exact incremental difference')
        return result

    def logit_gradient(self, probability, indices=None, batch_size=128):
        probability = np.asarray(probability, dtype=np.float64)
        if probability.shape != self.bits.shape or not np.isfinite(probability).all():
            raise ValueError('Probability/registered slots mismatch')
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError('Invalid probability')
        indices = np.arange(len(self.bits)) if indices is None else np.asarray(indices, dtype=int)
        p = probability[indices]
        return p*(1-p)*self.differences(indices, batch_size)
