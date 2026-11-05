import random
import unittest
import pathlib
import subprocess
from itertools import combinations, combinations_with_replacement
from features_cpu import extract, same_component


class Features(unittest.TestCase):
    pairs = tuple(combinations_with_replacement(range(3), 2))

    def run_case(self, types, edges):
        return extract(types, edges, [self.pairs.index(tuple(sorted((types[u], types[v])))) for u, v in edges], self.pairs, 3)

    def test_typed_triangle_and_disconnected_node(self):
        x = self.run_case([0, 1, 2, 0], [(0, 1), (0, 2), (1, 2)])
        self.assertEqual([sum(r) for r in x['triangle_counts']], [1, 1, 1, 0])
        self.assertEqual(x['component_sizes'], [3, 1])
        self.assertEqual(same_component(x, [(0, 1), (0, 3)]), [1., 0.])
        self.assertAlmostEqual(x['node_features'][0][-4], .75)
        self.assertEqual(x['node_features'][3][-3:], [1., 0., 0.])

    def test_empty_and_singleton(self):
        for types in ([0], [0, 1, 2]):
            x = self.run_case(types, [])
            self.assertEqual(x['component_sizes'], [1] * len(types))
            self.assertTrue(all(sum(r) == 0 for r in x['triangle_counts']))

    def test_component_histogram_optimization_matches_reviewed_reference(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        old = subprocess.check_output(['git','show',
            '9c721ecc7ea08fd0db238bee977120da16dde990:experiments/structure_input_v1/features_cpu.py'],cwd=root)
        namespace = {}
        exec(compile(old,'reviewed_reference','exec'),namespace)
        types = [0,1,2,0,1]
        slots = list(combinations(range(5),2))
        for mask in range(1<<len(slots)):
            edges = [e for j,e in enumerate(slots) if mask & (1<<j)]
            families = [self.pairs.index(tuple(sorted((types[u],types[v])))) for u,v in edges]
            self.assertEqual(extract(types,edges,families,self.pairs,3),
                             namespace['extract'](types,edges,families,self.pairs,3))

    def test_invalid_graphs(self):
        for edges in ([(0, 0)], [(1, 0)], [(0, 1), (0, 1)]):
            with self.assertRaises(ValueError):
                self.run_case([0, 1], edges)
        with self.assertRaises(ValueError):
            extract([0, 1], [(0, 1)], [0], [(0, 0)], 2)

    def test_exhaustive_five_node_graphs_and_permutation(self):
        types = [0, 1, 2, 0, 1]
        slots = list(combinations(range(5), 2))
        perm = [3, 1, 4, 0, 2]
        perm_types = [0] * 5
        for i in range(5):
            perm_types[perm[i]] = types[i]
        rng = random.getstate()
        for mask in range(1 << len(slots)):
            edges = [e for j, e in enumerate(slots) if mask & (1 << j)]
            x = self.run_case(types, edges)
            edge_set = set(edges)
            expected = [0] * 5
            for triplet in combinations(range(5), 3):
                if all(e in edge_set for e in combinations(triplet, 2)):
                    for i in triplet:
                        expected[i] += 1
            self.assertEqual([sum(r) for r in x['triangle_counts']], expected)
            y = self.run_case(perm_types, [tuple(sorted((perm[u], perm[v]))) for u, v in edges])
            for i in range(5):
                self.assertEqual(x['node_features'][i], y['node_features'][perm[i]])
                self.assertTrue(all(0 <= v <= 1 for v in x['node_features'][i]))
            for u, v in slots:
                self.assertEqual(x['component_ids'][u] == x['component_ids'][v],
                                 y['component_ids'][perm[u]] == y['component_ids'][perm[v]])
        self.assertEqual(rng, random.getstate())


if __name__ == '__main__':
    unittest.main()
