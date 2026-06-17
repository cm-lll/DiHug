"""
HGHG subgraphs dataset adapter.

This module intentionally reuses the ACMSubgraphs loader pipeline because the
on-disk format is the same (subgraph_*/nodes.pt, edges.pt, meta.json).
We expose dedicated class names so experiment configs and logs use
"hghg_subgraphs" instead of "acm_subgraphs".
"""

from datasets.acm_subgraphs_dataset import ACMSubgraphsDataModule, ACMSubgraphsInfos


class HGHGSubgraphsDataModule(ACMSubgraphsDataModule):
    """Dedicated DataModule name for HGHG-format subgraphs."""


class HGHGSubgraphsInfos(ACMSubgraphsInfos):
    """Dedicated DatasetInfos name for HGHG-format subgraphs."""

