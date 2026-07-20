# Classic baseline protocol

The classic baselines are adaptations, not native heterogeneous generators.
All four receive the same fixed node types, sample each relation-family count
from its empirical Binomial marginal, and may only emit schema-valid edges.

* **NetGAN** uses HG2NP's published PyTorch NetGAN implementation.  Generated
  random-walk transition scores are masked by the heterogeneous schema before
  graph assembly.
* **VGAE** uses node-type embeddings, a GCN variational encoder, and one
  bilinear decoder per relation family.
* **GraphRNN** is the GraphRNN-S autoregressive variant conditioned on the
  fixed node-type sequence.  It is restricted to a finite predecessor window.
* **GraphVAE-small** uses a dense VAE over all schema-valid candidate edges and
  is rejected when the candidate domain exceeds the configured small-graph
  limit.

HG2NP's published IMDB/DBLP results are a separate protocol.  Its repository
splits the original networks by metadata such as year, country, language,
author area, and conference, then keeps graphs with at most 200 nodes.  Those
numbers must not be mixed into a full-graph table.  They may be reported in a
separate split-graph comparison if the authors' modified metadata is available.
