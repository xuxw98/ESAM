## Changes
Apart from hyperparameter tuning, we also make several modifications on the design.
#### Mask refine strategy：
In the decoder, the mask refinement process is no longer fixed at a specific level but adopts a coarse-to-fine refinement strategy. Compared to the original approach, where the initial mask and three subsequent refinements were all at the point level (P), the new strategy generates an initial mask at the superpoint (SP) level and performs the first refinement at this level. It is then converted to the point level for the following two refinements. This approach can further improve the quality of the masks.

Specifically, we added the corresponding code and modified the `mask_pred_mode` parameter in the decoder's configuration from `["P", "P", "P", "P"]` to `["SP", "SP", "P", "P"]`.

#### Merging metric:
In the original method, our merging metric was defined as a weighted sum of geometric similarity, contrastive similarity, and semantic similarity, with the threshold determined through experimental tuning. Now, we have adjusted the merging metric to be the product of geometric similarity and contrastive similarity, with a threshold set to 0. The new metric better measures the similarity between masks.
