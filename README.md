# HighDimensionalDataStreamClassification
Title: Learning High-Dimensional Evolving Data Streams With Limited Labels
-------------------------------------------------------------------------------------------------------------------------------
Abstract
-------------------------------------------------------------------------------------------------------------------------------
In the context of streaming data, learning algorithms often need to confront several unique challenges, such as concept drift, label scarcity, and high dimensionality. Several concept drift-aware data stream learning algorithms have been proposed to tackle these issues over the past decades. However, most existing algorithms utilize a supervised learning framework and require all true class labels to update their models. Unfortunately, in the streaming environment, requiring all labels is unfeasible and not realistic in many real-world applications. Therefore, learning data streams with minimal labels is a more practical scenario. Considering the problem of the curse of dimensionality and label scarcity, in this article, we present a new semisupervised learning technique for streaming data. To cure the curse of dimensionality, we employ a denoising autoencoder to transform the high-dimensional feature space into a reduced, compact, and more informative feature representation. Furthermore, we use a cluster-and-label technique to reduce the dependency on true class labels. We employ a synchronization-based dynamic clustering technique to summarize the streaming data into a set of dynamic microclusters that are further used for classification. In addition, we employ a disagreement-based learning method to cope with concept drift. Extensive experiments performed on many real-world datasets demonstrate the superior performance of the proposed method compared to several state-of-the-art methods.

-------------------------------------------------------------------------------------------------------------------------------

This is the version 1, and it will be constantly improved. We will update the progress.

-------------------------------------------------------------------------------------------------------------------------------

Reference: S. U. Din, J. Kumar, J. Shao, C. B. Mawuli and W. D. Ndiaye, "Learning High-Dimensional Evolving Data Streams With Limited Labels," in IEEE Transactions on Cybernetics, vol. 52, no. 11, pp. 11373-11384, Nov. 2022, doi: 10.1109/TCYB.2021.3070420.

-------------------------------------------------------------------------------------------------------------------------------
ATTN: This code were developed by Salah Ud Din (salahuddin@std.uestc.edu.cn). For any problem and suggestment, please feel free to contact Mr. Salah Ud Din.
