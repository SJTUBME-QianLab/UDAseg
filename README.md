# UDAseg

Title: 
Utilizing GCN and Meta-Learning Strategy in Unsupervised Domain Adaptation for Pancreatic Cancer Segmentation

Abstract:
Automated pancreatic cancer segmentation is highly crucial for computer-assisted diagnosis of pancreatic cancer. The general practice is to label images from selected modalities since it is expensive to label images from all modalities. This practice brought about a significant interest in learning the knowledge transfer from the labeled modalities to unlabeled ones. However, the imaging parameter inconsistency between labeled and unlabeled modalities leads to a domain shift, limiting the transfer learning performance. In this work, we propose a pancreatic cancer segmentation framework based on unsupervised domain adaptation. Our model promotes synergistic collaboration between image and feature adaptation modules. Specifically, we employ encoders to separate domain-invariant features from domain-specific ones, and adopt adversarial learning to perform implicit feature alignment. Subsequently, the domain-specific features are modified for appearance transfer from the source domain to the target domain, to achieve image alignment. Besides, we introduce a graph convolutional network to aggregate high-dimensional features, focus on pixel-wise response, and hence ensure the integrity of key structural features during adaptation. Moreover, we exploit a meta-learning strategy with good generalization capabilities to strike a reasonable balance of different adaptation aspects, and lead to improved unsupervised domain adaptation segmentation performance. Extensive experiments on four multi-parameter pancreatic-cancer magnetic resonance imaging (MRI) datasets demonstrate improved performance in all adaptation directions, thus confirming our model’s effectiveness for segmenting unlabeled images with small-sized pancreatic cancer. The obtained results are promising for reducing the burden of medical image annotation and improving the performance of computer-aided diagnosis for pancreatic cancer.  

Our source codes with a demon will be released.

Author list: Jun Li, Xiaohua Qian
