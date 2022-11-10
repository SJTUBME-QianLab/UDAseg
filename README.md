
# Utilizing GCN and Meta-Learning Strategy in Unsupervised Domain Adaptation for Pancreatic Cancer Segmentation

This repository holds the PyTorch code of our IEEE JBHI paper *Utilizing GCN and Meta-Learning Strategy in Unsupervised Domain Adaptation for Pancreatic Cancer Segmentation*. 

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (**Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University**) preserve the copyright and all legal rights of these codes.


## Author List

Jun Li, Chaolu Feng, Xiaozhu Lin, Xiaohua Qian


## Abstract

Automated pancreatic cancer segmentation is highly crucial for computer-assisted diagnosis. The general practice is to label images from selected modalities since it is expensive to label all modalities. This practice brought about a significant interest in learning the knowledge transfer from the labeled modalities to unlabeled ones. However, the imaging parameter inconsistency between modalities leads to a domain shift, limiting the transfer learning performance. Therefore, we propose an unsupervised domain adaptation segmentation framework for pancreatic cancer based on GCN and meta-learning strategy. Our model first transforms the source image into a target-like visual appearance through the synergistic collaboration between image and feature adaptation. Specifically, we employ encoders incorporating adversarial learning to separate domain-invariant features from domain-specific ones to achieve visual appearance translation. Then, the meta-learning strategy with good generalization capabilities is exploited to strike a reasonable balance in the training of the source and transformed images. Thus, the model acquires more correlated features and improve the adaptability to the target images. Moreover, a GCN is introduced to supervise the high-dimensional abstract features directly related to the segmentation outcomes, and hence ensure the integrity of key structural features. Extensive experiments on four multi-parameter pancreatic-cancer magnetic resonance imaging datasets demonstrate improved performance in all adaptation directions, confirming our model's effectiveness for unlabeled pancreatic cancer images. The results are promising for reducing the burden of annotation and improving the performance of computer-aided diagnosis of pancreatic cancer. Our source codes will be released at https://github.com/SJTUBME-QianLab/UDAseg, once this manuscript is accepted for publication.


## Requirements

* `pytorch 1.1.0`
* `numpy 1.17.2`
* `python 3.6.1`



## Citing the Work

If you find our code useful in your research, please consider citing:

```
@ARTICLE{9444631,
  author={Li, Jun and Feng, Chaolu and Lin, Xiaozhu and Qian, Xiaohua},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Utilizing GCN and Meta-Learning Strategy in Unsupervised Domain Adaptation for Pancreatic Cancer Segmentation}, 
  year={2022},
  volume={26},
  number={1},
  pages={79-89},
  doi={10.1109/JBHI.2021.3085092}}
```


## Contact

For any question, feel free to contact

```
Jun Li : dirk_li@sjtu.edu.cn
```


## Acknowledgements

This code is developed on the code base of [DISE](https://github.com/a514514772/DISE-Domain-Invariant-Structure-Extraction). Many thanks to the authors of this work. 