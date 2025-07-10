# GrokFormer
A PyTorch implementation of GrokFormer "GrokFormer: Graph Fourier Kolmogorov-Arnold Transformer". <br>
# Environment Settings
This implementation is based on Python3. To run the code, you need the following dependencies: <br>
* torch==1.8.1
* torch-geometric==1.7.2
* scipy==1.2.1
* numpy==1.19.5
* tqdm==4.59.0
* seaborn==0.11.2
* scikit-learn==0.24.2
# Node Classification Datasets
The data folder contains five homophilic benchmark datasets(Cora, Citeseer, Pubmed, Photo, Physics), and five heterophilic datasets(Penn94, Chameleon, Squirrel, Actor, Texas) from [BernNet](https://github.com/ivam-he/BernNet). We use the same experimental setting (60\%/20\%/20\% random splits for train/validation/test with the same epochs, run ten times, early stopping) as [BernNet](https://github.com/ivam-he/BernNet).  
# Run node classification experiment:
    $ sh run.sh
# Examples
 Training a model on the default dataset.  
![image](https://github.com/GGA23/GrokFormer/blob/main/GrokFormer_demo.gif)
# Baselines links
* [H2GCN](https://github.com/GitEventhandler/H2GCN-PyTorch)
* [HopGNN](https://github.com/JC-202/HopGNN)
* [GPRGNN](https://github.com/jianhao2016/GPRGNN)
* [BernNet](https://github.com/ivam-he/BernNet)
* [JacobiConv](https://github.com/GraphPKU/JacobiConv)
* [HiGCN](https://github.com/Yiminghh/HiGCN)
* [NodeFormer](https://github.com/qitianwu/NodeFormer)
* [SGFormer](https://github.com/qitianwu/SGFormer)
* [NAGphormer](https://github.com/JHL-HUST/NAGphormer)
* [PolyFormer](https://github.com/air029/PolyFormer)
* [Specformer](https://github.com/DSL-Lab/Specformer)
* The implementations of others are taken from the Pytorch Geometric library
## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ai2024grokformer,
  title     = {GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers},
  author    = {Ai, Guoguo and Pang, Guansong and Qiao, Hezhe and Gao, Yuan and Yan, Hui},
  booktitle = {Proceedings of the 42st International Conference on Machine Learning (ICML)},
  year      = {2025}
}

Feel free to contact guoguo@njust.edu.cn if you have any questions.

# Acknowledgements
The node-level code and filter learning code are implemented based on [Specformer: Spectral Graph Neural Networks Meet Transformers](https://github.com/DSL-Lab/Specformer).
The graph-level (TUD Benchmarks) code are implemented based on the public code (https://github.com/kavehhassani/mvgrl/tree/master/graph)

