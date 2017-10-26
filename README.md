# Information Bottleneck for Deep Learning

![](https://raw.githubusercontent.com/LargePanda/Information-Bottleneck-for-Deep-Learning/master/img/plot2.png)


**Dataset**: fashion mnist

**Model 1**: MLP with Batch Normalization

Refactored and reused my previous code for this implementation. 

URL: [link](https://github.com/LargePanda/Information-Bottleneck-for-Deep-Learning/blob/master/Fashion%20MNIST%20experiments.ipynb)

**Model 2**: CNN

Implemented in PyTorch

URL: [link](https://github.com/LargePanda/Information-Bottleneck-for-Deep-Learning/blob/master/CNN.ipynb)


**Model extension**: 
1. MLP with more than 3 layers (computationally expensive, in progress)
2. MLP with weights intialization via denoised autoencoder (in progress)

**Other parameters**: 

number of bins for MI: 10

**Papers**
0. [Opening the black box of Deep Neural Networks via Information](https://arxiv.org/pdf/1703.00810.pdf)
1. [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/pdf/1503.02406.pdf)
2. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
3. [Regularization of Neural Networks using DropConnect](https://cs.nyu.edu/~wanli/dropc/dropc.pdf)
