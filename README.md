# Deep Learning & NLP Labs

A selection of my coursework on deep learning and natural language processing. Each folder contains Jupyter notebooks (Python, PyTorch / TensorFlow-Keras, Hugging Face), going from the fundamentals (linear regression, backpropagation) to modern techniques (LoRA fine-tuning, RLHF with PPO, RAG).

## Repository structure

```
.
├── deep-learning/   # 15 notebooks: from scratch implementations, MLPs, CNNs, explainability, captioning
└── nlp/             # 4 labs: tokenization, fine-tuning, preference alignment, RAG
```

**Main tools:** PyTorch, TensorFlow/Keras, Hugging Face (`transformers`, `peft`, `trl`, `datasets`), LangChain, FAISS, NLTK, SentencePiece, scikit-learn, Weights & Biases, matplotlib.

---

## NLP

| Lab | Notebook | Content |
|-----|----------|---------|
| 1 | [`text-tokenization.ipynb`](nlp/text-tokenization.ipynb) | **Text tokenization.** NLTK, lexical statistics, testing **Zipf's law** on a book and on the undeciphered Voynich manuscript, word-based tokenization vs **Byte Pair Encoding** (SentencePiece). |
| 2 | [`emotion-classification.ipynb`](nlp/emotion-classification.ipynb) | **Emotion classifier.** Fine-tuning **GPT-2 with LoRA** (`peft`) on the `dair-ai/emotion` dataset (6 classes). Training vs. untrained baseline, accuracy / loss curves, classification report, **confusion matrix**, model export. |
| 3 | [`rlhf-ppo.ipynb`](nlp/rlhf-ppo.ipynb) | **Preference alignment (RLHF).** Custom **PPO** training loop on GPT-2 + LoRA, using a **reward model** and a frozen **reference model**; comparison of generations before and after alignment. |
| 4 | [`information-retrieval-and-reranking.ipynb`](nlp/information-retrieval-and-reranking.ipynb) | **RAG: retrieval and reranking.** Recall@N, sentence embeddings (`all-MiniLM-L6-v2`), **FAISS** vector store and its evaluation, **cross-encoder reranking** (`bge-reranker-base`), answer generation with LangChain. |

---

## Deep learning

### Fundamentals (from scratch)

| Notebook | Content |
|----------|---------|
| [`linear_regression.ipynb`](deep-learning/linear_regression.ipynb) | Data visualization, **normal equations** with NumPy and with PyTorch tensors. |
| [`linear_regression2.ipynb`](deep-learning/linear_regression2.ipynb) | Normalization, cost function, **gradient descent**, influence of the learning rate, PyTorch **autograd**. |
| [`computational_graphs_linear_regression.ipynb`](deep-learning/computational_graphs_linear_regression.ipynb) | Own **computational graph framework**; (batched) stochastic gradient descent, **momentum**, per-parameter learning rates, early stopping and **Adam**. |
| [`binary_classification.ipynb`](deep-learning/binary_classification.ipynb) | **Logistic regression**: full-batch gradient descent implemented by hand and with autograd, learning-rate tuning. |
| [`softmax_classification.ipynb`](deep-learning/softmax_classification.ipynb) | **Multinomial logistic regression on MNIST**: softmax, loss, mini-batch gradient descent, hyperparameter tuning, then full PyTorch implementation. |
| [`backpropagation.ipynb`](deep-learning/backpropagation.ipynb) | **Backpropagation implemented manually** (linear layers, activations, cost), validated against PyTorch autograd; overfitting check on a single sample, then full training. |

### MLPs and training techniques

| Notebook | Content |
|----------|---------|
| [`mlps_for_fashion_MNIST.ipynb`](deep-learning/mlps_for_fashion_MNIST.ipynb) | **MLPs on Fashion-MNIST**: effect of training-set size, model complexity, and **regularization** (cost and accuracy analysis). |
| [`pytorch_CIFAR10_mlp.ipynb`](deep-learning/pytorch_CIFAR10_mlp.ipynb) | 1-layer and 2-layer networks on CIFAR-10, **weight visualization** as image tiles, tuning for better performance. |
| [`parameter_initialization.ipynb`](deep-learning/parameter_initialization.ipynb) | **Weight initialization** on Fashion-MNIST (Keras/TensorFlow): zero, random, standard, uniform, **Xavier/Glorot**, **Kaiming/He**; comparison and analysis. |
| [`batchnormalization.ipynb`](deep-learning/batchnormalization.ipynb) | **Benefit of BatchNorm** on CIFAR-10: mixed CNN/MLP architecture, Tanh vs ReLU, BatchNorm after each layer, dropout, comparison plots. |

### Convolutional networks

| Notebook | Content |
|----------|---------|
| [`pytorch_CIFAR10_cnn.ipynb`](deep-learning/pytorch_CIFAR10_cnn.ipynb) | **CNNs on CIFAR-10** in PyTorch, from a simple CNN to deeper architectures. |
| [`CIFAR10_CNN_data_augmentation.ipynb`](deep-learning/CIFAR10_CNN_data_augmentation.ipynb) | CNN on CIFAR-10 (Keras): **data augmentation**, confusion matrix, visualizing what the model learns, deeper models and architecture analysis. |
| [`CNN_on_iCoSimal.ipynb`](deep-learning/CNN_on_iCoSimal.ipynb) | **Practical work on the iCoSimal V3 dataset**: depth of CNNs, **hyperparameter tuning** (filters, kernel sizes, pooling, learning rate, batch size, epochs), over/underfitting, **regularization** (dropout, early stopping), **optimizers** (Adam, RMSprop), **Weights & Biases**, data augmentation, transfer learning with **ResNet** and **GoogLeNet**. |
| [`GradCam.ipynb`](deep-learning/GradCam.ipynb) | **Explainability with Grad-CAM** on a small CNN trained on clothing images: heatmaps per class and analysis of where Grad-CAM is (and is not) informative. |

### Vision and language

| Notebook | Content |
|----------|---------|
| [`captioning_Flicker8k.ipynb`](deep-learning/captioning_Flicker8k.ipynb) | **Image captioning on Flickr8k**: ResNet encoder + **LSTM** decoder, teacher forcing, masked cross-entropy, **Show & Tell** and **Show, Attend & Tell**, evaluation with perplexity and **BLEU-1 to 4**, **beam search**, **attention map** visualization, experiment tracking with W&B. |

---

## Running the notebooks

```bash
git clone https://github.com/XaVIIIerg/deep-learning-nlp-coursework.git
cd deep-learning-nlp-coursework
pip install jupyter torch torchvision tensorflow transformers peft trl datasets scikit-learn matplotlib pandas
jupyter notebook
```

Some notebooks (GPT-2 fine-tuning, PPO, captioning) are much faster on a GPU, for example on Google Colab. Datasets such as CIFAR-10, MNIST and Fashion-MNIST are downloaded automatically; others (iCoSimal V3, Flickr8k) must be downloaded separately.

## Note

These notebooks were produced as part of my university coursework. They are shared to illustrate my learning path and the topics I have worked on.
