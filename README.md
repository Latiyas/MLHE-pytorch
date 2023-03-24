# MLHE-pytorch
Code Appendix for Monotone Learning with Hypothesis Evolution

## Requirements
- Python 3.7

## Installation
Install pytorch, scikit-learn, statsmodels, matplotlib, datasets, transformers, timm, opencv from the web.


## Document organization structure
```
|-- code                			# code
	|-- models          			# base model
|-- data                			# data set
	|-- CIFAR10				# CIFAR-10 data set can be put here
	|-- MNIST				# MNIST data set can be put here
	|-- SST2				# SST-2 data set can be put here
		|-- csv_data      		# the processed data from SST-2 can be put here
		|-- tsv_data      		# raw data from SST-2 can be put here
	|-- TinyImageNet				# Tiny ImageNet data set can be put here
|-- prediction_result				# final result will be stored here
|-- user_data           			# user data
	|-- tmp_data        			# to save temporary file
		|-- CIFAR10			# to save intermediate results for CIFAR-10
		|-- MNIST			# to save intermediate results for MNIST
		|-- SST2			# to save intermediate results for SST-2
		|-- TinyImageNet	# to save intermediate results for Tiny ImageNet
```

## Data set
* Download and extract [MNIST](http://yann.lecun.com/exdb/mnist/) data set in the directory '/data/MNIST/'.
* Download and extract [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) data set in the directory '/data/CIFAR10'.
* Download and extract [SST-2](https://gluebenchmark.com/tasks) data set in the directory '/data/SST2/tsv_data'.
* Download and extract [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) data set in the directory '/data/Tiny ImageNet'.

## Usage
```
# Enter code path:
cd code

# Data preprocessing:
python Preprocessing.py

# For simplicity, the scenario SiAj(i∈{1,2}, j∈{1,2}) in the paper is marked with different numbers, and the corresponding relationship is as follows: S1A1-4, S1A2-3, S2A1-6, S2A2-5. 
# Then take case S1A1-4 as an example to analyze. You can switch to other scenarios by modifying parameter "divide_type".
# Experiment on MNIST:
python main.py --algo_type MLHE --dataset MNIST --epochs 100 --divide_type 4 --execution_num 1 --split_num 2 --retain_num 16
python main.py --algo_type MTsimple --dataset MNIST --epochs 100 --divide_type 4 --execution_num 32
python main.py --algo_type retraining --dataset MNIST --epochs 100 --divide_type 4 --execution_num 1

# Experiment on CIFAR-10:
python main.py --algo_type MLHE --dataset CIFAR10 --epochs 100 --divide_type 4 --execution_num 1 --split_num 2 --retain_num 16
python main.py --algo_type MTsimple --dataset CIFAR10 --epochs 100 --divide_type 4 --execution_num 32
python main.py --algo_type retraining --dataset CIFAR10 --epochs 100 --divide_type 4 --execution_num 1

# Experiment on SST-2:
python main.py --algo_type MLHE --dataset SST2 --epochs 100 --divide_type 4 --execution_num 1 --split_num 2 --retain_num 16
python main.py --algo_type MTsimple --dataset SST2 --epochs 100 --divide_type 4 --execution_num 32
python main.py --algo_type retraining --dataset SST2 --epochs 100 --divide_type 4 --execution_num 1

# Experiment on Tiny ImageNet:
python main.py --algo_type MLHE --dataset TinyImageNet --epochs 50 --divide_type 4 --execution_num 1 --split_num 2 --retain_num 8
python main.py --algo_type MTsimple --dataset TinyImageNet --epochs 50 --divide_type 4 --execution_num 16
python main.py --algo_type retraining --dataset TinyImageNet --epochs 50 --divide_type 4 --execution_num 1

# Generate learning curves and calculate the probability of monotonic update:
python CurveAnalysis.py

# More analysis
# Stability analysis: run MLHE multiple times to calculate the mean and standard deviation.
python main.py --algo_type MLHE --dataset MNIST --epochs 100 --divide_type 4 --execution_num 3 --split_num 2 --retain_num 16
python StabilityAnalysis.py

# Analyze the influence of 𝛼 and 𝛽: the performance of MLHE in different 𝛼 and 𝛽 can be obtained by modifying the input of "split_num" and "retain_num". 
python main.py --algo_type MLHE --dataset MNIST --epochs 100 --divide_type 4 --execution_num 1 --split_num 2 --retain_num 4
python Tuning.py
```
