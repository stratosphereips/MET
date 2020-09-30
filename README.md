# Model extraction attack framework

The goal of this project is to allow easily test different types of model extraction attacks, on either framework or 
user supplied models and datasets. 

## Prerequisites

The code requires Python 3.8 and depends on Pytorch 1.5.1, Torchvision 0.6.1, Numpy 1.18.5, PyYaml 5.3.1 , tqdm 4.48.2, 
dacite 1.5.1, pytorch_lighting 0.9.0 and dataclasses 0.7. To get all neccesary packages run `pip install -r requirements.txt`.

The example `copycat_goc.py` also requires ImageNet2012 to be downloaded, which youcan download [here][1].

[1]: http://image-net.org/challenges/LSVRC/2012/downloads.php#images 

## Running

To test the framework simply run of the examples in the examples folder.

### Configuration

The whole framework is configured using the _config.yaml_ file. The configuration file is separated into two parts 
test and attacks. Test part contains configuration settings for the framework itself. The attacks part contains
configuration settings for the respective settings. You can find all the possible different configuration options for
both the framework and attacks on [wiki][5].

The repository contains example _config.yaml_ file, which can be used to run all the attacks at the same time with safe
values.

## Framework information

This section contains all of the attacks, model architectures that come built-in in the framework.

### Attacks

| Attack name     | Description |
| :--------       | :----       |
| CopyCat         | Performs the attack by learning new model from a dataset that is labeled by the target model.|
| ActiveThief     | Performs the attack by learning new model from a dataset that is labeled by the target model. To minimize the number of queries required to learn the new model, active learning techniques are used.|

More details on the attacks can be found on [wiki][4].

[2]: https://arxiv.org/pdf/1711.01768.pdf
[3]: https://arxiv.org/pdf/1806.05476.pdf

### Model architectures

| Model architecture   | Description |
| :--------            | :----       |
| SimpleNet-V1         | Convolution network based on this [paper][5]  |
| VGGNet               | Modified version of pytorch built-in VGGNet that takes in input of arbitrary resolution |

[5]: https://arxiv.org/abs/1608.06037

[5]: TODO
## References

add citations
