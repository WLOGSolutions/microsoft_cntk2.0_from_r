# Showcase: calling [Microsoft Cognitive Toolkit (CNTK) 2.0](https://github.com/Microsoft/CNTK) deep learning library fromwithin R using [reticulate](https://github.com/rstudio/reticulate) package and [Azure DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-virtual-machine-overview). #

## Introduction ##

On 2017/06/01 Microsoft released its Microsoft Cognitive toolkit (CNTK) 
(check [CNTK release page](https://github.com/Microsoft/CNTK/releases/tag/v2.0)). 
The toolkit is very interesting and one can read more about it [here](https://docs.microsoft.com/en-us/cognitive-toolkit/). 
Especially chapter [Reasons to Switch from TensorFlow to CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/reasons-to-switch-from-tensorflow-to-cntk).

Unfortunately CNTK does not have bindings to R. Fortunately it has to Python. I am very big fan of R and wanted 
to have CNTK available from this great tool. I present shortly R version of [SimpleMNIST.py](https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Classification/MLP/Python/SimpleMNIST.py). This example shows how to apply MLP (infamous) [MNIST](http://yann.lecun.com/exdb/mnist) dataset of handwritten digits.

## Preparing environment ##
I was working on [Azure DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-virtual-machine-overview) using [NV6 Instance](https://azure.microsoft.com/en-us/blog/azure-n-series-general-availability-on-december-1/) with Tesla M60 GPU and Ubuntu OS 16.04.2 LTS.

### Creating anaconda environment ###

I like to have clean workspace. The good practice is to create a new environment for a project. I issued the following
instructions

```sh
conda create -n cntk2.0 python=3.5
```

This gave me a new environment with python 3.5.

### Installing CNTK ###

I selected appropriate CNTK version from this [list](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python)
I used version for Python 3.5 with GPU and 1-SGD compiled ([link])

```sh
source activate cntk2.0
pip install https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.0-cp35-cp35m-linux_x86_64.whl
source deactivate cntk2.0
```

After those commands I had the newest CNTK installed in Anaconda environment called `cntk2.0`.

### Installing R packages ###

I have used:

* [R](https://www.r-project.org/) in version 3.3.2 that was shipped with Azure DSVM.
* [R Suite](https://github.com/WLOGSolutions/RSuite) in version [0.9-211](https://github.com/WLOGSolutions/RSuite/releases/tag/211)

To install packages and build cntkR package run following commands
```sh
rsuite proj depsinst
rsuite proj build
```

### Downloading datasets ###

From CNTK github [examples repo](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/DataSets/MNIST) I copied 
two python scripts that download and prepare datasets. To download and prepare datasets just issue the instruction.

```sh
cd python
python install_mnist.py
```

In folder `data` you should find two files `Test-28x28_cntk_text.txt` and `Train-28x28_cntk_text.txt`. 

## Running CNTK from within R ##

Scipt `mnist.R` contains my version of `SimpleMNIST.py`. Before running the script you should modify line 2 in file `config_templ.txt` to point to your python installation. In the version I prepared this line looks like this

```
python_path: ~/.conda/envs/cntk2.0/bin
```

This path should work for you if you are following my post using Azure DSVM.

Finally, we can run our R script using instruction

```sh
Rscript R/mnist.R
```

Below you can see fitting process in progress - it takes around `0.7-1.0s` to perform one epoch (60 000 steps).

![cntk_R_console.PNG](https://github.com/WLOGSolutions/microsoft_cntk2.0_from_r/blob/master/img/cntkr_R_console.PNG)

And here we present output of `nvidia-smi` taken during the fitting process

![cntk_R_nvidia_smi.PNG](https://github.com/WLOGSolutions/microsoft_cntk2.0_from_r/blob/master/img/cntkr_nvidia_smi.PNG)


It takes around 23 seconds to build an MLP network and score it on test dataset. It reported 
`2.3%` classification error, which is not bad but it was not most important for this showcase.

### Power of GPU ###

Script `mnist.R` accepts a parameter `device` that can take two values:

* `cpu` - run computation on CPU (default)
* `gpu` - run copuation on GPU (id = 0)

On my machine GPU was Tesla M60 GPU. When you switch to `cpu()` you will notice around **30x** slowdown!

Even more performance gain you can see calling `mnist_conv.R` script that implements Convolution Network that is more 
computing intensive (check [ConvNet_MNIST.py](https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Classification/ConvNet/Python/ConvNet_MNIST.py) forPython version). Below we present output of running this model using GPU. It takes around `3.7sec` for one epoch (60 000 steps). 
**Running this model on CPU is hopeless.**

![cntk_R_console_2.PNG](https://github.com/WLOGSolutions/microsoft_cntk2.0_from_r/blob/master/img/cntkr_R_console_2.PNG)

# Conclusion #

I have been using R for large scale analytical solutions for 12 years. It is a great analytics oriented glue-language. I gives me 
access to different powerful tools like [H2O.AI](h2o.ai) or [Apache Spark](https://spark.apache.org/). Unfortunately
sometimes R API is missing. Fortunately if there is Python API we can benefit from using great 
[reticulate](https://github.com/rstudio/reticulate) package from [R Studio](https://www.rstudio.com/). The package worked 
smoothly with CNTK and Python 3.5. I was really astonished and I am happy I can now benefit from both R and Python toolboxes.
