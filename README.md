# Showcase: calling [Microsoft Cognitive Toolkit (CNTK) 2.0](https://github.com/Microsoft/CNTK) deep learning library fromwithin R using [reticulate](https://github.com/rstudio/reticulate) package and [Azure DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-virtual-machine-overview). #

## Introduction ##

On 2017/06/01 Microsoft released its Microsoft Cognitive toolkit (CNTK) 
(check [CNTK release page](https://github.com/Microsoft/CNTK/releases/tag/v2.0)). 
The toolkit is very interesting and one can read more about it [here](https://docs.microsoft.com/en-us/cognitive-toolkit/). 
Especially chapter [Reasons to Switch from TensorFlow to CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/reasons-to-switch-from-tensorflow-to-cntk).

Unfortunately CNTK does not have bindings to R. Fortunately it has to Python. I am very big fan of R and wanted 
to have CNTK available from this great tool. I present shortly R version of [SimpleMNIST.py](https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Classification/MLP/Python/SimpleMNIST.py). This example shows how to apply MLP (infamous) [MNIST](http://yann.lecun.com/exdb/mnist) dataset of handwritten digits.

## Preparing environment ##
I was working on [Azure DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-virtual-machine-overview) using [NV6 Instance](https://azure.microsoft.com/en-us/blog/azure-n-series-general-availability-on-december-1/) with Tesla M60 GPU.

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

I used GNU R version 3.3.2 that was shipped with Azure DSVM. To install packages run the following code from repository folder.

```sh
Rscript.exe install_packages.R
```

It will install necessary R packages in local folder `lib`. It should not interferre with your global R configuration.

### Downloading datasets ###

From CNTK github [examples repo](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/DataSets/MNIST) I copied 
two python scripts that download and prepare datasets. To download and prepare datasets just issue the instruction.

```sh
python install_mnist.py
```

In folder `data` you should find two files `Test-28x28_cntk_text.txt` and `Train-28x28_cntk_text.txt`. 

## Running CNTK from within R ##

Scipt `mnist.R` contains my version of `SimpleMNIST.py`. Before running the script you should modify line 8 to point to your
python installation. In the version I prepared this line looks like this

```r
my_python <- "~/.conda/envs/cntk2.0/bin"
```

This path should work for you if you are following my post using Azure DSVM.

Finally, we can run our R script using instruction

```sh
Rscript mnist.R
```

It takes around 23 seconds to build an MLP network and score it on test dataset. It reported 
`2.3%` classification error, which is not bad but it was not most important for this showcase.

### Power of GPU ###

In script `mnist.R` in line 45 I set default device to be used. By default it is `gpu(0L)` which means Tesla M60 GPU. When you 
switch to `cpu()` you will notice around **32x** slowdown!

# Conclusion #

I have been using R for large scale analytical solutions for 12 years. It is a great analytics oriented glue-language. I gives me 
access to different powerful tools like [H2O.AI](h2o.ai) or [Apache Spark](https://spark.apache.org/). Unfortunately
sometimes R API is missing. Fortunately if there is Python API we can benefit from using great 
[reticulate](https://github.com/rstudio/reticulate) package from [R Studio](https://www.rstudio.com/). The package worked 
smoothly with CNTK and Python 3.5. I was really astonished and I am happy I can now benefit from both R and Python toolboxes.
