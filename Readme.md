# cnld

### Installation

In Linux, setup environment and clone repository.
``` sh
apt-get install -y build-essential libblas-dev liblapack-dev gfortran libomp-dev
git clone -b refactor https://github.com/bdshieh/cnl-dyna.git
```

Install conda packages
``` sh
conda install numpy scipy matplotlib cython pandas jupyterlab nomkl setuptools pip openblas
conda install -c conda-forge tqdm namedlist openmp
```
    
Install cnl-dyna
``` sh
pip install .
```
Pip will attempt to make H2Lib and compile cython C-extensions. The API can be accessed in Python code from the cnld.api module.

### Docker 

``` sh
sudo docker pull bdshieh/cnl-dyna
sudo docker run -p 8888:8888 bdshieh/cnl-dyna 
```
which should launch a Jupyter server. Click the link provided to open Jupyter in your browser.
