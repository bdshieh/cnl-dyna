# cnld

### Installation

In Linux, setup environment and clone repository.
``` sh
apt-get install -y build-essential libblas-dev liblapack-dev gfortran libomp-dev
git clone https://github.com/bdshieh/cnl-dyna.git
```

Run in python environment (Conda recommended).
``` sh
pip install .
```
Pip will attempt to make H2Lib and compile cython C-extensions. The API can be accessed in Python code from the cnld.api module.
