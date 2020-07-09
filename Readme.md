# cnld

### Installation

``` sh
apt-get install -y build-essential libblas-dev liblapack-dev gfortran libomp-dev
git clone https://github.com/bdshieh/cnl-dyna.git
cp $cnld/H2Lib
make
cp libh2.a $cnld/lib
cd $cnld
pip install .
```