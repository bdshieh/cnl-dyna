FROM continuumio/miniconda3

LABEL maintainer='bernard.shieh@eng.ox.ac.uk'

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \ 
    vim \
    nano \
    wget \
    bzip2 \ 
    git \
    build-essential \ 
    # libblas-dev \
    liblapack-dev \
    gfortran \
    # libomp-dev \
    && apt-get clean

# ENV PATH /opt/conda/bin:$PATH

RUN conda update -y conda

RUN conda install -y \
    numpy \
    scipy \
    matplotlib \
    cython \
    pandas \
    jupyterlab \
    nomkl \  
    setuptools \
    pip \
    openblas

RUN conda install -y -c conda-forge \
    tqdm \
    namedlist \
    openmp

RUN conda clean -ya

# clone cnl-dyna
RUN cd /opt/ && \
    git clone -b refactor https://github.com/bdshieh/cnl-dyna.git

# install cnl-dyna
RUN cd /opt/cnl-dyna/ && \
    pip install .

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

WORKDIR /opt/