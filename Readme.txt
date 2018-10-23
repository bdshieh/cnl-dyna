

Other dependencies:

For Windows:

    C++ compiler
    https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017

    MATLAB API for Python
    https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html
    cd matlabroot\extern\engines\python
    python setup.py build --build-base='builddir' install --user

For Linux:

    MATLAB API for Python
    https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html
    cd matlabroot\extern\engines\python
    python setup.py build --build-base='builddir' install --user


To install:

    cd interaction3
    pip install .


Field II Copyright Information (interaction3 / mfield / core)

The executable program is copyrighted freeware by Jørgen Arendt Jensen. It may be distributed freely unmodified. It is, however, not permitted to utilize any part of the software in commercial products without prior written consent of Jørgen Arendt Jensen.

[1] J.A. Jensen: Field: A Program for Simulating Ultrasound Systems, Paper presented at the 10th Nordic-Baltic Conference on Biomedical Imaging Published in Medical & Biological Engineering & Computing, pp. 351-353, Volume 34, Supplement 1, Part 1, 1996.

[2] J.A. Jensen and N. B. Svendsen: Calculation of pressure fields from arbitrarily shaped, apodized, and excited ultrasound transducers, IEEE Trans. Ultrason., Ferroelec., Freq. Contr., 39, pp. 262-267, 1992.