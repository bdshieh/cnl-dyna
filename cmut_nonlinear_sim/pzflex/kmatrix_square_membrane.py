'''
'''
import numpy as np
import subprocess, os
from scipy.io import loadmat
from scipy.interpolate import interp2d
from scipy.linalg import inv
import argparse

from cmut_nonlinear_sim import abstract

# define configuration for this script
_Config = {}
_Config['x_len'] = 40e-6
_Config['y_len'] = 40e-6
_Config['thickness'] = 2e-6
_Config['x_load'] = 5e-6
_Config['y_load'] = 5e-6
_Config['dx'] = 1e-6
_Config['dy'] = 1e-6
_Config['xymin'] = 40e-6 / 30
_Config['zin'] = 2e-6 / 8
Config = abstract.register_type('Config', _Config)


pzscript = '''
c ** GEOMETRY **
symb xx1 = -$x_len / 2
symb xx2 = $x_load - $dx / 2
symb xx3 = $x_load + $dx / 2
symb xx4 = $x_len / 2
symb #keycordt x 1 $xx1 $xx2 $xx3 $xx4

symb yy1 = -$y_len / 2
symb yy2 = $y_load - $dy / 2
symb yy3 = $y_load + $dy / 2
symb yy4 = $y_len / 2
symb #keycordt y 1 $yy1 $yy2 $yy3 $yy4

symb zz1 = 0
symb zz2 = $thickness / 2
symb zz3 = $thickness
symb #keycordt z 1 $zz1 $zz2 $zz3

symb #get { idx } rootmax x
symb #get { jdx } rootmax y
symb #get { kdx } rootmax z

symb #keyindx i 1 $idx 1 $xymin 1
symb #keyindx j 1 $jdx 1 $xymin 1
symb #keyindx k 1 $kdx 1 $zin 1 
symb indgrd = $i$idx
symb jndgrd = $j$jdx
symb kndgrd = $k$kdx

grid $indgrd $jndgrd $kndgrd
geom keypnt $idx $jdx $kdx

c ** MATERIALS **
matr
	type elas
	wvsp on
	prop sin 2200 11000 6250
c 	mdmp sin 1e6 db 4000

c ** MATERIAL ASSIGNMENT **
site regn sin

c ** BOUNDARY CONDITIONS **
boun
	side xmin fixd
	side xmax fixd
	side ymin fixd
	side ymax fixd

c ** PRESSURE LOADING **
c func astep 1 * 0 0 cos 1 100e-9 0 *
func step 1 0 0 0

plod
	pdef pld1 func 1e9
	vctr vct1 0 0 1
	sdef pld1 vct1 $i2 $i3 $j2 $j3 $k2 $k2

c ** CALCULATIONS **
calc disp

c ** SOLVING **
prcs

symb #get { step } timestep 
symb tstop = 20e-6
symb nstep = $tstop / $step

c static solver with auto dynamic relaxation 
drlx auto
exec $nstep $tstop stat

c ** DATA OUT **
data
	file out zdsp.mat 
	form out matlab
	out zdsp
'''

def symb(f, var, val):
    f.write(f'symb {str(var)} = {val} \n')


def symbs(f, **kwargs):
    for k, v in kwargs.items():
        symb(f, k, v)


def gen_flxinp(**args):
    with open('pzmodel.flxinp', 'w+') as f:
        symbs(f, **args)
        f.write(pzscript)


def run_pzflex():
    subprocess.call(['pzflex', 'pzmodel.flxinp'])


def postproc(verts):
    ''''''
    m = loadmat('zdsp.mat')
    for k in m.keys():
        if k[-4:] == 'xcrd':
            xcrdk = k
        elif k[-4:] == 'ycrd':
            ycrdk = k
        elif k[-4:] == 'zdsp':
            zdspk = k
    
    xv = m[xcrdk]
    yv = m[ycrdk]
    zdsp = m[zdspk]
    zdsp = zdsp.mean(axis=-1)

    fi = interp2d(xv, yv, zdsp)
    dsp = []
    for x, y, z in verts:
        dsp.append(fi(x, y))
    
    return np.array(dsp) 
        

def main():
    pass
    # define and parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mesh_file')
    # parser.add_argument('file')
    # parser.add_argument('xl', type=float)
    # parser.add_argument('yl', type=float)
    # parser.add_argument('refn', nargs=2)
    # args = vars(parser.parse_args())

    # mesh_file = args['mesh_file']

    # with np.load(mesh_file) as npf:
    #     refns = npf['refns']
    #     verts = npf['verts']
    
    
    # pzargs = {}
    # pzargs['x_len'] = 40e-6
    # pzargs['y_len'] = 40e-6
    # pzargs['thickness'] = 2e-6
    # pzargs['x_load'] = 5e-6
    # pzargs['y_load'] = 5e-6
    # pzargs['dx'] = 1e-6
    # pzargs['dy'] = 1e-6
    # pzargs['xymin'] = 40e-6 / 30
    # pzargs['zin'] = 2e-6 / 8

    # gen_flxinp(**pzargs)
    # run_pzflex()
    # zdsp_row = postproc()
    # os.remove('pzmodel.flxinp')
    # os.remove('zdsp.mat')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # define subparser
    subparsers = parser.add_subparsers()
    config_parser = subparsers.add_parser('config')
    config_parser.add_argument('type')
    config_parser.add_argument('file')

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('config')
    run_parser.add_argument('file')
