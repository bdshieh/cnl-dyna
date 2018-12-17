'''
'''
import numpy as np
import subprocess, os
from scipy.io import loadmat
from scipy.interpolate import interp2d
from scipy.linalg import inv
from tqdm import tqdm


# base PZFlex script
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
symb #keyindx k 1 $kdx 1 $zmin 1 
symb indgrd = $i$idx
symb jndgrd = $j$jdx
symb kndgrd = $k$kdx

grid $indgrd $jndgrd $kndgrd
geom keypnt $idx $jdx $kdx

c ** MATERIALS **
c symb cbulk sqrt($E * (1 - $v) / ($rho * (1 + $v) * (1 - 2 * $v)))
c symb cshear sqrt($E / (2 * (1 + $v)) / $rho)
matr
	type elas
	wvsp on
    prop sin 2040 7846 4700
c   prop sin $rho $cbulk $cshear
c	prop sin 2200 11000 6250
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
	pdef pld1 func 1
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

def gen_flxinp(**args):
    '''Generate complete PZFlex script'''
    with open('pzmodel.flxinp', 'w+') as f:
        # write memory allocation for job
        f.write('mem 10 10\n')
        # write (prepend) symbols
        for k, v in args.items():
            f.write(f'symb {str(k)} = {v} \n')
        # write base script
        f.write(pzscript)


def run_pzflex():
    '''Run PZFlex'''
    # subprocess.call(['pzflex', 'pzmodel.flxinp'])
    subprocess.check_call(['flex_s', 'pzmodel.flxinp'], stdout=subprocess.DEVNULL)
    # subprocess.check_call(['winpty', '-Xallow-non-tty', 'pzflex.bat', 'pzmodel.flxinp'], stdout=subprocess.DEVNULL)


def postproc(vert, on_bound):
    '''Postprocess PZFlex output'''
    # load data from mat file
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

    # take average of z-disp along mesh z-axis
    zdsp = zdsp.mean(axis=-1)

    # interpolate z-disp at BE mesh vertices
    fi = interp2d(yv, xv, zdsp)
    dsp = np.zeros(len(vert))
    for i, (x, y, z) in enumerate(vert):
        # skip boundary vertices
        if on_bound[i]:
            continue
        dsp[i] = fi(y, x)
    
    return np.array(dsp) 
        

def main(cfg, args):

    with np.load(cfg.mesh_file) as npf:
        refns = npf['refns']
        verts = npf['verts']
        on_bounds = npf['on_bounds']
    
    pzargs = {}
    pzargs['x_len'] = cfg.x_len
    pzargs['y_len'] = cfg.y_len
    pzargs['thickness'] = cfg.thickness
    pzargs['dx'] = cfg.dx
    pzargs['dy'] = cfg.dy
    pzargs['xymin'] = cfg.xymin
    pzargs['zmin'] = cfg.zmin

    # iterate over different meshes/mesh refn
    Ks = []
    with tqdm(total=sum([len(vert) for vert in verts])) as pbar:
        for refn, vert, on_bound in zip(refns, verts, on_bounds):

            Kinv = np.zeros((len(vert), len(vert)))
            # apply loading onto each vertex
            for i, (x, y, z) in enumerate(vert):
                pbar.update()
                # skip boundary vertices
                if on_bound[i]:
                    continue

                # set load coordinates
                pzargs['x_load'] = x
                pzargs['y_load'] = y
                # generate script
                gen_flxinp(**pzargs)
                # run pzflex
                run_pzflex()
                # postprocess results
                Kinv[:,i] = postproc(vert, on_bound)

                # delete files
                os.remove('pzmodel.flxinp')
                os.remove('pzmodel.flxprt')
                os.remove('pzmodel.flxrsto')
                os.remove('pzmodel.flxwrn')
                os.remove('zdsp.mat')
            
            K = np.zeros((len(vert), len(vert)))
            K[np.ix_(~on_bound, ~on_bound)] = inv(Kinv[~on_bound, :][:, ~on_bound])
            # np.save('K.npy', K)
            Ks.append(K)

    np.savez(args.file, verts=verts, on_bounds=on_bounds, refns=refns, Ks=Ks)


if __name__ == '__main__':

    import sys
    from cnld import util

    # define configuration for this script
    Config = {}
    Config['mesh_file'] = ''
    Config['x_len'] = 40e-6
    Config['y_len'] = 40e-6
    Config['thickness'] = 2e-6
    Config['dx'] = 1e-6
    Config['dy'] = 1e-6
    Config['xymin'] = 40e-6 / 30
    Config['zmin'] = 2e-6 / 8

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)
