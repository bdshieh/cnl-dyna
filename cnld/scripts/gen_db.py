''' 
Generates patch-to-patch impulse responses (in time domain) database for an array of CMUT membranes.
'''
import numpy as np
import multiprocessing
from itertools import repeat
from tqdm import tqdm
import os, sys, traceback
from scipy.sparse.linalg import lgmres
from timeit import default_timer as timer

from cnld import abstract, util, bem, fem, frequency_response, impulse_response, database
from cnld.compressed_formats import MbkSparseMatrix
from cnld.mesh import Mesh


''' PROCESS FUNCTIONS '''

def init_process(_write_lock, _cfg, _file):
    global write_lock, cfg, file
    write_lock = _write_lock
    cfg = Config(**abstract.loads(_cfg))
    file = _file


def process(job):
    ''''''
    job_id, (f, k) = job

    # get options and parameters
    c = cfg.sound_speed
    rho = cfg.fluid_rho
    array = abstract.load(cfg.array_config)
    refn = cfg.mesh_refn

    # create finite element matrix
    Gfe = fem.array_mbk_spmatrix(array, refn, f)

    # create boundary element matrix
    hmkwrds = ['format', 'aprx', 'basis', 'admis', 'eta', 'eps', 'm', 'clf', 
        'eps_aca', 'rk', 'q_reg', 'q_sing', 'strict']
    hmargs = { k:getattr(cfg, k) for k in hmkwrds }
    # Z = bem.z_from_abstract(array, k, refn, format='HFormat', **hmargs)
    Z = bem.array_z_matrix(array, refn, k, **hmargs)
    omg = 2 * np.pi * f
    Gbe = -omg**2 * 2 * rho * Z

    # define total linear system and find its LU decomposition
    G = MbkSparseMatrix(Gfe) + Gbe
    Glu = G.lu()

    # create patch pressure loads
    F = fem.array_f_spmatrix(array, refn)
    AVG = fem.array_avg_spmatrix(array, refn)
    mesh = Mesh.from_abstract(array, refn)
    ob = mesh.on_boundary

    # solve for each source patch
    npatch = abstract.get_patch_count(array)
    source_patch = np.arange(npatch)
    dest_patch = np.arange(npatch)
    # patches = abstract.get_patches_from_array(array)
    # patch_areas = np.array([p.area for p in patches])

    for sid in source_patch:
        # get RHS
        b = np.array(F[:, sid].todense())

        # solve
        start = timer()
        # conjugate so phase is consistent with -iwt convention used by h2lib
        x = np.conj(Glu.lusolve(b))
        time_solve = timer() - start
        x[ob] = 0

        # average displacement over patches
        x_patch = (AVG.T).dot(x)

        # write patch displacement results to frequency response database
        data = {}
        data['frequency'] = repeat(f)
        data['wavenumber'] = repeat(k)
        data['source_patch'] = repeat(sid)
        data['dest_patch'] = dest_patch
        data['displacement_real'] = np.real(x_patch)
        data['displacement_imag'] = np.imag(x_patch)
        data['time_solve'] = repeat(time_solve)

        with write_lock:
            database.append_patch_to_patch_freq_resp(file, **data)

        # write node displacement results to frequency response database
        data = {}
        data['node_id'] = range(len(mesh.vertices))
        data['x'] = mesh.vertices[:,0]
        data['y'] = mesh.vertices[:,1]
        data['z'] = mesh.vertices[:,2]
        data['frequency'] = repeat(f)
        data['wavenumber'] = repeat(k)
        data['source_patch'] = repeat(sid)
        data['displacement_real'] = np.real(x)
        data['displacement_imag'] = np.imag(x)

        with write_lock:
            database.append_patch_to_node_freq_resp(file, **data)
    
    with write_lock:
        util.update_progress(file, job_id)
    

def run_process(*args, **kwargs):
    try:
        return process(*args, **kwargs)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def postprocess(file, interp):

    # postprocess and convert frequency response to impulse response
    freqs, ppfr = database.read_patch_to_patch_freq_resp(file)
    t, ppir = impulse_response.fft_to_fir(freqs, ppfr, interp=interp, axis=-1, use_kkr=True)
    source_patches, dest_patches, times = np.meshgrid(np.arange(ppir.shape[0]), 
        np.arange(ppir.shape[1]), t, indexing='ij')

    data = {}
    data['source_patch'] = source_patches.ravel()
    data['dest_patch'] = dest_patches.ravel()
    data['time'] = times.ravel()
    data['displacement'] = ppir.ravel()
    database.append_patch_to_patch_imp_resp(file, **data)


''' ENTRY POINT '''

def main(cfg, args):
    ''''''
    # get parameters from config and args
    file = args.file
    write_over = args.write_over
    threads = args.threads if args.threads else multiprocessing.cpu_count()
    f_start, f_stop, f_step = cfg.freqs
    c = cfg.sound_speed

    # calculate job-related values
    freqs = np.arange(f_start, f_stop + f_step, f_step)
    wavenums = 2 * np.pi * freqs / c
    is_complete = None
    njobs = len(freqs)
    ijob = 0

    # check for existing file
    if os.path.isfile(file):
        if write_over:  # if file exists, write over
            # remove existing files
            os.remove(file)  

            # create databases
            database.create_db(file)
            util.create_progress_table(file, njobs)

            # append node information
            amesh = Mesh.from_abstract(abstract.load(cfg.array_config), refn=cfg.mesh_refn)
            nodes = amesh.vertices
            database.append_node(file, node_id=range(len(nodes)), x=nodes[:,0], y=nodes[:,1], 
                z=nodes[:,2])

        else: # continue from current progress
            is_complete, ijob = util.get_progress(file)
            if np.all(is_complete): return
    else:
        # make directories if they do not exist
        file_dir = os.path.dirname(os.path.abspath(file))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create databases
        database.create_db(file)
        util.create_progress_table(file, njobs)

        # append node information
        amesh = Mesh.from_abstract(abstract.load(cfg.array_config), refn=cfg.mesh_refn)
        nodes = amesh.vertices
        database.append_node(file, node_id=range(len(nodes)), x=nodes[:,0], y=nodes[:,1], 
            z=nodes[:,2])

    # start multiprocessing pool and run process
    try:
        write_lock = multiprocessing.Lock()

        pool = multiprocessing.Pool(threads, initializer=init_process, initargs=(write_lock, 
            abstract.dumps(cfg), file), maxtasksperchild=1)

        jobs = util.create_jobs((freqs, 1), (wavenums, 1), mode='zip', is_complete=is_complete)
        result = pool.imap_unordered(run_process, jobs, chunksize=1)

        for r in tqdm(result, desc='Running', total=njobs, initial=ijob):
            pass

        postprocess(file, cfg.freq_interp)

    except Exception as e:
        print(e)
    finally:
        pool.close()
        pool.terminate()


# define default configuration for this script
_Config = {}
_Config['freqs'] = 0, 50e6, 200e3
_Config['sound_speed'] = 1500.
_Config['fluid_rho'] = 1000.
_Config['array_config'] = ''
_Config['mesh_refn'] = 9
_Config['format'] = 'HFormat'
_Config['aprx'] = 'paca'
_Config['basis'] = 'linear'
_Config['admis'] = '2'
_Config['eta'] = 1.2
_Config['eps'] = 1e-12
_Config['m'] = 4
_Config['clf'] = 16
_Config['eps_aca'] = 1e-2
_Config['rk'] = 0
_Config['q_reg'] = 2
_Config['q_sing'] = 4
_Config['strict'] = True
_Config['freq_interp'] = 2
Config = abstract.register_type('Config', _Config)


if __name__ == '__main__':

    import sys
    from cnld import util

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)


