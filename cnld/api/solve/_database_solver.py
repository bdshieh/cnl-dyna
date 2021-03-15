''''''
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import repeat
from time import default_timer as timer
import os
from cnld import fem, bem, database, util, impulse_response
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


class DatabaseSolver:
    '''
    '''

    def __init__(self,
                 file=None,
                 layout=None,
                 grids=None,
                 freqs=None,
                 freq_interp=2,
                 fluid_c=1500,
                 fluid_rho=1000,
                 refn=7,
                 overwrite=False,
                 threads=None,
                 notebook=False,
                 **kwargs):

        self._layout = layout
        self._grids = grids
        self.file = file
        self.freqs = freqs
        self._freq_interp = freq_interp
        self._fluid_c = fluid_c
        self._fluid_rho = fluid_rho
        self._refn = refn
        self._overwrite = overwrite
        self._threads = threads
        self._notebook = notebook

        self.hmargs = kwargs

    @property
    def file(self):
        self._file

    @file.setter
    def set_file(self, f):
        self._file = os.path.normpath(f)

    @property
    def layout(self):
        return self._layout

    @property
    def grids(self):
        return self._grids

    @property
    def freqs(self):
        return self._freqs

    @freqs.setter
    def set_freqs(self, *args):
        if len(args) == 3:
            self._freqs = np.arange(*args)
        elif len(args) == 1:
            self.freqs = np.array(args[0])
        else:
            raise TypeError

    @property
    def freq_interp(self):
        return self._freq_interp

    @property
    def fluid_c(self):
        return self._fluid_c

    @property
    def fluid_rho(self):
        return self._fluid_rho

    @property
    def refn(self):
        return self._refn

    @property
    def overwrite(self):
        return self._overwrite

    @property
    def notebook(self):
        return self._notebook

    @property
    def threads(self):
        return self._threads

    @threads.setter
    def set_threads(self, th):
        if th is None:
            self._threads = multiprocessing.cpu_count()
        else:
            self._threads = th

    @property
    def hmargs(self):
        return self._hmargs

    @hmargs.setter
    def set_hmargs(self, args):
        hmargs = dict('linear',
                      m=4,
                      q_reg=2,
                      q_sing=4,
                      aprx='paca',
                      admis='2',
                      eta=1.0,
                      eps_aca=1e-2,
                      strict=False,
                      clf=16,
                      rk=0)
        for k, v in args.items():
            if k in hmargs:
                hmargs[k] = v

        self._hmargs = hmargs

    @staticmethod
    def _init_process(_write_lock):

        global write_lock
        write_lock = _write_lock

    def _process(self, job):
        '''
        Process which executes a job.
        '''
        # get options and parameters
        # c = self.fluid_c
        rho = self.fluid_rho
        layout = self.layout
        grids = self.grids
        hmargs = self.hmargs
        file = self.file

        job_id, f, k = job
        omg = 2 * np.pi * f
        # k = omg / c

        # generate fem lhs matrix
        Gfem = fem.mbk_mat_spm_from_layout(layout, grids, f)

        # generate bem lhs matrix
        Z = bem.z_mat_hm_from_layout(layout, grids, k, **hmargs)
        Gbem = -omg**2 * 2 * rho * Z

        # define total lhs and find LU decomposition
        G = Gfem + Gbem
        Glu = G.lu()

        # create patch pressure loads
        P = fem.p_mat_sps_from_layout(layout, grids)
        AVG = fem.avg_mat_sps_from_layout(layout, grids)
        ob = grids.bem.on_boundary

        # solve for each source patch
        nctrldom = len(layout.controldomainlist)
        cd_source = np.arange(nctrldom)
        cd_dest = np.arange(nctrldom)

        for sid in cd_source:
            # get RHS
            b = np.array(P[:, sid].todense())

            # solve
            start = timer()

            # conjugate so phase is consistent with -iwt convention
            # used by h2lib
            x = np.conj(Glu.lusolve(b))

            time_solve = timer() - start
            x[ob] = 0

            # average displacement over patches
            x_cd = AVG.T @ x

            # write patch displacement results to frequency response database
            data = {}
            data['frequency'] = repeat(f)
            data['wavenumber'] = repeat(k)
            data['source_patch'] = repeat(sid)
            data['dest_patch'] = cd_dest
            data['displacement_real'] = np.real(x_cd)
            data['displacement_imag'] = np.imag(x_cd)
            data['time_solve'] = repeat(time_solve)

            with write_lock:
                database.append_patch_to_patch_freq_resp(file, **data)

            # write node displacement results to frequency response database
            data = {}
            data['node_id'] = range(len(grids.bem.vertices))
            data['x'] = grids.bem.vertices[:, 0]
            data['y'] = grids.bem.vertices[:, 1]
            data['z'] = grids.bem.vertices[:, 2]
            data['frequency'] = repeat(f)
            data['wavenumber'] = repeat(k)
            data['source_patch'] = repeat(sid)
            data['displacement_real'] = np.real(x)
            data['displacement_imag'] = np.imag(x)

            with write_lock:
                database.append_patch_to_node_freq_resp(file, **data)

        with write_lock:
            database.update_progress(file, job_id)

    def postprocess(self):
        '''
        Postprocess generated database by calculating impulse responses from
        frequency data.
        '''
        file = self.file
        freq_interp = self.freq_interp

        # postprocess and convert frequency response to impulse response
        freqs, ppfr = database.read_patch_to_patch_freq_resp(file)
        t, ppir = impulse_response.fft_to_fir(freqs,
                                              ppfr,
                                              interp=freq_interp,
                                              axis=-1,
                                              use_kkr=True)
        source_patches, dest_patches, times = np.meshgrid(
            np.arange(ppir.shape[0]),
            np.arange(ppir.shape[1]),
            t,
            indexing='ij')

        data = {}
        data['source_patch'] = source_patches.ravel()
        data['dest_patch'] = dest_patches.ravel()
        data['time'] = times.ravel()
        data['displacement'] = ppir.ravel()
        database.append_patch_to_patch_imp_resp(file, **data)

    def _create_new_db(self):

        file = self.file
        grids = self.grids
        freqs = self.freqs

        njobs = len(freqs)

        database.create_db(file)
        database.create_progress_table(file, njobs)

        # append node information
        nodes = grids.bem.vertices
        database.append_node(file,
                             node_id=range(len(nodes)),
                             x=nodes[:, 0],
                             y=nodes[:, 1],
                             z=nodes[:, 2])

    def _create_or_load_db(self):

        file = self.file
        overwrite = self.overwrite

        is_complete = None
        ijob = 0

        # check for existing file
        if os.path.isfile(file):
            if overwrite:  # if file exists, write over
                # remove existing files
                os.remove(file)

                # create databases
                self._create_new_db()

            else:  # continue from current progress
                is_complete, ijob = database.get_progress(file)

        else:
            # make directories if they do not exist
            file_dir = os.path.dirname(os.path.abspath(file))
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # create databases
            self._create_new_db()

        return is_complete, ijob

    def solve(self):
        '''
        '''
        # get parameters from config and args
        threads = self.threads
        freqs = self.freqs
        c = self.fluid_c
        notebook = self.notebook

        # create or load existing database
        is_complete, ijob = self._create_or_load_db()

        # calculate job-related values
        wavenums = 2 * np.pi * freqs / c
        jobs = util.create_jobs((freqs, 1), (wavenums, 1),
                                mode='zip',
                                is_complete=is_complete)
        njob = len(freqs)

        # start multiprocessing pool and run process
        write_lock = multiprocessing.Lock()

        with ProcessPoolExecutor(threads,
                                 initializer=DatabaseSolver._init_process,
                                 initargs=(write_lock,),
                                 maxtasksperchild=1) as executor:

            result = executor.map(self._process, jobs, chunksize=1)

            if notebook:
                for r in tqdm_notebook(result,
                                       desc='Running',
                                       total=njob,
                                       initial=ijob):
                    pass
            else:
                for r in tqdm(result, desc='Running', total=njob, initial=ijob):
                    pass

            self.postprocess()