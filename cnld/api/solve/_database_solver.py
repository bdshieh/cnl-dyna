''''''
__all__ = ['DatabaseSolver']
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import repeat
from timeit import default_timer as timer
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

        self.layout = layout
        self.grids = grids
        self.file = file
        self.freqs = freqs
        self.freq_interp = freq_interp
        self.fluid_c = fluid_c
        self.fluid_rho = fluid_rho
        self.refn = refn
        self.overwrite = overwrite
        self.threads = threads
        self.notebook = notebook

        self.hmargs = kwargs

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, f):
        if f is not None:
            self._file = os.path.normpath(f)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, arg):
        self._layout = arg

    @property
    def grids(self):
        return self._grids

    @grids.setter
    def grids(self, arg):
        self._grids = arg

    @property
    def freqs(self):
        return self._freqs

    @freqs.setter
    def freqs(self, arg):
        if arg is not None:
            if len(arg) == 3:
                self._freqs = np.arange(*arg)
            elif len(arg) == 1:
                self._freqs = np.array(arg)
            else:
                raise TypeError

    @property
    def freq_interp(self):
        return self._freq_interp

    @freq_interp.setter
    def freq_interp(self, arg):
        self._freq_interp = arg

    @property
    def fluid_c(self):
        return self._fluid_c

    @fluid_c.setter
    def fluid_c(self, arg):
        self._fluid_c = arg

    @property
    def fluid_rho(self):
        return self._fluid_rho

    @fluid_rho.setter
    def fluid_rho(self, arg):
        self._fluid_rho = arg

    @property
    def refn(self):
        return self._refn

    @refn.setter
    def refn(self, arg):
        self._refn = arg

    @property
    def overwrite(self):
        return self._overwrite

    @overwrite.setter
    def overwrite(self, arg):
        self._overwrite = arg

    @property
    def notebook(self):
        return self._notebook

    @notebook.setter
    def notebook(self, arg):
        self._notebook = arg

    @property
    def threads(self):
        return self._threads

    @threads.setter
    def threads(self, th):
        if th is None:
            self._threads = multiprocessing.cpu_count()
        else:
            self._threads = th

    @property
    def hmargs(self):
        return self._hmargs

    @hmargs.setter
    def hmargs(self, arg):
        hmargs = dict(basis='linear',
                      m=4,
                      q_reg=2,
                      q_sing=4,
                      aprx='paca',
                      admis='2',
                      eta=0.8,
                      eps_aca=1e-2,
                      strict=True,
                      clf=16,
                      rk=0)
        for k, v in arg.items():
            if k in hmargs:
                hmargs[k] = v

        self._hmargs = hmargs

    @staticmethod
    def _init_process(*args):

        global write_lock, rho, layout, grids, hmargs, file
        write_lock, rho, layout, grids, hmargs, file = args

    @staticmethod
    def _process(job):
        '''
        Process which executes a job.
        '''
        # get options and parameters
        # c = self.fluid_c

        job_id, (f, k) = job
        omg = 2 * np.pi * f
        # k = omg / c

        # generate fem lhs matrix
        Gfem = fem.mbk_mat_spm_from_layout(layout, grids, f)

        # generate bem lhs matrix
        Z = bem.z_mat_hm_from_grid(grids.bem, k, **hmargs)
        # Z = bem.z_mat_fm_from_grid(grids.bem, k)

        Gbem = -omg**2 * 2 * rho * Z

        # define total lhs and find LU decomposition
        G = Gfem + Gbem
        Glu = G.lu()

        # create patch pressure loads
        P = fem.p_cd_mat_sps_from_layout(layout, grids)
        AVG = fem.avg_cd_mat_sps_from_layout(layout, grids)
        ob = grids.bem.on_boundary

        # solve for each source patch
        nctrldom = len(layout.controldomains)
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

        # remove uneccessary second half due to kkr
        nfir = len(t)
        t = t[:(nfir // 2)]
        ppir = ppir[..., :(nfir // 2)]

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
        rho = self.fluid_rho
        layout = self.layout
        grids = self.grids
        hmargs = self.hmargs
        file = self.file
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

        initargs = write_lock, rho, layout, grids, hmargs, file

        with ProcessPoolExecutor(threads,
                                 initializer=DatabaseSolver._init_process,
                                 initargs=initargs) as executor:

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


class Database:

    def __init__(self, dbfile):
        self._dbfile = os.path.normpath(dbfile)
        self._ppfr = None
        self._ppir = None

    @property
    def dbfile(self):
        return self._dbfile

    @property
    def frequency_table(self):

        if self._ppfr is None:
            freqs, ppfr = database.read_patch_to_patch_freq_resp(self.dbfile)
            self._ppfr = freqs, ppfr

         return self._ppfr

    @property
    def impulse_table(self):

        if self._ppir is None:
            times, ppir = database.read_patch_to_patch_imp_resp(self.dbfile)
            self._ppir = times, ppir
        return self._ppir

    @property
    def nctrldom(self):
        return self.frequency_table.shape[0]

    @property
    def nfreq(self):
        return self.frequency_table.shape[-1]

    @property
    def ntime(self):
        return self.impulse_table.shape[-1]

    @property
    def xfr(self, p, pix=None):
        freqs, ppfr = self.frequency_table

        if pix is not None:
            _p = np.zeros(self.nctrldom)
            _p[pix] = p
        else:
            _p = p

        return freqs, np.sum(_p[:, None, None] * ppfr, axis=0)

    @property
    def xim(self, p, pix=None):
        times, ppir = self.impulse_table

        if pix is not None:
            _p = np.zeros(self.nctrldom)
            _p[pix] = p
        else:
            _p = p

        return times, np.sum(_p[:, None, None] * ppir, axis=0)

    def recalculate_fir(self, use_kkr=True, interp=4):

        freqs, ppfr = self.frequency_table
        times, ppir = impulse_response.fft_to_fir(freqs,
                                                 ppfr,
                                                 interp=interp,
                                                 axis=-1,
                                                 use_kkr=use_kkr)

        # remove uneccessary second half due to kkr
        if use_kkr:
            nfir = len(times)
            times = firtimes_t[:(nfir // 2)]
            ppir = ppir[..., :(nfir // 2)]

        self._ppir = times, ppir