''''''
import numpy as np
from concurrent.futures import ProcessPoolExecutor


class DatabaseSolver:
    '''
    '''
    def __init__(self, file=None, layout=None, grids=None, freqs=None, 
        fluid_c=1500, fluid_rho=1000, refn=7, overwrite=False):

        self.layout = layout
        self.grids = grids
        self.file = file
        self.fluid_c = fluid_c
        self.fluid_rho = fluid_rho
        self.refn = refn
        self.overwrite = overwrite


    @property
    def file(self):
        self._file
    
    @file.setter
    def set_file(self, f):
        self._file = f

    @property
    def layout(self):
        return self._layout
    
    @property
    def grids(self):
        return self._grids

    @property
    def freqs(self):
        self._freqs
    
    @freqs.setter
    def set_freqs(self, *args):
        if len(args) == 3:
            self._freqs = np.arange(*args)
        elif len(args) == 1
            self.freqs = np.array(args[0])
        else:
            raise TypeError

    @property
    def fluid_c(self):
        self._fluid_c

    @property
    def fluid_rho(self):
        self._fluid_rho

    @property
    def refn(self):
        self._refn
    
    @property
    def overwrite(self):
        self._overwrite

    def _process(self, f):
        '''
        Process which executes a job.
        '''
        # get options and parameters
        c = self.fluid_c
        rho = self.fluid_rho
        layout = self.layout
        refn = self.refn
        grids = self.grids

        omg = 2 * np.pi * f
        k = omg / c

        # generate fem lhs matrix
        Gfem = fem.mbk_mat_sps_from_layout(layout, grids, f)

        # generate bem lhs matrix
        hmkwrds = [
            'format', 'aprx', 'basis', 'admis', 'eta', 'eps', 'm', 'clf', 'eps_aca',
            'rk', 'q_reg', 'q_sing', 'strict'
        ]
        hmargs = {k: getattr(cfg, k) for k in hmkwrds}
        Z = bem.z_mat_hm_from_layout(layout, grids, k, **hmargs)
        Gbem = -omg**2 * 2 * rho * Z

        # define total lhs and find LU decomposition
        G = MbkSparseMatrix(Gfem) + Gbem
        Glu = G.lu()

        # create patch pressure loads
        P = fem.p_mat_sps_from_layout(layout, grids)
        AVG = fem.avg_mat_sps_from_layout(layout, grids)
        ob = grids.bem.on_boundary

        # solve for each source patch
        npatch = len(layout.controldomainlist)
        cd_source = np.arange(npatch)
        cd_dest = np.arange(npatch)

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
            x_cd = AVG.T @ x

            # write patch displacement results to frequency response database
            data = {}
            data['frequency'] = repeat(f)
            data['wavenumber'] = repeat(k)
            data['source_patch'] = repeat(sid)
            data['dest_patch'] = dest_patch
            data['displacement_real'] = np.real(x_cd)
            data['displacement_imag'] = np.imag(x_cd)
            data['time_solve'] = repeat(time_solve)

            with write_lock:
                database.append_patch_to_patch_freq_resp(file, **data)

            # write node displacement results to frequency response database
            data = {}
            data['node_id'] = range(len(mesh.vertices))
            data['x'] = mesh.vertices[:, 0]
            data['y'] = mesh.vertices[:, 1]
            data['z'] = mesh.vertices[:, 2]
            data['frequency'] = repeat(f)
            data['wavenumber'] = repeat(k)
            data['source_patch'] = repeat(sid)
            data['displacement_real'] = np.real(x)
            data['displacement_imag'] = np.imag(x)

            with write_lock:
                database.append_patch_to_node_freq_resp(file, **data)

        with write_lock:
            util.update_progress(file, job_id)
    
    def postprocess(file, interp):
        '''
        Postprocess generated database by calculating impulse responses from frequency data.
        '''
        # postprocess and convert frequency response to impulse response
        freqs, ppfr = database.read_patch_to_patch_freq_resp(file)
        t, ppir = impulse_response.fft_to_fir(freqs,
                                            ppfr,
                                            interp=interp,
                                            axis=-1,
                                            use_kkr=True)
        source_patches, dest_patches, times = np.meshgrid(np.arange(ppir.shape[0]),
                                                        np.arange(ppir.shape[1]),
                                                        t,
                                                        indexing='ij')

        data = {}
        data['source_patch'] = source_patches.ravel()
        data['dest_patch'] = dest_patches.ravel()
        data['time'] = times.ravel()
        data['displacement'] = ppir.ravel()
        database.append_patch_to_patch_imp_resp(file, **data)


    def solve(self):
        '''
        '''
        # get parameters from config and args
        file = self.file
        overwrite = self.overwrite
        threads = args.threads if args.threads else multiprocessing.cpu_count()
        freqs = self.freqs
        c = self.fluid_c

        # calculate job-related values
        
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
                database.create_db(file, **cfg._asdict())
                util.create_progress_table(file, njobs)

                # append node information
                amesh = Mesh.from_abstract(abstract.load(cfg.array_config),
                                        refn=cfg.mesh_refn)
                nodes = amesh.vertices
                database.append_node(file,
                                    node_id=range(len(nodes)),
                                    x=nodes[:, 0],
                                    y=nodes[:, 1],
                                    z=nodes[:, 2])

            else:  # continue from current progress
                is_complete, ijob = util.get_progress(file)
                if np.all(is_complete): return
        else:
            # make directories if they do not exist
            file_dir = os.path.dirname(os.path.abspath(file))
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # create databases
            database.create_db(file, **cfg._asdict())
            util.create_progress_table(file, njobs)

            # append node information
            amesh = Mesh.from_abstract(abstract.load(cfg.array_config), refn=cfg.mesh_refn)
            nodes = amesh.vertices
            database.append_node(file,
                                node_id=range(len(nodes)),
                                x=nodes[:, 0],
                                y=nodes[:, 1],
                                z=nodes[:, 2])

        # start multiprocessing pool and run process
        try:
            write_lock = multiprocessing.Lock()

            pool = multiprocessing.Pool(threads,
                                        initializer=init_process,
                                        initargs=(write_lock, abstract.dumps(cfg), file),
                                        maxtasksperchild=1)

            jobs = util.create_jobs((freqs, 1), (wavenums, 1),
                                    mode='zip',
                                    is_complete=is_complete)
            result = pool.imap_unordered(run_process, jobs, chunksize=1)

            for r in tqdm(result, desc='Running', total=njobs, initial=ijob):
                pass

            postprocess(file, cfg.freq_interp)

        except Exception as e:
            print(e)
        finally:
            pool.close()
            pool.terminate()