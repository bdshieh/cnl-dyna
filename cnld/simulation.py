'''
'''
import numpy as np
import scipy as sp
import scipy.signal
from scipy.constants import epsilon_0 as e_0

from cnld import util, fem, mesh




def pressure_es(v, x, g_eff):
    '''
    '''
    return -e_0 / 2 * v**2 / (x + g_eff)**2
    

@util.memoize
def mem_collapse_voltage(mem, , maxdc=100, atol=1e-10,  maxiter=100):
    '''
    '''
    for i in range(1, maxdc):
        _, is_collapsed = mem_static_disp(K, e_mask, i, h_eff, tol)
        
        if is_collapsed:
            return i
    raise('Could not find collapse voltage')


@util.memoize
def mem_static_disp(mem, vdc, refn=7, atol=1e-10, maxiter=100):
    '''
    '''
    mem_mesh = mesh.square(mem.length_x, mem.length_y, refn)
    K = fem.mem_k_matrix(mem_mesh, mem.y_modulus, m.thickness, m.p_ratio)
    g_eff = mem.gap + mem.isol / mem.permittivity
    F = fem.mem_f_vector(mem_mesh, 1)

    nnodes = K.shape[0]
    x0 = np.zeros(nnodes)

    for i in range(maxiter):
        x0_new = Kinv.dot(F * pressure_es(vdc, x0, g_eff)).squeeze()
        
        if np.max(np.abs(x0_new - x0)) < atol:
            is_collapsed = False
            return x0_new, is_collapsed
        
        x0 = x0_new

    is_collapsed = True
    return x0, is_collapsed


def convolve_fir(A, b, fs):
    '''
    '''
    nsrc, ndest, nfir = A.shape
    _, nsample = b.shape
    # x = np.zeros(ndest)

    if b.ndim == 1:
        b = b[...,None]

    x = np.sum(np.sum(A[:,:,:nsample] * b[:,None,::-1], axis=-1), axis=0).squeeze()
    # for i in range(nsrc):
        # for j in range(ndest):
            # x[i,j] += np.sum(A[i,j,:nsample] * b[i,::-1]) / fs
    return x


def gausspulse(fc, fbw, fs, tpr=-100, retquad=False):
    '''
    '''
    cutoff = sp.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=tpr, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, quad = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)
    
    if retquad:
        return quad
    
    return pulse





class FixedStepSolver:
    
    def __init__(self, fir, v, x0, tstart, tstop, hmin):
        
        t = np.arange(tstart, tstop, hmin)
        npatch = fir.shape[0]
        x = np.zeros((npatch, len(t)))
        x[:,0] = x0
        f = np.zeros((npatch, len(t)))
        # hmax = round((tstop - tstart) / 50)

        self.t = t
        self.v = v
        self.x = x
        self.f = f
        self.fir = fir
        self.hmin = hmin
        self.ti = 0
        # self.hmax = hmax
    
    def step(self):
        
        ti = self.ti
        v = self.v
        x = self.x
        f = self.f
        fir = self.fir
        hmin = self.hmin

        g_eff = 100e-9
        p_es = pressure_es(v[ti], x[:,ti], g_eff)
        f[:,ti] = p_es

        x_new = convolve_fir(fir, f[:,:(ti + 1)], fs=(1 / hmin))
        x[:,ti + 1] = x_new

        self.ti += 1
    

class VariableStepSolver:
    pass
        
    

    
