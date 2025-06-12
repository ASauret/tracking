import at
import slurm
import pickle
import numpy as np
import abc
import time


class Slurm_runner(abc.ABC):
    def __init__(self, args):
        self.args = args
        self.slurm = slurm.Slurm()

    @abc.abstractmethod
    def run(self, varin):
        return
           
    def submit(self):
        t0 = time.time()
        methodname = self.run.__name__
        *data,= self.slurm.submit(self,methodname,
                                  jobname=self.slurm.jobname,
                                  args=self.args)
        print('Computation took:',time.time()-t0,' s')
        return data
    

class Injeff_runner(Slurm_runner):
    def __init__(self, all_varin, ringpath, nturns, pulsed_elements_refpts=None, use_6d=True, use_mp=False):
        self.ringpath = ringpath
        self.pulsed_elements_refpts = pulsed_elements_refpts
        self.nturns = nturns
        self.use_6d = use_6d
        self.use_mp = use_mp
        super().__init__(all_varin)
              
    def run(self, varin):
        ring = at.load_lattice(self.ringpath)
        if self.use_6d:
            ring.enable_6d()
        if self.pulsed_elements_refpts is not None:
            ring.track(varin, nturns=1, use_mp=self.use_mp, in_place=True)
            for r in self.pulsed_elements_refpts:
                ring[r].PolynomB *= 0.0
                ring[r].PolynomA *= 0.0
        ring.track(varin, nturns=self.nturns, use_mp=self.use_mp, in_place=True)
        return 1-np.sum(np.isnan(varin[0,:]))/len(varin[0])


class Injeff_runner_EBS(Slurm_runner):
    def __init__(self, all_varin, ringpath, endtl2_path, nturns, pulsed_elements_refpts=None, use_6d=True, use_mp=False):
        self.ringpath = ringpath
        self.endtl2_path = endtl2_path
        self.pulsed_elements_refpts = pulsed_elements_refpts
        self.nturns = nturns
        self.use_6d = use_6d
        self.use_mp = use_mp
        super().__init__(all_varin)

    def track_idx(self, varin):
        end_tl2 = at.load_lattice(self.endtl2_path, use='RING', check=False)
        idx_nlk2 = end_tl2.get_uint32_index('NLK2')[0]
        idx_dr2 = end_tl2.get_uint32_index('DR_NLK')[0]
        idx_nlk3 = end_tl2.get_uint32_index('NLK3')[0]
        s_marker = 0.5*end_tl2[idx_nlk2].Length + end_tl2[idx_dr2].Length + end_tl2[idx_nlk3].Length
        ring = at.load_lattice(self.ringpath, mat_key='ring')
        ring.enable_6d()
        new_ring = ring.sbreak(break_s=s_marker)
        idx_mk = new_ring.get_uint32_index('sbreak')[0]
        rot_ring = new_ring.rotate(idx_mk)
        idx_id04 = rot_ring.get_uint32_index('ID04')[0]
        o6,*_ = rot_ring.find_orbit()
        for i in range(6):
            varin[i] += o6[i]
        varout, *_ = rot_ring.track(varin, nturns=1, refpts=idx_id04)
        return np.squeeze(varout[:,:,0,0])

              
    def run(self, varin):
        """
        Rotation of the ring performed to start tracking in the ring at the end of NLK3.
        """
        varin = self.track_idx(varin)
        ring = at.load_lattice(self.ringpath)
        if self.use_6d:
            ring.enable_6d()
        if self.pulsed_elements_refpts is not None:
            ring.track(varin, nturns=1, use_mp=self.use_mp, in_place=True)
            for r in self.pulsed_elements_refpts:
                ring[r].PolynomB *= 0.0
                ring[r].PolynomA *= 0.0
        ring.track(varin, nturns=self.nturns, use_mp=self.use_mp, in_place=True)
        return 1-np.sum(np.isnan(varin[0,:]))/len(varin[0])
    
    

class Injeff_runner_EBS_thin(Slurm_runner):
    def __init__(self, all_varin, ringpath, nturns, pulsed_elements_refpts=None, use_6d=True, use_mp=False):
        self.ringpath = ringpath
        self.pulsed_elements_refpts = pulsed_elements_refpts
        self.nturns = nturns
        self.use_6d = use_6d
        self.use_mp = use_mp
        super().__init__(all_varin)

    
    def run(self, varin):
        """
        No ring rotation needed. 
        Thin element injection is directly at the center of the straight section
        """
        ring = at.load_lattice(self.ringpath)
        if self.use_6d:
            ring.enable_6d()
        if self.pulsed_elements_refpts is not None:
            ring.track(varin, nturns=1, use_mp=self.use_mp, in_place=True)
            for r in self.pulsed_elements_refpts:
                ring[r].PolynomB *= 0.0
                ring[r].PolynomA *= 0.0
        ring.track(varin, nturns=self.nturns, use_mp=self.use_mp, in_place=True)
        return 1-np.sum(np.isnan(varin[0,:]))/len(varin[0])