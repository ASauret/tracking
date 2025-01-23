#import numpy as np
import os
import pickle
import numpy as np
import subprocess
import time
import sys

class Slurm():
    def __init__(self):
        self.thisdir =os.getcwd()
        #self.sshc = 'ssh -t cluster-access "cd '+self.thisdir+' ;"'
        self.sshc = ''
        self.exitc = 'exit'
        self.shname = 'sbatch.sh'
        self.pname = 'fparam'
        self.batchc = 'sbatch '+self.shname
        self.excp = '/machfs/swhite/pyenvs/python38_rnice/bin/python' 
        self.runc = self.excp+' $(sed -n ${SLURM_ARRAY_TASK_ID}p '+self.pname+')'
        self.jobname = 'pyAT'
        self.ntasks=1
        self.ntasks_per_node=1
        self.cpus_per_task=1
        self.mem_per_cpu = '2G'
        self.time = '24:00:00'
        self.njobs = 1
        self.exclusive = False
        self.partition = 'asd'
        self.commands={}

    def set_commands(self):
        commands={'job-name':self.jobname,
                  'ntasks':str(self.ntasks),
                  'ntasks-per-node':str(self.ntasks_per_node),
                  'cpus-per-task':str(self.cpus_per_task),
                  'mem-per-cpu':self.mem_per_cpu,
                  'time':self.time,
                  'array': '1-'+str(self.njobs),
                  'partition': self.partition}
        if self.exclusive:
            commands['exclusive']=''
        self.commands=commands

    def write_batch(self):
        fs = open(self.shname,'w')
        fs.write('#!/bin/sh\n')
        for key in self.commands:
            fs.write('#SBATCH --'+key+' '+self.commands[key]+'\n')
        fs.write(self.runc+'\n')
        fs.close()

    def write_params(self):
        fp = open(self.pname,'w')
        for i in range(self.njobs):
            fp.write(self.thisdir+'/slurm.py '+self.thisdir+'/fin'+' '+self.thisdir+'/fout'+' '+str(i)+'\n')
        fp.close()

    def check_results(self, jobid, timeout=10):
        nok=0
        nrunning = self.njobs
        while nok!=self.njobs and nrunning != 0:
            noka = np.zeros(self.njobs)
            for i in range(self.njobs):
                name = 'fout'+str(i)
                noka[i]=name.encode() in subprocess.check_output('ls', shell=True).splitlines()
            running = os.popen('squeue --jobs='+jobid).read().splitlines()
            nrunning = len(running)-1
            if(nrunning==0):
                time.sleep(1)                           
            nok=sum(noka)
            print('Number of valid files',int(nok),'/',self.njobs)
            print('Number of jobs running',int(nrunning),'/',self.njobs)
            
        if nrunning==0 and nok!=self.njobs:
            print('All jobs finished', int(nok),'/',self.njobs,'successful')
            return[i for i,ii in enumerate(noka) if ii == False]               
        else:
            return None

    def combine(self,njobs=1):
        results=[]
        for i in range(njobs):
            f = open(self.thisdir+'/fout'+str(i),'rb')
            data=pickle.load(f)
            results.append(data.pop('results'))
        return results

    def submit(self,classi,methodname,jobname='pyAT',args=None):
        self.thisdir =os.getcwd()
        self.jobname = jobname
        try:
          self.njobs = len(args)
        except:
          self.njobs = 1
          args = [args]
        data={'class':classi,'method':methodname,'args':args,'njobs':self.njobs}
        f = open('fin','wb')
        pickle.dump(data,f)
        f.close()
        self.set_commands()
        self.write_batch()
        self.write_params()
        command = self.sshc+' '+self.batchc+'; '+self.exitc
        jobid = os.popen(command).read().split()[3]
        missing = self.check_results(jobid)
        if missing is not None:  
           njobtot = self.njobs        
           print('Re-running missing jobs',missing)
           args = [args[i] for i in missing]
           self.submit(classi,methodname,jobname,args)
           return self.combine(njobs=njobtot)
        else:
           return self.combine(njobs=self.njobs)

    def clean(self):
        os.system('rm -rf fout* fin* slurm-*')
        os.system('rm '+self.shname+' '+self.pname)


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 4:
       ind = sys.argv[3]
       f = open(sys.argv[1],'rb')
       data=pickle.load(f)
       f.close()
       c = data.pop('class',None)
       m = data.pop('method',None)
       a = data.pop('args',None)
       if c is not None and m is not None:
           if a is None:
               out=getattr(c,m)()
           else:
               out=getattr(c,m)(a[int(ind)])
           f = open(sys.argv[2]+ind,'wb')
           pickle.dump({'results':out,'args':a},f)
           f.close()
           print('success')
       else:
          print('Input not well defined: Failed')
       sys.exit()





