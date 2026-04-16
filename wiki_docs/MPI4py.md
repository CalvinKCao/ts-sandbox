# MPI4py

[MPI for Python](https://mpi4py.readthedocs.io/en/stable/) provides Python bindings for the Message Passing Interface (MPI) standard, allowing Python applications to exploit multiple processors on workstations, clusters and supercomputers.
__FORCETOC__

= Available versions =
<code>mpi4py</code> is available as a module, and not from the [wheelhouse](Available_Python_wheels.md) as typical Python packages are.
You can find available version with
```bash
module spider mpi4py
```

and look for more information on a specific version with
```bash
module spider mpi4py/X.Y.Z
```
where <code>X.Y.Z</code> is the exact desired version, for instance <code>4.0.0</code>. 

= Famous first words: Hello World =
1. Run a short [interactive job](Running_jobs#Interactive_jobs.md).
```bash
salloc --account{{=
```<your account> --ntasks{{=}}5}}

2. Load the module.
```bash
module load mpi4py/4.0.0 python/3.12
```

3. Run a Hello World test.
```bash
srun python -m mpi4py.bench helloworld
result=
Hello, World! I am process 0 of 5 on node1.
Hello, World! I am process 1 of 5 on node1.
Hello, World! I am process 2 of 5 on node3.
Hello, World! I am process 3 of 5 on node3.
Hello, World! I am process 4 of 5 on node3.
```
In the case above, two nodes (<code>node1</code> and <code>node3</code>) were allocated, and the jobs were distributed across the available resources.

= mpi4py as a package dependency =
Often <code>mpi4py</code> is a dependency of another package. In order to fulfill this dependency :

1. Deactivate any Python virtual environment.
```bash
test $VIRTUAL_ENV && deactivate
```

<b>Note:</b> If you had a virtual environment activated, it is important to deactivate it first, then load the module, before reactivating your virtual environment.

2. Load the module.
```bash
module load mpi4py/4.0.0 python/3.12
```

3. Check that it is visible by <code>pip</code>
```bash
pip list {{!
``` grep mpi4py
|result=
mpi4py            4.0.0
}}
and is accessible for your currently loaded python module.
```bash
python -c 'import mpi4py'
```
If no errors are raised, then everything is OK!

4. [Create a virtual environment and install your packages](Python#Creating_and_using_a_virtual_environment.md).

= Running jobs =
You can run mpi jobs distributed across multiple nodes or cores. 
For efficient MPI scheduling, please see:
- [MPI job](Running_jobs#MPI_job.md)
- [Advanced MPI scheduling](Advanced_MPI_scheduling.md)

## CPU
1. Write your python code, for instance, broadcasting a numpy array.
**File: "mpi4py-np-bc.py"**
```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(100, dtype='i')
else:
    data = np.empty(100, dtype='i')

comm.Bcast(data, root=0)

for i in range(100):
    assert data[i] == i
```
The example above is based on the [mpi4py tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html#running-python-scripts-with-mpi).

2. Write your submission script.
<tabs>
<tab name="Distributed">
**File: submit-mpi4py-distributed.sh**
```bash
#!/bin/bash

#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --ntasks=4                # adjust this to match the number of tasks/processes to run
#SBATCH --mem-per-cpu=4G          # adjust this according to the memory you need per process

1. Run on cores across the system : https://docs.alliancecan.ca/wiki/Advanced_MPI_scheduling#Few_cores,_any_number_of_nodes

1. Load modules dependencies.
module load StdEnv/2023 gcc mpi4py/4.0.0 python/3.12

1. create the virtual environment on each allocated node: 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index numpy==2.1.1
EOF

1. activate only on main node
source $SLURM_TMPDIR/env/bin/activate;

1. srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python mpi4py-np-bc.py;
```
</tab>

<tab name="Whole nodes">
**File: submit-mpi4py-whole-nodes.sh**
```bash
#!/bin/bash

#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=01:00:00           # adjust this to match the walltime of your job
#SBATCH --nodes=2                 # adjust this to match the number of whole node
#SBATCH --ntasks-per-node=40      # adjust this to match the number of tasks/processes to run per node
#SBATCH --mem-per-cpu=1G          # adjust this according to the memory you need per process

1. Run on N whole nodes : https://docs.alliancecan.ca/wiki/Advanced_MPI_scheduling#Whole_nodes

1. Load modules dependencies.
module load StdEnv/2023 gcc openmpi mpi4py/4.0.0 python/3.12

1. create the virtual environment on each allocated node: 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index numpy==2.1.1
EOF

1. activate only on main node
source $SLURM_TMPDIR/env/bin/activate;

1. srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python mpi4py-np-bc.py;
```
</tab>
</tabs>

3. Test your script.

Before submitting your job, it is important to test that your submission script will start without errors. You can do a quick test in an [interactive job](Running_jobs#Interactive_jobs.md).

4. Submit your job to the scheduler.
```bash
sbatch submit-mpi4py-distributed.sh
```

## GPU
1. From a login node, download the demo example.
```bash
wget https://raw.githubusercontent.com/mpi4py/mpi4py/refs/heads/master/demo/cuda-aware-mpi/use_cupy.py
```
The example above and others, can be found in the [demo folder](https://github.com/mpi4py/mpi4py/tree/master/demo).

2. Write your submission script.
**File: submit-mpi4py-gpu.sh**
```bash
#!/bin/bash

#SBATCH --account=def-someprof    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=08:00:00           # adjust this to match the walltime of your job
#SBATCH --ntasks=2                # adjust this to match the number of tasks/processes to run
#SBATCH --mem-per-cpu=2G          # adjust this according to the memory you need per process
#SBATCH --gpus=1

1. Load modules dependencies.
module load StdEnv/2023 gcc cuda/12 mpi4py/4.0.0 python/3.11

1. create the virtual environment on each allocated node:
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index cupy numba

srun python use_cupy.py;
```

3. Test your script.

Before submitting your job, it is important to test that your submission script will start without errors.
You can do a quick test in an [interactive job](Running_jobs#Interactive_jobs.md).

4. Submit your job
```bash
sbatch submit-mpi4py-gpu.sh
```

= Troubleshooting =

## ModuleNotFoundError: No module named 'mpi4py'
If <code>mpi4py</code> is not accessible, you may get the following error when importing it:
<code>
ModuleNotFoundError: No module named 'mpi4py'
</code>

Possible solutions:
- check which Python versions are compatible with your loaded mpi4py module using <code>module spider mpi4py/X.Y.Z</code>. Once a compatible Python module is loaded, check that <code>python -c 'import mpi4py'</code> works.
- load the module before activating your virtual environment: please see the [mpi4py as a package dependency](MPI4py#mpi4py_as_a_package_dependency.md) section above.

See also [ModuleNotFoundError: No module named 'X'](Python#ModuleNotFoundError:_No_module_named_'X'.md).