# Using node-local storage

When [Slurm](Running_jobs.md) starts a job, it creates a temporary directory on each node assigned to the job.
It then sets the full path name of that directory in an environment variable called <code>SLURM_TMPDIR</code>.

Because this directory resides on local disk, input and output (I/O) to it
is almost always faster than I/O to a [network storage](Storage_and_file_management.md) (/project, /scratch, or /home).
Specifically, local disk is better for frequent small I/O transactions than network storage.
Any job doing a lot of input and output (which is most jobs!) may expect
to run more quickly if it uses <code>$SLURM_TMPDIR</code> instead of network storage.

The temporary character of <code>$SLURM_TMPDIR</code> makes it more trouble to use than 
network storage.
Input must be copied from network storage to <code>$SLURM_TMPDIR</code> before it can be read,
and output must be copied from <code>$SLURM_TMPDIR</code> back to network storage before the job ends
to preserve it for later use.  

= Input =

In order to <i>read</i> data from <code>$SLURM_TMPDIR</code>, you must first copy the data there.  
In the simplest case, you can do this with <code>cp</code> or <code>rsync</code>:
```

cp /project/def-someone/you/input.files.* $SLURM_TMPDIR/

```

This may not work if the input is too large, or if it must be read by processes on different nodes.
See [Multinode jobs](Using_node-local_storage#Multinode_jobs.md) and [Amount of space</i>](Using_node-local_storage#Amount_of_space.md) below for more.

## Executable files and libraries

A special case of input is the application code itself. 
In order to run the application, the shell started by Slurm must open
at least an application file, which it typically reads from network storage.
But few applications these days consist of exactly one file; 
most also need several other files (such as libraries) in order to work.

We particularly find that using an application in a [Python](Python.md) virtual environment 
generates a large number of small I/O transactions—more than it takes 
to create the virtual environment in the first place.  This is why we recommend  
[creating virtual environments inside your jobs](Python#Creating_virtual_environments_inside_of_your_jobs.md)
using <code>$SLURM_TMPDIR</code>.

= Output =

Output data must be copied from <code>$SLURM_TMPDIR</code> back to some permanent storage before the
job ends.  If a job times out, then the last few lines of the job script might not 
be executed.  This can be addressed three ways:
- request enough runtime to let the application finish, although we understand that this isn't always possible;
- write [checkpoints](Points_de_contrôle.md) to network storage, not to <code>$SLURM_TMPDIR</code>;
- write a signal trapping function.

## Signal trapping

You can arrange that Slurm will send a signal to your job shortly before the runtime expires,
and that when that happens your job will copy your output from <code>$SLURM_TMPDIR</code> back to network storage.
This may be useful if your runtime estimate is uncertain,
or if you are chaining together several Slurm jobs to complete a long calculation.

To do so you will need to write a shell function to do the copying, 
and use the <code>trap</code> shell command to associate the function with the signal.
See [this page](https://services.criann.fr/en/services/hpc/cluster-myria/guide/signals-sent-by-slurm/) from
CRIANN for an example script and detailed guidance.

This method will not preserve the contents of <code>$SLURM_TMPDIR</code> in the case of a node failure,
or certain malfunctions of the network file system.

= Multinode jobs =

If a job spans multiple nodes and some data is needed on every node, then a simple <code>cp</code> or <code>tar -x</code> will not suffice.

## Copy files

Copy one or more files to the <code>SLURM_TMPDIR</code> directory on every node allocated like this:
```bash
srun --ntasks{{=
```$SLURM_NNODES --ntasks-per-node{{=}}1 cp file [files...] $SLURM_TMPDIR}}

## Compressed archives

### ZIP

Extract to the <code>SLURM_TMPDIR</code>:
```bash
srun --ntasks{{=
```$SLURM_NNODES --ntasks-per-node{{=}}1 unzip archive.zip -d $SLURM_TMPDIR}}

### Tarball
Extract to the <code>SLURM_TMPDIR</code>:
```bash
srun --ntasks{{=
```$SLURM_NNODES --ntasks-per-node{{=}}1 tar -xvf archive.tar.gz -C $SLURM_TMPDIR}}

= Amount of space =

At <b>[Trillium](Trillium.md)</b>, $SLURM_TMPDIR is implemented as <i>RAMdisk</i>, 
so the amount of space available is limited by the memory on the node,
less the amount of RAM used by your application.

At the general-purpose clusters, 
the amount of space available depends on the cluster and the node to which your job is assigned.

{| class="wikitable sortable"
! cluster !! space in $SLURM_TMPDIR !! size of disks
|-
| [Fir](Fir.md)   || 7T || 7.84T
|-
| [Narval](Narval.md)  || 800G || 960G, 3.84T
|-
| [Nibi](Nibi.md)  || 3T || 3T, 11T
|-
| [Rorqual](Rorqual.md)  || 375G || 480G, 3.84T
|}

If your job reserves [whole nodes](Advanced_MPI_scheduling#Whole_nodes.md), 
then you can reasonably assume that this much space is available to you in $SLURM_TMPDIR on each node.
However, if the job requests less than a whole node, then other jobs may also write to the same filesystem
(but a different directory!), reducing the space available to your job.

Some nodes at each site have more local disk than shown above.  
See <i>Node characteristics</i> at the appropriate cluster's page ([Fir](Fir.md), [Narval](Narval.md), [Nibi](Nibi.md), [Rorqual](Rorqual.md)) for guidance.