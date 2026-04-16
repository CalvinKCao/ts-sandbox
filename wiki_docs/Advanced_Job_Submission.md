# Advanced Job Submission

## Submitting numerous compute tasks

In case you have multiple data files to process with many different combinations of parameters, you may use at least one of the following tools to better manage numerous similar calculation cases:

- <b>[Job Arrays](Job_arrays.md)</b>: they allow using a single script to submit many similar jobs at once. This method is ideal when each individual job is at least one hour long and the total number of jobs is at most one thousand.
- <b>[GNU Parallel](GNU_Parallel.md)</b>: when the serial or small parallel jobs are too short individually or too numerous, GNU Parallel can run and manage numerous calculation cases, including parameter sweeps, on a single node reserved via a parallel job. Thanks to that, the number of jobs submitted to the scheduler is considerably reduced.
- <b>[GLOST](GLOST.md)</b>: the <i>Greedy Launcher Of Small Tasks</i> uses [MPI](MPI.md) and a manager-worker architecture to progressively run a long list of serial tasks on the CPU cores reserved for a parallel job.
- <b>[META](META-Farm.md)</b>: a suite of scripts designed in SHARCNET to automate high-throughput computing (running a large number of related serial, parallel, or GPU calculations).

## Inter-job dependencies

While Slurm jobs are the building-blocks of compute pipelines, inter-job dependencies are the links and relationships between the steps of a pipeline. For example, if two different jobs need to run one after the other, the second job *depends* on the first one. The dependency could be by the start time, the end time or the final status of the first job. Typically, we want the second job to be started only once the first job has succeeded. For example:

```bash
JOBID1{{=
```$(sbatch --parsable job1.sh)           # Save the first job ID
|sbatch --dependency{{=}}afterok:$JOBID1 job2.sh   # Depends on the first job
}}

Notes
- Multiple jobs can have the same dependency (multiple jobs waiting after one job).
- A job can have multiple dependencies (one job waiting after multiple jobs).
- There are multiple types of dependencies: <code>after</code>, <code>afterany</code>, <code>afterok</code>, <code>afternotok</code>, etc. For more details, see the <code>--dependency</code> option on the [official <code>sbatch</code> documentation page](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

## Heterogeneous jobs

The Slurm scheduler supports [heterogeneous jobs](https://slurm.schedmd.com/heterogeneous_jobs.html). This could be very useful if you know in advance that your [MPI](MPI.md) application will require more CPU cores and more memory for the main process than for the other processes.

For example, if the main process requires 8 cores and a total of 32GB of RAM, while the other processes only require 1 core and 1GB of RAM, we can specify both types of requirements in a job script:

**File: heterogeneous_mpi_job.sh**
```sh
#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=8 --mem-per-cpu=4000M
#SBATCH hetjob
#SBATCH --ntasks=15 --cpus-per-task=1 --mem-per-cpu=1000M

srun application.exe
```

Or we can separate resource requests with a colon (<code>:</code>) on the <code>sbatch</code> command line:

```bash
sbatch --ntasks{{=
```1 --cpus-per-task{{=}}8 --mem-per-cpu{{=}}4000M : --ntasks{{=}}15 --cpus-per-task{{=}}1 --mem-per-cpu{{=}}1000M  mpi_job.sh
}}