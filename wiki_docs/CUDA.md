# CUDA

"CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs)."<ref>[NVIDIA CUDA Home Page](https://developer.nvidia.com/cuda-toolkit). CUDA is a registered trademark of NVIDIA.</ref>

It is reasonable to think of CUDA as a set of libraries and associated C, C++, and Fortran compilers that enable you to write code for GPUs. See [OpenACC Tutorial](OpenACC_Tutorial.md) for another set of GPU programming tools.

## Quick start guide

### Compiling
Here we show a simple example of how to use the CUDA C/C++ language compiler, <code>nvcc</code>, and run code created with it. For a longer tutorial in CUDA programming, see [CUDA tutorial](CUDA_tutorial.md).

First, load a CUDA [module](Utiliser_des_modules.md).
```console
$ module purge
$ module load cuda
```

The following program will add two numbers together on a GPU. Save the file as <code>add.cu</code>. <i>The <code>cu</code> file extension is important!</i>. 

**File: add.cu**
```c++
#include <iostream>

__global__ void add (int *a, int *b, int *c){
  *c = *a + *b;
}

int main(void){
  int a, b, c;
  int *dev_a, *dev_b, *dev_c;
  int size = sizeof(int);
  
  //  allocate device copies of a,b, c
  cudaMalloc ( (void**) &dev_a, size);
  cudaMalloc ( (void**) &dev_b, size);
  cudaMalloc ( (void**) &dev_c, size);
  
  a=2; b=7;
  //  copy inputs to device
  cudaMemcpy (dev_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy (dev_b, &b, size, cudaMemcpyHostToDevice);
  
  // launch add() kernel on GPU, passing parameters
  add <<< 1, 1 >>> (dev_a, dev_b, dev_c);
  
  // copy device result back to host
  cudaMemcpy (&c, dev_c, size, cudaMemcpyDeviceToHost);
  std::cout<<a<<"+"<<b<<"="<<c<<std::endl;
  
  cudaFree ( dev_a ); cudaFree ( dev_b ); cudaFree ( dev_c );
}
```

Compile the program with <code>nvcc</code> to create an executable named <code>add</code>.
```console
$ nvcc add.cu -o add
```

### Submitting jobs
To run the program, create a Slurm job script as shown below. Be sure to replace <code>def-someuser</code> with your specific account (see [Accounts and projects](Running_jobs#Accounts_and_projects.md)). For options relating to scheduling jobs with GPUs see [Using GPUs with Slurm](Using_GPUs_with_Slurm.md). 
**File: gpu_job.sh**
```sh
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=400M                # memory (per node)
#SBATCH --time=0-00:10            # time (DD-HH:MM)
./add #name of your program
```

Submit your GPU job to the scheduler with 
```console
$ sbatch gpu_job.sh
Submitted batch job 3127733
```For more information about the <code>sbatch</code> command and running and monitoring jobs, see [Running jobs](Running_jobs.md).

Once your job has finished, you should see an output file similar to this:
```console
$ cat slurm-3127733.out
2+7=9
```
If you run this without a GPU present, you might see output like <code>2+7=0</code>. 

### Linking libraries
If you have a program that needs to link some libraries included with CUDA, for example [cuBLAS](https://developer.nvidia.com/cublas), compile with the following flags
```console
nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64
```

To learn more about how the above program works and how to make the use of GPU parallelism, see [CUDA tutorial](CUDA_tutorial.md).

## Troubleshooting

### Compute capability

NVidia has created this technical term, "which indicates what features are supported by that GPU and specifies some hardware parameters for that GPU."
See [Compute Capability and Streaming Multiprocessor Versions](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-compute-capability-sm-version)
for more details.

The following errors are connected with compute capability:

```

nvcc fatal : Unsupported gpu architecture 'compute_XX'

```

```

no kernel image is available for execution on the device (209)

```

If you encounter either of these errors, you may be able to fix it by adding the correct <i>flag</i> to the <code>nvcc</code> call:

```

-gencode arch=compute_XX,code=[sm_XX,compute_XX]

```

If you are using <code>cmake</code>, provide the following flag:

```

cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX

```

where “XX” is the compute capability of the Nvidia GPU that you expect to run the application on. 
To find the value to replace “XX“, see [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) and omit the decimal point.

<b>For example,</b> if you will run your code on a Narval A100 node, the NVidia table gives its compute capability as "8.0".
The correct flag to use when compiling with <code>nvcc</code> is then:

```

-gencode arch=compute_80,code=[sm_80,compute_80]

```

The flag to supply to <code>cmake</code> is:

```

cmake .. -DCMAKE_CUDA_ARCHITECTURES=80

```