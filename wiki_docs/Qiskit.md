# Qiskit

Developed in Python by IBM, [Qiskit](https://docs.quantum.ibm.com/) is an open-source quantum computing library. Like [PennyLane](PennyLane.md) and [Snowflurry](Snowflurry.md), it allows you to build, simulate and run quantum circuits.

## Installation
1. Load the Qiskit dependencies.
```bash
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2
```

2. Create and activate a [Python virtual environment](Python#Creating_and_using_a_virtual_environment.md).
```bash
virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate
```

3. Install a version of Qiskit.
```bash
prompt=(ENV) [name@server ~]
pip install --no-index --upgrade pip
pip install --no-index qiskit{{=
```{{=}}X.Y.Z  qiskit_aer{{=}}{{=}}X.Y.Z}}
where <code>X.Y.Z</code> is the version number, for  example <code>1.4.0</code>. To install the most recent version available on our clusters, do not specify a number. Here, we only imported <code>qiskit</code> and <code>qiskit_aer</code>. You can add other Qiskit software with the syntax <code>qiskit_package==X.Y.Z</code> where <code>qiskit_package</code> is the softare name, for example <code>qiskit-finance</code>. To see the wheels that are currently available, see [Available Python wheels](Available_Python_wheels.md). 

4. Validate the installation.
```bash
prompt=(ENV)[name@server ~]|python -c 'import qiskit'
```

5. Freeze the environment and its dependencies.
```bash
prompt=(ENV)[name@server ~]|pip freeze --local > ~/qiskit_requirements.txt
```
## Running Qiskit on a cluster
**File: script.sh**
```sh
#!/bin/bash
#SBATCH --account=def-someuser #Modify with your account name
#SBATCH --time=00:15:00        #Modify as needed
#SBATCH --cpus-per-task=1      #Modify as needed
#SBATCH --mem-per-cpu=1G       #Modify as needed

1. Load module dependencies.
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2 

1. Generate your virtual environment in $SLURM_TMPDIR.                                                                                                         
virtualenv --no-download ${SLURM_TMPDIR}/env                                                                                                                   
source ${SLURM_TMPDIR}/env/bin/activate  

1. Install Qiskit and its dependencies.                                                                                                                                                                                                                                                                                    
pip install --no-index --upgrade pip                                                                                                                            
pip install --no-index --requirement ~/qiskit_requirements.txt

1. Modify your Qiskit program.                                                                                                                                                                       
python qiskit_example.py
```
You can then [submit your job to the scheduler](Running_jobs.md). 
## Using Qiskit with MonarQ (in preparation)
<!-- You can import a Qiskit project in Pennylane to run it on [MonarQ/en}MonarQ](MonarQ/en}MonarQ.md). 
1. Install PennyLane (if it is not yet installed).
```bash
pip install --no-index pennylane
```

2. Install the pennylane-qiskit plugin.
```bash
pip install --no-index pennylane-qiskit
```

3. Install the pennylane-snowflurry plugin.
```bash
pip install pennylane-snowflurry
```

To use a Qiskit circuit with PennyLane, follow the instructions in the [documentation](https://docs.pennylane.ai/en/stable/introduction/importing_workflows.html). You can then execute the circuit on MonarQ by following the instructions in the [pennylane-snowflurry documentation](https://github.com/calculquebec/pennylane-snowflurry).
-->

## Use case: Bell states
Before you create a simulation of the first Bell state on [Narval](Narval.md), the required modules need to be loaded. 
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit, transpile
    from qiskit.visualization import plot_histogram

Define the circuit. Apply an Hadamard gate to create a superposition state on the first qubit and a CNOT gate to intricate the first and second qubits.
    circuit = QuantumCircuit(2,2)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.measure_all()

We will use the default simulator <code>AerSimulator</code>. This provides the final number of qubits after having made 1000 measurements.
    simulator = AerSimulator()
    result = simulator.run(circuit, shots=1000).result()
    counts = result.get_counts()
    print(counts)
    {'00': 489, '11': 535}
The results are displayed.
    plot_histogram(counts)

[thumb|Results of 1000 measurements on the first Bell 
 state](File:Qiskit_counts.png.md)