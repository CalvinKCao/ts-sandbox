# NCCL

= What is NCCL =
Please see the [NVIDIA webpage](https://developer.nvidia.com/nccl).

= Troubleshooting =
To activate NCCL debug outputs, set the following variable before running NCCL:
 NCCL_DEBUG=info

To fix <code>Caught error during NCCL init [...] connect() timed out</code> errors, set the following variable before running NCCL:
 export NCCL_BLOCKING_WAIT=1