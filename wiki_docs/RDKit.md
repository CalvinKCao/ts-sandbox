# RDKit

[RDKit](https://www.rdkit.org/) is a collection of cheminformatics and machine-learning software written in C++ and Python.

__FORCETOC__

= Available versions =
<code>rdkit</code> C++ libraries and Python bindings are available as a module.

You can find available versions with:
```bash
module spider rdkit
```

and look for more information on a specific version with:
```bash
module spider rdkit/X.Y.Z
```
where <code>X.Y.Z</code> is the exact desired version, for instance <code>2024.03.5</code>.

= Python bindings =
The module contains bindings for multiple Python versions. To discover which are the compatible Python versions, run:
```bash
module spider rdkit/X.Y.Z
```

where <TT>X.Y.Z</TT> represents the desired version.

## rdkit as a Python package dependency
When <code>rdkit</code> is a dependency of another package, the dependency needs to be fulfilled:

1. Deactivate any Python virtual environment.
```bash
test $VIRTUAL_ENV && deactivate
```

<b>Note:</b> If you had a virtual environment activated, it is important to deactivate it first, then load the module, before reactivating your virtual environment.

2. Load the module.
```bash
module load rdkit/2024.03.5 python/3.12
```

3. Check that it is visible by <code>pip</code>
```bash
pip list {{!
``` grep rdkit
|result=
rdkit            2024.3.5
}}

```bash
python -c 'import rdkit'
```
If no errors are raised, then everything is OK!

4. [Create a virtual environment and install your packages](Python#Creating_and_using_a_virtual_environment.md).

= Troubleshooting =

## ModuleNotFoundError: No module named 'rdkit'
If <code>rdkit</code> is not accessible, you may get the following error when importing it:
<code>
ModuleNotFoundError: No module named 'rdkit'
</code>

Possible solutions:
- check which Python versions are compatible with your loaded RDKit module using <code>module spider rdkit/X.Y.Z</code>. Once a compatible Python module is loaded, check that <code>python -c 'import rdkit'</code> works.
- load the module before activating your virtual environment: please see the  [rdkit as a package dependency](RDKit#rdkit_as_a_Python_package_dependency.md) section above.

See also [ModuleNotFoundError: No module named 'X'](Python#ModuleNotFoundError:_No_module_named_'X'.md).