# EasyBuild

[EasyBuild](https://easybuild.io/) is a tool for building, installing, and maintaining software on high-performance computing systems. 
We use it to build almost everything in our software repository, [CVMFS](Accessing_CVMFS.md).

= EasyBuild and modules =
One of the key features of EasyBuild is that it automatically generates environment [modules](Utiliser_des_modules.md) which can be used to make a software package available in your session. In addition to defining standard Linux environment variables such as <code>PATH</code>, <code>CPATH</code> and <code>LIBRARY_PATH</code>, EasyBuild also defines some environment variables specific to EasyBuild, two of which may be particularly interesting to users:
- <code>EBROOT<name></code>: Contains the full path to the location where the software <code><name></code> is installed.
- <code>EBVERSION<name></code>: Contains the full version of the software <code><name></code> loaded by this module.

For example, the module <code>python/3.10.2</code> on Narval defines: 
- <code>EBROOTPYTHON</code>: <code>/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/python/3.10.2</code>
- <code>EBVERSIONPYTHON</code>: <code>3.10.2</code>

You can see the environment variables defined by the <code>python/3.10.2</code> module using: 
```bash
module show python/3.10.2 {{!
``` grep EB}}

= Installation recipes and logs =
EasyBuild keeps a copy of the recipe used to install each software package, as well as a detailed log inside the installation directory. This is accessible in the directory <code>$EBROOT<name>/easybuild</code>. For example, for the <code>python/3.10.2</code> module, the installation directory contains, amongst other things: 
- <code>$EBROOTPYTHON/easybuild/Python-3.10.2.eb</code>
- <code>$EBROOTPYTHON/easybuild/easybuild-Python-3.10.2-*.log</code>

= Using EasyBuild in your own account =
EasyBuild can be used to install software packages in your own account. However, in most cases, it is preferable to ask our [technical support](technical_support.md) to install the software centrally for you. This is because that will ensure that the software package is available on all of our clusters. It will also avoid using your quota, and it will avoid causing undue load on the parallel filesystems. 

{{Warning|title=When to use or not use EasyBuild to install software in your home
|content=There are a few use cases in which you may want to use EasyBuild to install software in your own space: 
- if you need a custom or modified build
- if you need to install a nightly build, or a version of a software which does not have a release number
- if we are not allowed to install the package centrally for licensing reasons, such as some commercial software packages ([VASP](VASP.md) and [Materials Studio](Materials_Studio.md) in particular)

On the contrary, you <b>should not</b> install software packages in your own space for the following reasons: 
- if you need a different release version
- if you need a software package built using a different compiler, MPI or CUDA implementation

When in doubt, please ask our [technical support](technical_support.md) for advice. }}

## What is a recipe


Recipes, also known as EasyConfig files are text files containing the information EasyBuild needs to build a particular piece of software in a particular environment. They are named following a convention: 
- <code><name>-<version>-<toolchain name>-<toolchain version>.eb</code>
where <code><name></code> is the name of the package, <code><version></code> is its version, <code><toolchain name></code> is the name of the toolchain and <code><toolchain version></code> is its version. More on toolchains later.

## Finding a recipe
EasyBuild contains a lot of recipes which may or may not compile with the toolchains we have. The surest way to get a recipe that works is to start from one of the recipes which we have installed. These can be found either in the installation folder, as mentioned above, or in the <code>/cvmfs/soft.computecanada.ca/easybuild/ebfiles_repo/$EBVERSIONGENTOO</code> folder. 

{{Callout|title=What is in a toolchain?
|content=Toolchains are a combination of compiler, MPI implementation, CUDA version, and mathematical libraries, which are used to compile the software package. They usually hold a rather obscure name such as <code>gofbc</code> which, in this case, means it is a combination of GCC, OpenMPI, FlexiBlas and CUDA. However, you do not need to remember this naming, since toolchains themselves have recipes, which are also available in the <code>/cvmfs/soft.computecanada.ca/easybuild/ebfiles_repo/$EBVERSIONGENTOO</code> directory. For example, the <code>gofbc</code> toolchain, version <code>2020.1.403.114</code> contains, as per <code>/cvmfs/soft.computecanada.ca/easybuild/ebfiles_repo/$EBVERSIONGENTOO/gofbc/gofbc-2020.1.403.114.eb</code>: 
```

local_gccver = '9.3.0'

1. specify subtoolchains as builddependencies
1. this way they will be considered as subtoolchains but
1. are not loaded in the modulefile or software compiled
1. with this toolchain
builddependencies = [
    ('gccflexiblascuda', '2020.1.114'),
    ('gompic', version),
]

dependencies = [
    ('GCC', local_gccver),  # part of gcccuda
    ('CUDA', '11.4', '', ('GCC', local_gccver)),  # part of gcccuda
    ('OpenMPI', '4.0.3', '', ('gcccuda', '2020.1.114')),
    ('FlexiBLAS', '3.0.4'),
]

```

which means that it contains GCC version 9.3.0, OpenMPI 4.0.3, CUDA 11.4, and FlexiBLAS 3.0.4. The <code>builddependencies</code> part means that the <code>gofbc</code> toolchain is a superset of the <code>gompic</code> and the <code>gccflexiblascuda</code> toolchains. When a toolchain is a superset of other toolchains, it allows software packages built with the former to have dependencies on software packages built with the latter, i.e. software packages built with <code>gofbc</code> can depend on software packages built with <code>gompic</code>, but not the other way around. 
}}

## Installing a software with EasyBuild
Once you have found a recipe matching your needs, copy its recipe from the <code>/cvmfs/soft.computecanada.ca/easybuild/ebfiles_repo/$EBVERSIONGENTOO</code> folder, and modify it as needed. Then, run 
```bash
eb <recipe.eb>
```
to install it. This will install the software inside of your home directory, in <code>$HOME/.local/easybuild</code>. After the installation is completed, exit your session and reconnect to the cluster, and it should be available to load as a module.

### Reinstalling an existing version
If you are reinstalling the exact same version as one we have installed centrally, but with modified parameters, you need to use 
```bash
eb <recipe.eb> --force
```
to install a local version in your home.

### Installing in a different location
You may want to install the software package in a different location than your home directory, for example in a project directory. To do so, use the following: 
```bash
eb <recipe.eb> --installpath /path/to/your/project/easybuild
```
Then, to get these modules available in your sessions, run
```bash
export RSNT_LOCAL_MODULEPATHS{{=
```/path/to/your/project/easybuild/modules}}
If you want to have this available by default in your sessions, you can add this command to your <code>.bashrc</code> file in your home.

= Additional resources =

- Webinar [<i>Building software on Compute Canada clusters using EasyBuild</i>](https://westgrid.github.io/trainingMaterials/getting-started/#building-software-with-easybuild) (recording and slides)
- Our staff-facing documentation [is available here](https://github.com/ComputeCanada/software-stack/blob/main/doc/easybuild.md).
- Many [tutorials](https://easybuild.io/tutorial/) on EasyBuild are available