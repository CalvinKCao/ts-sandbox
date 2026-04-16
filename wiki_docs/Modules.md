# Modules

In computing, a module is a unit of software that is designed to be independent, interchangeable, and contains everything necessary to provide the desired functionality.
<ref>[Wikipedia, "Modular programming"](https://en.wikipedia.org/wiki/Modular_programming)</ref>
The term "module" may sometimes have a more specific meaning depending on the context.
This page describes a few types of modules and suggests links to further documentation content. 

## Disambiguation

### Lmod modules

Also called "environment modules", Lmod modules are used to alter your (shell) environment so as to enable you to use a particular software package,
or to use a non-default version of certain common software packages such as compilers.  See [Using modules](Using_modules.md).

### Python modules

In Python, a module is a file of code (usually Python code) which can be loaded with the <code>import ...</code> or <code>from ... import ...</code> statements to provide functionality.  A Python package is a collection of Python modules; the terms "package" and "module" are frequently interchanged in casual use.
<ref>[Tutorialspoint.com, "What is the difference between a python module and a python package?"](https://www.tutorialspoint.com/What-is-the-difference-between-a-python-module-and-a-python-package)</ref>

Certain frequently used Python modules such as Numpy can be imported if you first load the <code>scipy-stack</code> Lmod module at the shell level.
See [SciPy stack](Python#SciPy_stack.md) for details.

We maintain a large collection of [Python "wheels."](Python#Available_wheels.md)
These are modules which are pre-compiled to be compatible with the [Standard software environments](Standard_software_environments.md).
Before importing modules from our wheels, you should create a [virtual environment](Python#Creating_and_using_a_virtual_environment.md).  

Python modules which are not in the <code>scipy-stack</code> Lmod module or in our wheels collection can be installed from the internet
as described in the [Installing packages](Python#Installing_packages.md) section.

## Other related topics

The main [Available software](Available_software.md) page is a good starting point. Other related pages are:
- [Standard software environments](Standard_software_environments.md): as of April 1, 2021, <code>StdEnv/2020</code> is the default collection of Lmod modules
- Lmod [modules specific to Niagara](modules_specific_to_Niagara.md)
- Tables of Lmod modules optimized for [AVX](Modules_avx.md), **[AVX2](Modules_avx2.md)** and **[AVX512](Modules_avx512.md)** [CPU instructions](Standard_software_environments#Performance_improvements.md)
- [Category *Software*](:Category:Software.md): a list of different software pages in this wiki, including commercial or licensed software

## Footnotes