# Make

## Description
[make](http://en.wikipedia.org/wiki/Makefile) is a utility that automatically builds files, such as executables or libraries, from other files, such as source code.

The <tt>make</tt> command interprets and executes the instructions within a file named <tt>makefile</tt>. Unlike a simple script, <tt>make</tt> only executes the commands that are necessary. The goal is to arrive at a result (compiled or installed software, formatted documentation, etc.) without needing to redo all steps. 

The <tt>makefile</tt> contains information on *dependencies*. 
For example, if the <tt>makefile</tt> indicates that an object (<tt>.o</tt>) file depends on a source file, and the source file has changed, then the source file is recompiled to update the object file.
In the same way, if an executable depends on any object files which have changed then the linking step will be rerun to update the executable. 
All dependencies must be included in the <tt>makefile</tt>. Then it is not necessary to recompile all files for every modification; the <tt>make</tt> command takes care of recompiling and relinking only what is necessary.

## Examples for using make
The main argument of the <tt>make</tt> command is the *target*. The *target* may be the name of some file that <tt>make</tt> should build, or it may be an abstract target such as *all*, *test*, *check*, *clean*, or *install*. 
The targets that are available depend on the contents of the <tt>makefile</tt>, but the ones just listed are conventional and are
specified in many makefiles. If <tt>make</tt> is invoked with no target specified, like so:
```bash
make
```
then the typical behaviour is to construct everything, equivalent to:
```bash
make all
```

The *test* or *check* targets are generally used to run tests to validate if the application or compiled library functions correctly. Usually these targets depend on the *all* target. Hence you can verify the compilation using
```bash
make all && make check
```
or
```bash
make all && make test
```

The *clean* target erases all previously compiled binary files to be able to recompile from scratch. There is sometimes also a *distclean* target, which not only deletes files made by <tt>make</tt>, but also files created at configuration time by [configure](Autotools.md) or [cmake](CMake.md). So to clean the compilation directory, you can usually run
```bash
make clean
```
and sometimes
```bash
make distclean
```

The *install* target normally installs a compiled program or library. Where the installation is put depends on the <tt>makefile</tt>, but can often be modified using an additional *prefix* parameter, like this:
```bash
make install prefix{{=
```$HOME/PROGRAM}}

The targets <tt>all, test, check, clean, distclean</tt> and <tt>install</tt> are only conventions and a <tt>makefile</tt> author could very well choose another convention. To get more information on typical target names, notably supported by all GNU applications, visit [this page](http://www.gnu.org/software/make/manual/make.html#Standard-Targets). Options to configure installation and other directories are [listed here](http://www.gnu.org/software/make/manual/make.html#Directory-Variables).

## Example of a <tt>Makefile</tt>
The following example, of general use, includes a lot of explanations and comments. For a detailed guide on how to create a <tt>makefile</tt>, visit [the GNU Make web site](http://www.gnu.org/software/make/manual/make.html#Introduction).

**File: Makefile**
```make
1. Makefile to easily update the compilation of a program (.out)
1. --------
1. 1. by Alain Veilleux, 4 August 1993
1. Last revision : 30 March 1998
1. 1. GOAL AND FUNCTIONING OF THIS SCRIPT:
1. Script in the form of a "Makefile" allowing to update a program containing
1. multiple separated routines on the disk. This script is not executed by itself,
1. but is instead read and interpreted by the "make" command. When it is called,
1. the "make" command verifies the dates of the various components your program is
1. built from. Only routines that were modified after the last compilation of the
1. program are recompiled in object form (files ending in .o). Recompiled .o files
1. are subsequently linked together to form an updated version of the final program.
1. 1. TO ADAPT THIS SCRIPT TO YOUR PROGRAM:
1. Modify the contents of the variables hereunder. Comments will guide you how and
1. where.
1. 1. USING "make" ON THE UNIX COMMAND LINE:
1. 1- Type "make" to update the whole program.
1. 2- Type "make RoutineName" to only update the RoutineName routine.
1. #====================  Definition of variables  =====================
1. Remark : variables are sometimes called "macros" in Makefiles.

1. Compiler to use (FORTRAN, C or other)
CompilerName= xlf

1. Compilation options: the below options are usually used to compile FORTRAN
1. code. You can assign other values than those suggested
1. in the "CompilationOptions" variables.
#CompilationOptions= -O3
1. Remove the below "#" to activate compilation in debug mode
#CompilationOptions= -g
1. Remove the below "#" to use "gprof", which indicates the computation time in
1. each subroutine
#CompilationOptions= -O3 -pg

1. List of routines to compile: here we list all object files that are needed.
1. Put a "\" at the end of each line that if you want to continue the list of
1. routines on the following line.
ObjectFiles= trnb3-1.part.o mac4251.o inith.o dsite.o initv.o main.o \
             entree.o gcals.o defvar1.o defvar2.o magst.o mesure.o

1. Name of the final executable
ProgramOut= trnb3-1.out
#=====  End of variable definitions =====
#===============  There is nothing to change below this line  =============


1. Defines a rule: how to build an object file (ending in ".o")
1. from a source file (ending in ".f")
1. note: "$<" symbols will be replaced by the name of the file that is compiled
1. Compiling Fortran files:
.f.o:
	$(CompilerName) $(CompilationOptions) -c $<

1. Defines a rule: how to build an object file (ending in ".o")
1. from a source file (ending in ".c")
1. note: "$<" symbols will be replaced by the name of the file that is compiled
1. Compiling C files:
.c.o:
	$(CompilerName) $(CompilationOptions) -c $<

1. Defines a rule: how to build an object file (ending in ".o")
1. from a source file (ending in ".C")
1. note: "$<" symbols will be replaced by the name of the file that is compiled
1. Compiling C++ files:
.C.o:
	$(CompilerName) $(CompilationOptions) -c $<

1. Dependencies of the main executable on the object files (".o") it is built from.
1. The dependency of object files on source files (".f" and ".c") is implied by the above
1. implicit rules.
$(ProgramOut): $(ObjectFiles)
	$(CompilerName) $(CompilationOptions) -o $(ProgramOut) \
							$(ObjectFiles)
```