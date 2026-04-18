The goal of the project was to implement classical genetic algorithm for minimization and maximization of functions of multiple arguments.

TO RUN ONE MUST HAVE INSTALLED: 
python; libraries: matplotlib, math, numpy, random, time, tkinter
After installing all of that, it is enough to copy the repository locally and run the python file: GUI.py

Features of the algorithm: 
- binary representations of genes
- cross over methods implemented: shuffle, towpoint, onepoint, threepoint, uniform, replacement, grain, devastating
- selection methods implemented: best, worst, random, roulette, ranking, tournament
- mutation techniques: onepoint, twopoint, edge
- available option to keep the best individual: elitism
- specific functions for testing the algorithm were added (Rastrigin, Hypersphere, Dejong3, Rosenbrock, Hyperellipsoid, Schwefel,
Ackley, Michalewicz), the list can be further extneded easily if some new function needs to be optimized

To ease the experimenting was developed UI, in which user can set up all necessary options and parameters to run an experiment.
