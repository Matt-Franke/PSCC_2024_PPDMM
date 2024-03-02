# Code Base for PSCC 2024 Paper: Privacy-Preserving Distributed Market Mechanism for Active Distribution Networks
# Authors: Matthias Franke (ETH), Ognjen Stanojev (ETH), Lesia Mitridate (DTU), Gabriela Hug (ETH)
# Main Developer: Matthias Franke (ETH)

This repository contains the code for the simulations of this paper, as well as the various results in the /data folder.
It is provided AS-IS. Since it is a proof-of-concept research code and not an open-source library, the
resulting code is of limited optimisation. 
Notably, on higher complexity settings, it is quite RAM and CPU intensive, and will crash if not enough is available.
The results were generated on a single Windows 10 laptop with an AMD Ryzen 5 4500U, 2375 MHz, 6-core processor and 16 GB of RAM.
More cores, especially 8-core processors, will be able to run the code faster.
However, less than 16 GB of RAM will cause crashes.
The code was developed on Ubuntu and Windows 10, and was able to run fine.
Other Operating Systems and OS Versions have not been tested.


## Description
The purpose of the project is to design a local flexibility market platform for heterogeneous DERs, 
and to expand and demonstrate the performance of SMPC protocols under diverse sets of market schemes, 
growing system complexity and uncertainty, and more capable adversaries.

## Setup
To run the program, you need the following:
1. Python 3.10
2. If you are running on Windows, you may need the Microsoft C++ Build Tools, Version 14.0 or higher
2. Install the dependecies (preferably in a virtual environment) via the requirements.txt
3. Ensure the folders "runtime/temp" and "runtime/states" exist prior to execution and ideally also empty

## Running the main protocol

**NOTE: As a Proof-of-Concept, this protocol is rather resource intensive, and so a lack of RAM may cause the script to
crash at higher complexities!**


1. Open "main.py" where in the section labelled "Setup" you can select the main parameters
   1. PLOT: True/False. Selects whether to plot the results of the market clearing
   2. MODE: 0-3. Selects which of version of the current solver to use, which describes the complexity of the formulation.
      1. Mode 0 = Deterministic, no chance constraining
      2. Mode 1 = Linear chance constraining of nodal quantities, but not inter-nodal quantities (A Quadratic Program)
      3. Mode 2 = Linear chance constraining of nodal quantities, but not inter-nodal quantities (A Second-Order Cone Program)
      4. Mode 3 = Full chance constraining (A Second-Order Cone Program)
      5. Mode 4 = A prototype of nodal frequency response, not used in full paper. **DO NOT USE**
   3. SCENARIO: 0-5. Selects which scenario to run. See main.py for Details
   4. MULTI: 0.0,0.5,1.5. Selects the multiplier to use in the current scenario. For baseline use Multi=0.0
   5. BAL = 0/1. Selects whether ADMM balances are calculated using dual (0) or global (1) prices.
   6. SOLVER: SOLVERS[0],SOLVERS[1],SOLVERS[2]. Selects which family of solvers to use (Central, ADMM, SEC-ADMM).
   7. SECURE: False/True. Whether to run security features. **Set to true ONLY if SOLVERS[2] is used!**
   8. COV: False/True. Whether to run the covariance estimator at the end. **Since this is synthetic and quite long, it is very much optional.**
2. Results are then as follows:
   1. The output into the console will give high-level overview of operation
   2. Two files in data/Processed:
      1. results_[The specific solver]_DATA_[Scenario]_[Multiplier].csv with the full results of the market clearing
      2. results_[The specific solver]_TIME.csv for the per-iteration results, such as convergence criteria.
   3. The otherwise hidden output of the solvers (the ADMM ones): 
      1. "runtime/temp/results_[The specific solver]_DATA_[Scenario]_[Multiplier].txt"


## FAQ
A non-exhaustive list of questions

1. Why is this SMPC script not working but crashing with weird error?
   1. I was using a minor release of "mpyc", namely version 0.8.15 for large parts of the project
   2. I upgraded to "mpyc 0.9" for ease of installation for the main files, but have not retested older files
   3. Version 0.9 appears to have changed some automatic timeouts which results in weird crashes
2. Why does this SMPC script not finish?
   1. Two options:
      1. The machine does not have enough processing power, particularly enough cores
         1. The scripts were tested on a 6 core, midlevel laptop and work. They work even better with more cores
         2. They do not work well on older machines or machines with fewer cores.
         3. Since we use multiprocessing we have really high CPU usage and this cannot be avoided, unfortunately.
      2. It just got stuck and the reason is deep inside the MPyC library
         1. The current version of the SMPC files in runtime are not susceptible to this and run just fine
         2. However, this has appeared repeatedly, in particular for SMPC multiplications for the balance calculations
         3. MPyC added secure array support in newer versions, so that could fix it fully, which have not been included
         4. Individual nodes seem to sometimes just have low-level issues and if one fails, the SMPC script does not finish
3. Why is the folder "modules" both at the highest level and in runtime?
   1. This project was developed in Pycharm, which is very helpful in how it adds folders to the system's path
   2. This means scripts were a bit better than they were supposed to at finding the modules folder even if they were one level done
   3. This is a band-aid solution to ensures the scripts run with other IDEs besides Pycharm
4. The script could not find files in either the "temp" or the "states" subfolder of "/runtime"?
   1. Make sure those folders exist. The program creates the files within them automatically, but not the folders
   2. Since git does not store folders if they are empty, we have left a file in each to make sure the folders exist
5. I would like to run this on a cluster, does that work?
   1. The program uses Python itself for multiprocessing as well as MPyC which has it in-built
   2. So cluster computing would be pretty good and there would be no fighting a self-made multi-processor
   3. However, the main script uses .json files to shuttle data between CVXPY optimisation and MPyC SMPC
   4. So there is a lot of I/O, albeit with very small files, which computing nodes often struggle with
6. I would like to debug the ADMM code, does that work?
   1. It does, but running in debug is  worse performance-wise than just running normally
   2. So there may be buffer overflows and such, but these are due to the debugging not the actual script itself
7. Any other questions?
   1. While my development will obviously end with the paper, I am happy to answer any questions down the line about the code.
   2. Furthermore the code is in Python and uses only two libraries that are a bit unknown (cvxpy and mpyc)
   3. So the code should be very easily comprehensible otherwise as it has been extensively commented

## Notes
Using SCIP 8.0.3 under Apache 2.0 License
