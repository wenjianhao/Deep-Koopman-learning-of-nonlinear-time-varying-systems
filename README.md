# Data Driven Control for Time-Varying Systems

**This project is started at 09/2021, Authors: Wenjian Hao (AAE, PhD, Purdue University), Bowen Huang (PosDoc, PNNL (Current), EE, PhD, IOWA STATE)** <br />

__Third Dependencies__ <br />
- *Numpy, Pytorch, Casadi,*

__The goal of this project__ <br />
- *Optimal Control for time-variant systems using data driven method*

__Current Method__<br />
- *Data Driven Control with Machine learning*<br />

__3 Folders__<br />
- *Data Collecting*<br />
- *Deep Koopman Training and Optimal Control designing (Working on building our own private controller packages, casadi versions) [MPC, LQR, FiniteLQR, finished]*<br />
- *Plots for papers (Making it a private package)*<br />

__Things to do in the Nov__<br />
- *Proof: As the parameters of NN lifting go to the infinity, the K will converge*<br />
- *Proof: Convergence of the time variant system*<br />
- *Lemma 1: As we get the new data, the new approxmated k is related to the previous k matrix (Done)*<br />
- *Lemma 2: Place the weight for the data, by the order of the time*<br />
