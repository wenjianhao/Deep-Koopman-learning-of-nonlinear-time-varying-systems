# Paper codes of paper *'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." Automatica 159 (2024): 111372.'*

__Description__ <br />
- *Started at 09/2021, last Revision: Jan 2022.*
- **Author: Wenjian Hao, PhD, AAE, Purdue University.**
- *This code is about the first example in the paper, readers can refine the codes accordingly based on their research.*
- *If the paper or codes help your research or other projects please cite @article{hao2024deep,
  title={Deep Koopman learning of nonlinear time-varying systems},
  author={Hao, Wenjian and Huang, Bowen and Pan, Wei and Wu, Di and Mou, Shaoshuai},
  journal={Automatica},
  volume={159},
  pages={111372},
  year={2024},
  publisher={Elsevier}
}.*

__Third Dependencies__ <br />
- *Numpy, Pytorch, Casadi, CUDA, Control, Openai-Gym*
- *Windows Or Ubuntu Or Macos*
- *Python*

__The goal of this project__ <br />
- *To learn the dynamics of various nonlinear time-varying systems (NTVS)*

__Method theory__<br />
- *Koopman Operator and deep learning*<br />

__Usage__<br />
- *Data Generating*<br />
- *Deep Koopman Training and Optimal Control designing <br/> (Working on building our own private controller packages, casadi versions) [Have finished {MPC, LQR, FiniteLQR}]*<br />

__Things have achieved__<br />
- *Theorem 1: As the parameters of NN lifting go to the infinity, the K will converge (Done)*<br />
- *Theorem 2: Error Boundary (Done)*<br />
- *Simulation for NTVS identification (Done)*<br />
- *Simulation for Cartpole with time-varying friction coefficient of the ground (Done)*<br />
