# Paper codes of paper *'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." Automatica 159 (2024): 111372.'*

__Description__ <br />
- *Started at 09/2021, last Revision: Jan 2022.*
- **Author: Wenjian Hao, PhD, AAE, Purdue University.**
- *This code is about the first example in the paper, readers can refine the codes accordingly based on their research.*
- *If the paper or codes help your research or other projects please cite: 
- @article{hao2024deep,<br />
  title={Deep Koopman learning of nonlinear time-varying systems},<br />
  author={Hao, Wenjian and Huang, Bowen and Pan, Wei and Wu, Di and Mou, Shaoshuai},<br />
  journal={Automatica},<br />
  volume={159},<br />
  pages={111372},<br />
  year={2024},<br />
  publisher={Elsevier}
}.*

- __The goal of this project__ <br />
- *To learn the dynamics of various nonlinear time-varying systems (NTVS)*

__Third Dependencies__ <br />
- *torch, numpy, os, scipy, matplotlib, joblib*

__Method theory__<br />
- *Koopman Operator and deep learning*<br />

__Usage__<br />
- *Parameters are defined in config.py*<br />
- *In folder '/SavedResults/', we include the plots data of the published paper, in particular, /SavedResults/SavedResults08/ contains the plots data with changing rate \gamma=0.8. *<br />

__Possible extension based on this data__<br />
- *Change the dynamics by replacing the codes in main.py-data generation based on your research or projects.*<br />
- *Add the control inputs following the paper.*<br />
- *Develop various model-based control using the learning dynamics matrices A, B, and C and neural networks, see paper 'Han, Y., Hao, W., & Vaidya, U. (2020, December). Deep learning of Koopman representation for control. In 2020 59th IEEE Conference on Decision and Control (CDC) (pp. 1890-1895). IEEE.'*<br />

