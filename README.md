# Codes of paper *'Hao, Wenjian, Bowen Huang, Wei Pan, Di Wu, and Shaoshuai Mou. "Deep Koopman learning of nonlinear time-varying systems." Automatica 159 (2024): 111372.'*

__Description__ <br />
- *Started on Sep/2021, last revision on Jan/2022.*
- **Author: Wenjian Hao, PhD, AAE, Purdue University.**
- *This code is about the first example in the paper, readers can refine the codes accordingly based on their research.*
- If the paper or codes help your research or other projects please cite:<br />
```
  @article{hao2024deep,
  title={Deep Koopman learning of nonlinear time-varying systems},
  author={Hao, Wenjian and Huang, Bowen and Pan, Wei and Wu, Di and Mou, Shaoshuai},
  journal={Automatica},
  volume={159},
  pages={111372},
  year={2024},
  publisher={Elsevier}
}
```

- __The goal of this project is to learn the dynamics of various nonlinear time-varying systems (NTVS).__ <br />

__Dependencies__ <br />
- *torch, numpy, os, scipy, matplotlib, joblib.*

__Method theory__<br />
- *Koopman operator and deep learning.*<br />

__Usage__<br />
- *Run main.py to learn and save the learned dynamics, then run the plot_comparison.py for results visualization and comparison.*<br />
- *Parameters are defined in config.py.*<br />
- *In folder '/SavedResults/', we include the plots data of the published paper. In particular, /SavedResults/SavedResults08/ contains the plots data with changing rate \gamma=0.8.*

__Possible extensions based on this codes__<br />
- *Change the dynamics by replacing the codes in block 'data generation' of main.py based on your research or projects.*<br />
- *Add the control inputs following the paper (changes need to be made in utils.py).*<br />
- *Develop various model-based control using the learning dynamics matrices A, B, and C and neural networks, see paper 'Han, Y., Hao, W., & Vaidya, U. (2020, December). Deep learning of Koopman representation for control. In 2020 59th IEEE Conference on Decision and Control (CDC) (pp. 1890-1895). IEEE.' for an example of developing model predictive control based on the learned deep Koopman model.*<br />

