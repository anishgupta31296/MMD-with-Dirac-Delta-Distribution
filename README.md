# MMD-with-Dirac-Delta-Distribution
The research paper for this work can be found https://drive.google.com/file/d/1_u_BKUML-D6mjpw3uPK9UCcD3KxnctdF/view?usp=sharing

The main.py files is used to run the algorithm and other baselines
In that file, line 83 defines which algorith to use
Planner(param=1,samples_param=25,optimizer='KLD',device='cuda:0',gaussian_approximation=False)
optimizer can be 'KLD', 'PVO', 'MMD_Dirac_Delta'
The parameter gaussain_approximation decides whether to use gaussian approximation for the non-parametric noise

The following explains why our algorithm works better than the other baselines:
https://youtu.be/BBQatGRawQM

Effect of value of RBF kernels hyperparameter:
1. Gamme = 0.1
https://user-images.githubusercontent.com/27779024/166189868-536f34eb-e86d-4189-8010-594894c7e97f.mp4
https://user-images.githubusercontent.com/27779024/166189874-08d7f802-520b-4775-9eb5-cde74b91378c.mp4

