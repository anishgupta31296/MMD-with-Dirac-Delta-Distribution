# MMD-with-Dirac-Delta-Distribution
The research paper for this work can be found https://drive.google.com/file/d/1_u_BKUML-D6mjpw3uPK9UCcD3KxnctdF/view?usp=sharing

The main.py files is used to run the algorithm and other baselines
In that file, line 83 defines which algorith to use
Planner(param=1,samples_param=25,optimizer='KLD',device='cuda:0',gaussian_approximation=False)
optimizer can be 'KLD', 'PVO', 'MMD_Dirac_Delta'
The parameter gaussain_approximation decides whether to use gaussian approximation for the non-parametric noise
