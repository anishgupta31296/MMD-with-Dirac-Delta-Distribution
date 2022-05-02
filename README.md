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
1. Gamma = 0.1

https://user-images.githubusercontent.com/27779024/166197857-b8b30ee4-784a-45e7-80cc-20b3f08b1640.mp4


2. Gamma = 1.0

https://user-images.githubusercontent.com/27779024/166197916-984095a8-fa72-4cef-8d6e-824d3da0050e.mp4


3. Gamma = 5.0

https://user-images.githubusercontent.com/27779024/166197974-4dee6c68-5451-4e0a-84be-a4f606fc65b7.mp4


Other Qualitative Results:
1. 2 obstacle case

https://user-images.githubusercontent.com/27779024/166198043-20213450-8b4b-4211-a47d-ff8a51884967.mp4


2. 3 obstacles case 

https://user-images.githubusercontent.com/27779024/166198079-6bfba900-5ecb-4582-9754-c41b989d3af1.mp4


3. 4 obstacles case

https://user-images.githubusercontent.com/27779024/166198104-77035dcb-d7cf-4fc5-8c47-8fbacb6f1c73.mp4
