# MMD-with-Dirac-Delta-Distribution
The research paper for this work can be found https://drive.google.com/file/d/1_u_BKUML-D6mjpw3uPK9UCcD3KxnctdF/view?usp=sharing

The main.py files is used to run the algorithm and other baselines
In that file, line 83 defines which algorith to use
Planner(param=1,samples_param=25,optimizer='KLD',device='cuda:0',gaussian_approximation=False)
optimizer can be 'KLD', 'PVO', 'MMD_Dirac_Delta'
The parameter gaussain_approximation decides whether to use gaussian approximation for the non-parametric noise

The following explains why our algorithm works better than the baselines:
https://youtu.be/BBQatGRawQM

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/BBQatGRawQM/maxresdefault.jpg)](https://www.youtube.com/watch?v=BBQatGRawQM)

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


4. 7 Obstacles case

https://user-images.githubusercontent.com/27779024/174946763-83b90ae0-9ddc-466b-9cdc-47967a45fc53.mp4

5 Obsacles Case with Different Noise Types:

ia) Noise:

![dist_1](https://user-images.githubusercontent.com/27779024/174952907-89501205-6a1d-4ca6-8d04-66eba1ac041d.png)

ib) Collision Avoidance Video:

https://user-images.githubusercontent.com/27779024/174952952-db4fe0e6-dbe7-488c-9559-01d0e7e0a7b8.mp4

iia) Noise:

![dist_7](https://user-images.githubusercontent.com/27779024/174953133-7095d33b-911e-4adf-a356-b031419b8143.png)

iib) Collision Avoidance Video:

https://user-images.githubusercontent.com/27779024/174953203-a8ccf4a4-e1b7-4e2f-a03d-8cec752412f3.mp4

iiia) Noise:

![dist_9](https://user-images.githubusercontent.com/27779024/174953266-756d8a57-f259-4d36-b40b-27eee2d1cbc5.png)

iiib) Collision Avoidance Video:

https://user-images.githubusercontent.com/27779024/174953322-24abde44-8fd4-4726-a5e6-6b3f8f599a09.mp4
