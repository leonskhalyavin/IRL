% Inverted pendulum for LQR system
% x = [x ; x_dot ; theta ; theta_dot]

%% Compute forward problem for expert
% Initialize state space model
A1 = [0 1 0 0 ;
    0 -.1 3 0 ;
    0 0 0 1 ;
    0 -.5 30 0];

B1 = [0 ; 2 ; 0 ; 5];

C1 = [1 0 0 0 ;
     0 0 1 0];

D1 = 0;

% Initialize cost functions
Q1 = [1 0 0 0 ;
     0 0 0 0 ;
     0 0 1 0 ;
     0 0 0 0];

R1 = 1;

% Compute gain matrix
K1 = lqr(A1,B1,Q1,R1);

% Plot step response
sys1 = ss(A1-B1*K1,B1,C1,D1);
step(sys1);

%% Perform inverse problem for expert
% Initialize trajectories
n_trajectories = 100;
trajectory_length = 50;
x0_min = -1;
x0_max = 1;
dt = .1;
[x_trajectories,u_trajectories,x0_trajectories] = ...
    generate_trajectories(n_trajectories,trajectory_length,A1,B1,K1,x0_min,x0_max,dt);

% Perform IOC from sampled trajectories
max_iter = 1000;
[Q_learner,R_learner] = maxmargin_IOC(x_trajectories,u_trajectories,x0_trajectories,A1,B1,max_iter,dt);

