function [x_trajectories,u_trajectories,x0_trajectories] = generate_trajectories(n_trajectories,trajectory_length,A,B,K,x0_min,x0_max,dt)
%{
Generate n random trajectories of the system x_dot = (A-BK)x + Bu
INPUTS:
    - n_trajectories: number of trajectories
    - trajectory_length: length of the trajectories
    - A: A matrix (nxn)
    - B: B matrix (nxm)
    - K: K matrix (nxm)
    - x0_min: min x0 value (double 1x1)
    - x0_max: max x0 value (double 1x1)
    - dt: time derivative (double 1x1)
OUTPUTS:
    - x_trajectories: set of trajectories of x values (nxn_trajectories)
    - u_trajectories: set of trajectories of u values (mxn_trajectories)
    - x0_trajectories: list of all x0 for trajectories
%}
    n = size(A,2);
    m = size(B,2);
    x_trajectories = {};
    u_trajectories = {};
    x0_trajectories = zeros(n,n_trajectories);
    for i = 1:n_trajectories
        x_trajectory = zeros(n,trajectory_length);
        u_trajectory = zeros(m,trajectory_length);
        x = x0_min*rand(n,1) + x0_max;
        x0_trajectories(:,i) = x;
        for j = 1:trajectory_length
            x_dot = (A-B*K)*x + B*(-K*x);
            x = x + x_dot*dt;
            x_trajectory(:,j) = x;
            u_trajectory(:,j) = -K*x;
        end
        x_trajectories{i} = x_trajectory;
        u_trajectories{i} = u_trajectory;
    end
end

