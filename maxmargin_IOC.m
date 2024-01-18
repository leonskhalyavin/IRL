function [Q_learner,R_learner] = maxmargin_IOC(x_trajectories,u_trajectories,x0_trajectories,A,B,max_iter,n_trajectories,trajectory_length,x0_min,x0_max,dt)
%{
Find IOC from sampled trajectories
INPUTS:
    - x_trajectories: set trajectories of x values
    - u_trajectories: set of trajectories of u values
    - x0_trajectories: list of x0 trajectories
    - A: A matrix (nxn)
    - B: B matrix (nxm)
OUTPUTS:
    - Q_learner: Estimated positive semi-definite Q matrix
    - R_learner: Estimated positive semi-definite R matrix
%}
    J = @(x,u,Q,R) x'*Q*x + u'*R*u;
    n = size(A,2);
    m = size(B,2);

    % Estimate policy
    

    % % Initialize random Q and R
    % Q_store{1} = eye(n);
    % R_store{1} = eye(m);
    % 
    % for i = 1:max_iter
    %     % Generate trajectories for new policy
    %     Q = cell2mat(Q_store(1));
    %     R = cell2mat(R_store(1));
    %     K = lqr(A,B,Q,R);
    %     [x_trajectories_learner,u_trajectories_learner,x0_trajectories_learner] = ...
    %         generate_trajectories(n_trajectories,trajectory_length,A,B,K,x0_min,x0_max,dt);
    %     x_trajectories_learner_store{i} = x_trajectories_learner;
    %     u_trajectories_learner_store{i} = u_trajectories_learner;
    %     x0_trajectories_learner_store{i} = x0_trajectories_learner;
    %     % [Q_learner,R_learner] = cost_optimization(A,B,Q,R,K_store);
    % 
    %     % Find expert cost
    %     expert_cost = 0;
    %     for k = 1:n_trajectories
    %         x = cell2mat(x0_trajectories(k));
    %         expert_cost = expert_cost + J(x,zeros(m,1),Q,R);
    %         for t = 1:trajetory_length
    %             x = cell2mat(x_trajectories(k));
    %             u = cell2mat(u_trajectories(k));
    %             expert_cost = expert_cost + J(x,u,Q,R);
    %         end
    %     end
    % 
    % end

    function [Q_learner,R_learner] = cost_optimization(expert_trajectories,learner_trajectories,A,B,Q,R,K_store)

    end
end

