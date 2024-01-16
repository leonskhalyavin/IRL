function [reward_learner,policy_learner,error] = maxent_irl(transition_probability,gamma,features,s0,reward_expert,policy_expert)
    %% Optimization parameters
    max_iter = 100;
    eps = .0001;
    M = 200;    % Number of trajectories
    H = 20;     % Trajectory length
    n_simulations = 100000;   % Monte carlo optimization
    delta = Inf;

    [n_states,~,~] = size(transition_probability);
    [n_features,~] = size(features);
    error_store = zeros(max_iter,1);

    % Find expert feature expectation
    fe_expert = feature_expectation(transition_probability,policy_expert,features,n_states,M,H,gamma,s0);
    
    % Initialize reward
    reward_learner = rand(n_states,1);

    for k = 1:max_iter
        reward_learner_old = reward_learner;
        
        % Compute log-likelihood
        expected_svf = 0;

        if delta < eps
            break;
        end
    end

    %% Functions
    function fe = feature_expectation(transition_probability,policy,features,n_states,M,H,gamma,s0)
        fe = zeros(n_states,1);
        for t = 1:M
            cumsum_s0 = cumsum(s0);
            r = rand();
            init_states = find(r < cumsum_s0);
            state = init_states(1);
            fe = fe + features(:,state);
            for s = 1:H
                action = policy(state);
                r = rand();
                cumsum_distribution = cumsum(squeeze(transition_probability(state,action,:)));
                state = find(r < cumsum_distribution);
                state = state(1);
                fe = fe + gamma^(s).*features(:,state);
            end
        end
        fe = fe ./ M;
    end
end

