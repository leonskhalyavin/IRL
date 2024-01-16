function [reward_learner,policy_learner,error_store] = maxmargin_irl(transition_probability,gamma,features,s0,reward_expert,policy_expert)
    %% Optimization parameters
    max_iter = 100;
    eps = .0001;
    M = 200;    % Number of trajectories
    H = 20;     % Trajectory length
    n_simulations = 100000;   % Monte carlo optimization
        
    [n_states,~,~] = size(transition_probability);
    [n_features,~] = size(features);
    fe_learner_store = zeros(n_states,max_iter);
    error_store = zeros(max_iter,1);
    
    % Find expert feature expectation
    fe_expert = feature_expectation(transition_probability,policy_expert,features,n_states,M,H,gamma,s0);
    
    for k = 1:max_iter
        [reward_learner,t] = monte_carlo_optim(n_features,fe_expert,fe_learner_store,n_simulations,k);
        if t <= eps && t ~= 0
            disp(['t = ' num2str(t)])
            break;
        end
        [~,policy_learner] = value_iteration(transition_probability,reward_learner,gamma);
        fe_learner = feature_expectation(transition_probability,policy_learner,features,n_states,M,H,gamma,s0);
        fe_learner_store(:,k) = fe_learner;
        % error_store(k) = norm(reward_expert - reward_learner');
        error_store(k) = t;
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
    
    function [reward_learner, t] = monte_carlo_optim(n_features,fe_expert,fe_learner_store,n_simulations,k)
        alpha_simulations = rand(n_simulations,n_features);
        alpha_simulations = alpha_simulations./sum(alpha_simulations,2);
        margins = zeros(k,n_simulations);
        t = -Inf;
        for i = 1:k
            fe_learner = fe_learner_store(:,k);
            margins(i,:) = -1*alpha_simulations*(fe_learner - fe_expert);
        end
        margins = min(margins);
        [M,I] = max(margins);
        reward_learner = alpha_simulations(I,:);
        t = M;
    end
end

