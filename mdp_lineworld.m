function [mdp] = mdp_lineworld(n_states,p,discount)
    mdp.max_iter = 1000;

    mdp.n_states = n_states;
    mdp.p = p;
    mdp.discount = discount;

    mdp.actions = [-1, 0, 1];
    mdp.n_actions = length(mdp.actions);
    
    % Create transition matrix
    mdp.T = zeros(mdp.n_states,mdp.n_actions,mdp.n_states);
    for i = 1:mdp.n_states
        for a = 1:mdp.n_actions
            for j = 1:mdp.n_states
                mdp.T(i,a,j) = transition_probability(mdp,i,a,j);
            end
        end
    end

    % Create reward function
    % mdp.terminal = randi(mdp.n_states);
    mdp.terminal = round(mdp.n_states / 2);
    mdp.reward = set_reward(mdp);

    % Solve for optimal policy
    mdp.policy = optimal_policy(mdp);

    function P = transition_probability(mdp,i,a,j)
        a = mdp.actions(a);
        
        if i+a == j % Intended move
            P = 1 - mdp.p;
        elseif (i+a > mdp.n_states || i+a < 1) && i == j % Hit a wall
            P = 1;
        elseif ~(i+a > mdp.n_states || i+a < 1) && (i+1 == j || i-1 == j) % Random probability
            if i == 1 || i == mdp.n_states % At a wall
                P = mdp.p;
            else % In the middle
                P = mdp.p / 2;
            end
        else
            P = 0;
        end
    end

    function reward = set_reward(mdp)
        reward = zeros(mdp.n_states,1);
        reward(mdp.terminal) = 1;
    end

    function policy = optimal_policy(mdp)
        policy = zeros(mdp.n_states,1);
        for state = 1:mdp.n_states
            if state < mdp.terminal
                policy(state) = 1;
            elseif state == mdp.terminal
                policy(state) = 0;
            else
                policy(state) = -1;
            end
        end
    end


end

