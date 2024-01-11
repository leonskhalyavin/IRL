function [value, policy] = value_iteration(T,gamma,reward)
    % Perform value iteration
    max_iter = 100;

    n_states = size(T,1);
    n_actions = size(T,2);

    value = reward;
    policy = zeros(size(n_states),1);

    for iter = 1:max_iter
        for i = 1:n_states
            
        end
    end
end

