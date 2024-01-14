function [V, policy] = value_iteration(T, R, gamma)
    % Get the number of states (S) and actions (A)
    [S, A, ~] = size(T);
    
    % Initialize value function and policy
    V = zeros(S, 1);
    policy = ones(S, 1);  % Default policy (action 1 for all states)
    
    % Perform value iteration
    max_iterations = 1000;
    tolerance = 1e-6;
    
    for iteration = 1:max_iterations
        prev_V = V;
        
        % Update value function using Bellman equation
        for s = 1:S
            % Calculate the expected future rewards for each action
            expected_rewards = zeros(A, 1);
            for a = 1:A
                expected_rewards(a) = R(s) + gamma * sum(squeeze(T(s, a, :))' * prev_V);
            end
            
            % Update value function with the maximum expected reward
            [V(s), policy(s)] = max(expected_rewards);
        end
        
        % Check for convergence
        if max(abs(V - prev_V)) < tolerance
            break;
        end
    end
end