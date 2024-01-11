function alpha = linear_irl(mdp)
    max_iter = 100;
    alpha = zeros(mdp.n_states,1);  % Output reward

    % Initialize random policy
    pi = ones(mdp.n_actions,1);

    for k = 1:max_iter
        % Iterate through solving LP
        A = eye(mdp.n_states);
        b = ones(mdp.n_states,1);
        
        % Find f
        

        alpha = linprog(f,A,b);

        % Compute optimal policy using new reward
        pi_new = value_iteration();
    end

    function x = p(x)
        % Penalize solutions out of constraint
        if x >= 0
            x = x;
        else
            x = 2*x;
        end
    end
end

