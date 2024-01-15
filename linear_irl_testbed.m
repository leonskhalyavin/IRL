% Initialize states
n_states = 5;
n_actions = 3;
gamma = 1;
max_iter = 100;
eps = .0001;
M = 200;    % Number of trajectories
H = 50;     % Trajectory length
n_simulations = 100000;   % Monte carlo optimization

%%
% Initialize transition matrix, features, and initial distribution
transition_probability = rand(n_states,n_actions,n_states);
features = eye(n_states);
s0 = rand(n_states,1);
s0 = s0/sum(s0);

% Normalize transition matrix
for state = 1:n_states
    for action = 1:n_actions
        transition_probability(state,action,:) = transition_probability(state,action,:)/sum(squeeze(transition_probability(state,action,:))); 
    end
end

% Initialize reward
% reward_expert = rand(n_states,1);
% reward_expert = reward_expert/sum(reward_expert);
reward_expert = zeros(n_states,1);
reward_expert(n_states) = 1;

% Find optimal policy
[value_expert,policy_expert] = value_iteration(transition_probability,reward_expert,gamma);

% Plot expert
plot_functions(reward_expert,policy_expert,'Expert',1);

%%
% Perform inverse RL
reward_learner = rand(n_states,1);
reward_learner = reward_learner/sum(reward_learner);
[value_learner,policy_learner] = value_iteration(transition_probability,reward_learner,gamma);

% Find expert feature expectation
fe_expert = feature_expectation(transition_probability,policy_expert,features,n_states,M,H,gamma,s0);

% Initialize LP constraints
A = eye(n_states);
b = ones(n_states,1);
A_eq = ones(1,n_states);
b_eq = 1;
lb = zeros(n_states,1);
ub = ones(n_states,1);

% Optimize
fe_learner = feature_expectation(transition_probability,policy_learner,features,n_states,M,H,gamma,s0);
fe_learner_store = zeros(n_states,max_iter+1);
fe_learner_store(:,1) = fe_learner;

err_store = zeros(max_iter,1);
for k = 1:max_iter
    c = zeros(n_states,1);
    for i = 1:k
        c = c-p(fe_expert-fe_learner_store(:,k),reward_learner,1);
    end
    % reward_learner = linprog(c,[],[],[],[],lb,ub);
    reward_learner = monte_carlo_optim(c,n_simulations);
    [value_learner,policy_learner] = value_iteration(transition_probability,reward_learner,gamma);
    fe_learner = feature_expectation(transition_probability,policy_learner,features,n_states,M,H,gamma,s0);
    err = abs(value_from_fe(fe_expert,reward_learner) - value_from_fe(fe_learner,reward_learner));
    err_store(k) = err;
    if err < eps
        disp(['Error = ' num2str(err)])
        break;
    else
        fe_learner_store(:,k+1) = fe_learner;
    end
end

% Plot learner
plot_functions(reward_learner,policy_learner,'Learner',2);

% Plot error
plot_error(err_store,3);

%%
function plot_functions(reward,policy,name,figure_number)
    figure(figure_number); subplot(2,1,1); bar(reward); title([name ' Reward'])
    subplot(2,1,2); bar(policy); title([name ' Policy'])
end

function plot_error(err_store,figure_number)
    figure(figure_number); plot(1:length(err_store),err_store); title('Error');
end

function x = p(x,reward,m)
    if reward'*x < 0
        x = m.*x;
    end
end

function V = value_from_fe(fe,reward)
    V = reward'*fe;
end

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

function x = monte_carlo_optim(c,n_simulations)
    x_sim = rand(n_simulations,length(c));
    x_sim = x_sim ./ sum(x_sim,2);
    values = x_sim * c;
    [~,I] = min(values);
    x = x_sim(I,:)';
end