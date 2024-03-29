% Initialize states
n_states = 5;
n_actions = 3;
gamma = 1;
max_iter = 100;
eps = .0001;
M = 200;    % Number of trajectories
H = 20;     % Trajectory length
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

%%
% Apprenticeship learning
% reward_learner = rand(n_states,1);
% reward_learner = reward_learner/sum(reward_learner);
fe_learner = zeros(n_states,1);
fe_learner_store = zeros(n_states,max_iter);
t_store = zeros(max_iter,1);

% Find expert feature expectation
fe_expert = feature_expectation(transition_probability,policy_expert,features,n_states,M,H,gamma,s0);

for k = 1:max_iter
    [reward_learner,t] = monte_carlo_optim(reward_learner,fe_expert,fe_learner_store,n_simulations,k);
    t_store(k) = t;
    if t <= eps && t ~= 0
        disp(['t = ' num2str(t)])
        break;
    end
    [value_learner,policy_learner] = value_iteration(transition_probability,reward_learner,gamma);
    fe_learner = feature_expectation(transition_probability,policy_learner,features,n_states,M,H,gamma,s0);
    fe_learner_store(:,k) = fe_learner;
end

% Plot expert
plot_functions(reward_expert,policy_expert,'Expert',1);

% Plot learner
plot_functions(reward_learner,policy_learner,'Learner',2);

% Plot t
plot_t(t_store,3);

%%
function plot_functions(reward,policy,name,figure_number)
    figure(figure_number); subplot(2,1,1); bar(reward); title([name ' Reward'])
    subplot(2,1,2); bar(policy); title([name ' Policy'])
end

function plot_t(t_store,figure_number)
    figure(figure_number); plot(1:length(t_store),t_store); title('Error');
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

function [reward_learner, t] = monte_carlo_optim(reward_learner,fe_expert,fe_learner_store,n_simulations,k)
    % alpha_simulations = rand(n_simulations,length(reward_learner));
    % margins = zeros(n_simulations,1);
    % for i = 1:k
    %     margin_k = alpha_simulations*(fe_expert - fe_learner_store(:,k));
    %     margin_k = margin_k + (2.*(margin_k < 0));
    %     margins = margins + margin_k;
    % end
    % [M,I] = max(margins);
    % reward_learner = alpha_simulations(I,:);
    alpha_simulations = rand(n_simulations,length(reward_learner));
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