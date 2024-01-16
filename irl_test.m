%% Settings
RANDOM_MDP = true;
RANDOM_REWARD = false;
PARTIALLY_OBSERVABLE = false;

%% Create MDP
if RANDOM_MDP
    % Initialize states
    n_states = 5;
    n_actions = 3;
    gamma = 1;

    features = eye(n_states);
    s0 = rand(n_states,1);
    s0 = s0/sum(s0);

    % Create transition matrix
    transition_probability = rand(n_states,n_actions,n_states);
    for state = 1:n_states
        for action = 1:n_actions
            transition_probability(state,action,:) = transition_probability(state,action,:)/sum(squeeze(transition_probability(state,action,:))); 
        end
    end

    % Create observation matrix
    n_observations = 5;
    observation_probability = rand(n_states,n_actions,n_observations);
    for state = 1:n_states
        for action = 1:n_actions
            observation_probability(state,action,:) = observation_probability(state,action,:)/sum(squeeze(observation_probability(state,action,:)));
        end
    end
    b0 = rand(n_states);
    b0 = b0/sum(b0);

else
    disp('working on it');
end

%% Create reward and find policy
if RANDOM_REWARD
    reward_expert = rand(n_states,1);
    reward_expert = reward_expert/sum(reward_expert);
else
    reward_expert = zeros(n_states,1);
    reward_expert(n_states) = 1;
end

% Find optimal policy
if PARTIALLY_OBSERVABLE
    [value_expert,policy_expert] = value_iteration_pomdp(transition_probability,observation_probability,reward_expert,gamma);
else
    [value_expert,policy_expert] = value_iteration(transition_probability,reward_expert,gamma);
end

%% Solve IRL
[reward_learner,policy_learner,error] = maxmargin_irl(transition_probability,gamma,features,s0,reward_expert,policy_expert);

%% Plot
% Plot expert
plot_functions(reward_expert,policy_expert,'Expert',1);

% Plot learner
plot_functions(reward_learner,policy_learner,'Learner',2);

% Plot error
plot_error(error,3);

%% Functions
function plot_functions(reward,policy,name,figure_number)
    figure(figure_number); subplot(2,1,1); bar(reward); title([name ' Reward'])
    subplot(2,1,2); bar(policy); title([name ' Policy'])
end

function plot_error(error,figure_number)
    figure(figure_number); plot(1:length(error),error); title('Error');
end