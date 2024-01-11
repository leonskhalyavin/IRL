n_states = 10;
p = .2;
discount = .9;
mdp = mdp_lineworld(n_states,p,discount);
figure_plot(mdp)

function figure_plot(mdp)
    figure(1); 
    subplot(2,1,1); bar(1:mdp.n_states,mdp.reward); title('Reward');
    subplot(2,1,2); bar(1:mdp.n_states,mdp.policy); title('Policy');
end