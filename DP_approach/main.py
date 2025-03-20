from YahtzeeEnvDP import YahtzeeEnvDP

YahtzeeEnvDP.initialize_class()
env = YahtzeeEnvDP()
state_space = env.ALL_STATE
print(state_space.shape)
env.make_transition_prob_table(state_space[:,])