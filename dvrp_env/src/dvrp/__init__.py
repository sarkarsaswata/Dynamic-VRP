from gymnasium.envs.registration import register
register(
    id='DVRP-v0', 
    entry_point='dvrp.envs:DVRPEnv', 
    nondeterministic=True # This is a flag that tells the environment is stochastic
         )
