import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test'], default='test', type=str)  # mode = 'train' or 'test'
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--iteration', default=5, type=int)
    parser.add_argument('--model', choices=['TD3', 'SAC'],default='TD3')
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=200000, type=int)  # replay buffer size
    parser.add_argument('--num_iteration', default=100000, type=int)  # num of  games
    parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
    parser.add_argument('--seed', default=1, type=int)

    # optional parameters
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--sample_frequency', default=256, type=int)
    parser.add_argument('--activation', default='Relu', type=str)
    parser.add_argument('--render', default=False, type=bool)  # show UI or not
    parser.add_argument('--log_interval', default=50, type=int)  #
    parser.add_argument('--load', default=False, type=bool)  # load model
    parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
    parser.add_argument('--policy_noise', default=0.1, type=float)
    parser.add_argument('--noise_clip', default=0.1, type=float)
    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--exploration_noise', default=45000, type=float)
    parser.add_argument('--max_episode', default=2000, type=int)
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--init_temperature',default=0.1,type=float)
    parser.add_argument('--discount',default=0.99,type=float)
    parser.add_argument('--actor_update_freq',default=2,type=int)
    parser.add_argument('--critic_target_update_freq',default=1,type=int)
    parser.add_argument('--critic_tau',default=0.01,type=float)
    return parser.parse_args()
