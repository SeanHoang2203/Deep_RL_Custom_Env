import argparse
import numpy as np
import tensorflow as tf
from sim_env_dynamic_single import ppo_Car_env
from policy_net_single import Policy_net_single
from ppo_single import PPOTrain_single
from util import *

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='test_log')
##    parser.add_argument('--savedir', help='save directory', default='trained_models'
    parser.add_argument('--port', nargs='+', type=int,default = 25000) #port to connect to vrep
    parser.add_argument('--time_step', default = 0.005)
    parser.add_argument('--model_path')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--test_iter',type=int, default=10)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--ep_len', type=float, default=1e5)

    params = vars(parser.parse_args())
    return params

def main(args):
    session = create_tf_session(use_gpu=args['use_gpu'])
    writer = tf.compat.v1.summary.FileWriter(args['logdir'],session.graph)
    
    policy = Policy_net_single('policy',session, units=args['units'],n_layers=args['layers'])
##    old_policy = Policy_net_single('old_policy',session, units=args['units'], n_layers=args['layers'])
##    trainer = PPOTrain_single(session, policy, old_policy, gamma=args['gamma'])
    saver = tf.compat.v1.train.Saver()

    saver.restore(session, args['model_path'])

    env = ppo_Car_env(port=args['port'][0], timestep=args['time_step'])
    env.time_episode = args['ep_len']      # a huge episode length
    env.start_simulation()

    for num in range(args['test_iter']):
        print('Test iteration: ', num)
        total_reward = test_env(policy, env)
        print('Reward last iteration: ',total_reward)
        print('...........................')

    env.stop_simulation()
    env.close_simulation()

if __name__ =='__main__':
    args = argparser()
    main(args)

