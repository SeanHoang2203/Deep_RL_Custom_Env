import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.compat.v1.keras.backend import set_session

import matplotlib.pyplot as plt
from sim_env_dynamic_single import ppo_Car_env
from policy_net_single import Policy_net_single
from ppo_single import PPOTrain_single
from imagine_trajectory import imagine_trajectory as imagine
from replay_buffer import replay_buffer
from model_based import *

from util import *
import multiprocessing as mp
from threading import Thread, Barrier, active_count
from executor import *
from collections import defaultdict
import sys
import datetime as dt
import pickle as pk

def argparser():
    savedir_default = 'C:/Users/e0442113/trained_models(mb)/' \
                      + str(dt.datetime.now()).replace(':','_')\
                      + '/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default=savedir_default)
    parser.add_argument('--savedir', help='save directory', default=savedir_default)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda', default=0.95, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--clip', default=0.1, type=float)
    parser.add_argument('--explore_rate', default=0.01, type=float)
    parser.add_argument('--iteration', default=1000, type=int)
    parser.add_argument('--rewarddatadir', default = 'reward_record')
    parser.add_argument('--time_step', default = 0.005)
    parser.add_argument('--epoch', type=int, default = 30)
    parser.add_argument('--trajectory', type=int, default = 50) # number of partial trajectory each iteration
    parser.add_argument('--batch_size', type=float, default = 1.0) # default train on all collected data
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--ep_len', type=float, default=None)
    parser.add_argument('--imag_step', type=int, default=20)
    parser.add_argument('--imag_time', type=int, default=8)
    parser.add_argument('--polyak', type=float, default=0.995)
    
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--port', nargs='+', type=int, default = 25000) #port to connect to vrep
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast_sim', action='store_true')
    parser.add_argument('--continue', action='store_true')
    parser.add_argument('--current_iter', type=int, default=0)
    parser.add_argument('--model_path')

    params = vars(parser.parse_args())
##    if params['batch_size'] is not None:
##        params['batch_size'] = float(param['batch_size'])
    
    assert params['workers'] == len(params['port']), 'Number of workers and port must be the same' 
    return params
    

def main(args_dict):
    obs_size = 22
    act_size = 2
    session = create_tf_session(use_gpu=args_dict['use_gpu'])
    graph = tf.compat.v1.get_default_graph()
    writer = tf.compat.v1.summary.FileWriter(args_dict['logdir'],session.graph)

    policy = Policy_net_single('policy',session, units=args_dict['units'],\
                               n_layers=args_dict['layers'])
    old_policy = Policy_net_single('old_policy',session, \
                                   units=args_dict['units'], \
                                   n_layers=args_dict['layers'])
    trainer = PPOTrain_single(session, policy, old_policy, \
                              gamma=args_dict['gamma'],\
                              lr=args_dict['lr'],\
                              clip_value=args_dict['clip'],\
                              c_2 = args_dict['explore_rate'])
    saver = tf.compat.v1.train.Saver()
    buffer = replay_buffer(sample_rate = 2/5)
    dyna_model = dynamics_model(obs_size+act_size,obs_size,learn_rate=0.001)       #hard-coded
    dyna_model.model._make_predict_function()
    q_func = q_model(obs_size+act_size,1,learn_rate=0.01)                 #hard-coded
    q_func.model._make_predict_function()
    q_target = q_model(obs_size+act_size,1,learn_rate=0.01)
    q_target.model._make_predict_function()
    
    session.run(tf.compat.v1.global_variables_initializer())
    
    if args_dict['continue']:
        with open(args_dict['model_path']+'\\replay_buffer.pk','rb') as file:
            buffer = pk.load(file)
        saver.restore(session, args['model_path']+'\model_updated.ckpt')
        dyna_model.model.load_weights(args['model_path']+'\dyna_model.h5')
        q_func.model.load_weights(args['model_path']+'\q_func.h5')
        new_clip = args_dict['clip'] * (1 - args_dict['current_iter']/args_dict['iteration'])
        lr = args_dict['lr'] / (args_dict['current_iter'])
        trainer.set_params(clip_value=new_clip, lr=lr)
        
    workers = args_dict['workers']
    imag_steps = args_dict['imag_step']
    imag_time = args_dict['imag_time']
    time_step = args_dict['time_step']
    time_episode = args_dict['ep_len']
    server_conn = []
    client_conn = []
    processes = []
    for i in range(workers):
        conn_1, conn_2 = mp.Pipe()
        server_conn.append(conn_1)
        client_conn.append(conn_2)
        p = mp.Process(target=execute, args=(args_dict['port'][i],conn_2,args_dict))
        p.start()
        processes.append(p)
        
    imag_list = [imagine(session,graph,time_step,time_episode,dyna_model,policy,imag_steps)\
                 for _ in range(workers)]
    b = Barrier(workers+1)    
    for iteration in range(args_dict['iteration']):
        threads = []
        train_obs = defaultdict(list)     
        train_rewards = defaultdict(list)
        train_actions = defaultdict(list)
        train_obs_next = defaultdict(list)
        train_vpred = defaultdict(list)
        train_vpred_next = defaultdict(list)
        train_obs_info = defaultdict(list)
        q_vals = []
        gaes = []
        obs = []
        rewards = []
        actions = []
        obs_next = []
        vpred = []
        vpred_next = []
        obs_info = []

        print('Iteration: {}'.format(iteration))
##        for op in tf.compat.v1.get_default_graph().get_operations():
##                print(str(op.name))
        for num in range(max(args_dict['trajectory']//workers, 1)):
            if num == max(args_dict['trajectory']//workers, 1) - 1 and iteration == args_dict['iteration'] - 1:
                terminal = True
            else:
                terminal = False
            for i in range(workers):
                thread = Thread(target=handle, args=(server_conn[i],policy,old_policy,b,buffer,imag_list[i],\
                                                     imag_time,imag_steps, train_obs[i],train_rewards[i],\
                                                     train_actions[i],train_obs_next[i],train_vpred[i],\
                                                     train_vpred_next[i],train_obs_info[i],terminal))
                thread.start()
                threads.append(thread)
            
            b.wait()
            print('Last trajectory rewards: ')
##            for thread in threads:
##                thread.join()
            for i in range(workers):
                print(sum(train_rewards[i][:-args_dict['imag_step']*args_dict['imag_time']]))
                obs.extend(train_obs[i])            # train_obs[i]: list 2d numpy array
                rewards.extend(train_rewards[i])    # list of 1d numpy array
                actions.extend(train_actions[i])    # list of 2d numpy array
                obs_next.extend(train_obs_next[i])  # list of 2d numpy array
                vpred.extend(train_vpred[i])        # list of 1d numpy array 
                vpred_next.extend(train_vpred_next[i])
                obs_info.extend(train_obs_info[i])
                # predict q_value
                with graph.as_default():
                    set_session(session)
                    q_val = q_target.predict(np.hstack((np.array(train_obs[i]).squeeze(),\
                                                      np.array(train_actions[i]).squeeze())))
                    q_func_val = q_func.predict(np.hstack((np.array(train_obs[i]).squeeze(),\
                                                          np.array(train_actions[i]).squeeze())))
                q_val[-1,0] = train_rewards[i][-1]       # last q_value = last reward
                q_func_val[-1,0] = train_rewards[i][-1]       # last q_value = last reward
                q_vals.extend(q_func_val)
                q_val = np.vstack((q_val,0))        # next q_value = 0
                q_func_val = np.vstack((q_func_val,0))
                data = np.hstack((np.asarray(train_obs[i]).squeeze(),\
                                  np.asarray(train_actions[i]).squeeze(),\
                                  np.array(train_rewards[i])[:,np.newaxis],\
                                  np.array(train_obs_next[i]).squeeze(),\
                                  np.array(train_obs_info[i])[:,np.newaxis],\
                                  q_val[:-1],q_val[1:]))
                buffer.add_to_buffer(data)
                gaes.extend(get_gaes(args_dict['gamma'],\
                                       args_dict['lambda'],\
                                       train_rewards[i],\
                                       train_vpred[i],\
                                       train_vpred_next[i]))
##                                       q_func_val[1:].squeeze()))
                train_obs[i] = []     
                train_rewards[i] = []
                train_actions[i] = []
                train_obs_next[i] = []
                train_vpred[i] = []
                train_vpred_next[i] = []
                train_obs_info[i] = []
##            for conn in server_conn:
##                conn.send('Next trajectory')

        obs = np.asarray(obs).squeeze()
        rewards = np.asarray(rewards)
        actions = np.asarray(actions).squeeze()
        vpred = np.asarray(vpred).squeeze()
        vpred_next = np.asarray(vpred_next).squeeze()
        adv = np.asarray(gaes).squeeze()
        q_vals = np.asarray(q_vals).squeeze()
##        adv = q_vals - vpred
        # normalize advantage
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).squeeze()
        
        experiences = [obs, actions, adv, rewards, vpred_next]
        
        #train agent
        print('Training agent...')
        indices = np.random.permutation(len(rewards))
        batch_size = int(args_dict['batch_size']*len(rewards))
        for start in range(int(np.ceil(len(rewards)/batch_size))):
            ind = indices[(start*batch_size):((start+1)*batch_size)]
            for epoch in range(args_dict['epoch']):
                kl_div = trainer.train(obs=experiences[0][ind],
                                      actions=experiences[1][ind],
                                      adv=experiences[2][ind],
                                      rewards=experiences[3][ind],
                                      v_preds_next=experiences[4][ind])
                mean_div = kl_div.mean()
##                print(len(kl_div), mean_div)

                #early stopping
                if (mean_div > max(100 / ((iteration + 1)*2),0.01)):
                    break

        #train dynamic model and q_function
        samples = buffer.sample()
        dyna_model.train(samples[:,:(obs_size+act_size)],\
                         samples[:,(obs_size+act_size+1):-3],\
                         epochs=15)
##        a, _ = policy.act(obs = samples[:,(obs_size+act_size+1):-3])
##        with graph.as_default():
##            set_session(session)
##            q_targ = samples[:,-3].squeeze() * \
##                     q_target.predict(np.hstack((samples[:,(obs_size+act_size+1):-3].squeeze(),\
##                                                a.squeeze()))).squeeze()
##        q_func.train(samples[:,:(obs_size+act_size)],\
##                     samples[:,(obs_size+act_size)] + args_dict['gamma'] * q_targ,\
##                     epochs=5)

        print(experiences[0][indices].shape)
        summary = trainer.get_summary(obs=experiences[0][indices],
                                      actions=experiences[1][indices],
                                      adv=experiences[2][indices],
                                      rewards=experiences[3][indices],
                                      v_preds_next=experiences[4][indices])
        writer.add_summary(summary, iteration)
        saver.save(session, args_dict['savedir']+'model_updated.ckpt')
        dyna_model.model.save_weights(args_dict['savedir'] + 'dyna_model.h5')
        q_func.model.save_weights(args_dict['savedir'] + 'q_func.h5')

        #save replay buffer
        with open(args_dict['savedir']+'replay_buffer.pk','wb') as input_file:
            pk.dump(buffer,input_file)

        # Update parameters
##        print('before assign: ')
##        pi = old_policy.get_trainable_variables()
##        session.run(tf.print(pi,output_stream=sys.stdout))
        trainer.assign_policy_parameters()
##        print('after assign: ')
##        pi = old_policy.get_trainable_variables()
##        session.run(tf.print(pi,output_stream=sys.stdout))
        new_clip = args_dict['clip'] * (1 - (iteration + args_dict['current_iter'])/args_dict['iteration'])
        lr = args_dict['lr'] / (iteration + args_dict['current_iter'] + 1)
        trainer.set_params(clip_value=new_clip, lr=lr)
        update_q(q_func, q_target, args_dict['polyak'])
        
        
    writer.close()
            
    for p in processes:
        p.terminate()


if __name__ == '__main__':
    args = argparser()
    main(args)



                    
                
            
