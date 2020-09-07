from sim_env_dynamic_single import ppo_Car_env
from numpy import amax
from util import get_gaes
import numpy as np

def execute(port, conn, args_dict):

    fast = args_dict['fast_sim']
    env = ppo_Car_env(port=port, fast=fast, timestep=args_dict['time_step'])
    env.start_simulation()

    if args_dict['ep_len'] is None:
        max_ep_length = amax(env.floor) * 3 // args_dict['time_step']
        env.time_episode = max_ep_length * env.timestep
    else:
        env.time_episode = float(args_dict['ep_len'])

##    print('env time episode: ', env.time_episode)

    while True:
        # Get states/observations
        obs = env.get_states()
        
        # Query for actions 
        conn.send(obs)
        action = conn.recv()[0]
##        print(action)

        # Update environment
##        env.action_update([action[0]/2,-action[0]/2]) #turning
##        env.action_update([0,0])
##        env.action_update([action[1]*5,action[1]*5])  #moving
        env.action_update(action)

        # Get rewards and report back    
        reward, done = env.get_rewards()
##        print(reward)
        conn.send(str(reward) + ' ' + str(done))
        conn.recv()

        # Get next states for v_pred_next
        next_obs = env.get_states()  
        conn.send(next_obs)

        if done:
            conn.send('Done trajectory')
            env.reset_sim()
        else:
            conn.send('Next step')

        if conn.recv() == 'Terminate':
            env.stop_simulation()
            break


def handle(conn, policy, old_policy, barrier, buffer, imagine, imagine_time, steps, obs_list, rew_list, \
           action_list, obs_next_list, vpred_list, vpredNe_list, obs_info, terminal=False):
    while True:

        # receive states
        obs = conn.recv()
        obs_list.append(obs)

        # get action
        action, v_pred = policy.act(obs=obs, stochastic=True)
        conn.send(action)
        action_list.append(action)
        vpred_list.append(v_pred)

        # get reward and next obs
        reward = conn.recv().split()
        rew_list.append(float(reward[0]))
        conn.send(0)
        next_obs = conn.recv()
        obs_next_list.append(next_obs)
        if int(reward[-1]):
            vpred_next = 0
        else:
            _, vpred_next = old_policy.act(obs=next_obs,stochastic=True)
        vpredNe_list.append(vpred_next)
        obs_info.append(1 - int(reward[1]))

        # get trajectory status
        if conn.recv() == 'Done trajectory':
            # begin imagine trajectory
            for t in range(imagine_time):
                idx = np.random.permutation(len(obs_list))[0]
                o, a, r, o_n, v, vn, o_i = imagine.roll_out(obs_list[idx],idx,steps)
                obs_list.extend(o)
                action_list.extend(a)
                rew_list.extend(r)
                obs_next_list.extend(o_n)
                vpred_list.extend(v)
                vpredNe_list.extend(vn)
                obs_info.extend(o_i)
                
            if terminal:
                conn.send('Terminate')
            else:
                conn.send('Next')
            barrier.wait()
            return
        else:
            conn.send('Continue')
    

    
    
    
