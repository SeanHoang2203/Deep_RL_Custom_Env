import numpy as np
import tensorflow as tf
import os
import copy

def get_gaes(gamma, lamb, rewards, v_preds, v_preds_next):
##    deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    deltas = np.asarray(rewards).squeeze() + gamma * np.asarray(v_preds_next).squeeze() - np.asarray(v_preds).squeeze()
    if (len(rewards) == 1):
        deltas = np.array([deltas])
    # calculate generalized advantage estimate(lambda = 1), see ppo paper eq(11)
##    gaes = copy.deepcopy(deltas)
    gamma_array = np.asarray([(lamb*gamma)**i for i in range(len(v_preds))])
    gaes = [np.sum(deltas[t:] * gamma_array[:len(v_preds)-t]) \
            for t in range(len(v_preds))]
    return gaes

def create_tf_session(use_gpu, gpu_frac=0.6, allow_gpu_growth=True, which_gpu=0):
    if use_gpu:
        # gpu options
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_frac,
            allow_growth=allow_gpu_growth)
        # TF config
        config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            allow_soft_placement=True,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        # set env variable to specify which gpu to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        # TF config without gpu
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

    # use config to create TF session
    sess = tf.compat.v1.Session(config=config)
    return sess

def update_q(q_func, q_target, polyak):
    old_weights = q_target.model.get_weights()
    updated_weights = q_func.model.get_weights()
    new_weights = [old_weights[i] * polyak + updated_weights[i] * (1-polyak) \
                   for i in range(len(old_weights))]
    q_target.model.set_weights(new_weights)


def test_env(policy,env):
##    env.reset_sim()
    done = 0 
    total_reward = 0
    while not done: 
        obs = env.get_states()
##        action, _ = policy.act(obs,stochastic=True)
        action = policy.act_direct(obs)
##        print(action)
        action = action.squeeze()
        env.action_update(action) #turning
        reward, done = env.get_rewards()
        total_reward += reward
    env.reset_sim()
    return total_reward

##def pseudo_rewards(states):
##    """
##    States : n-by-m numpy array contains n states vector with m features
##    """
##
##    assert len(states.shape) == 2, 'Only 2-D numpy array accepted'
##    rewards = []
##    for state in  states:
        
