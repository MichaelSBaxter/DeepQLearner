import gym
import random
import numpy as np
import tensorflow as tf

''' 
Compatible environments:
    'CartPole-v0'
    'Breakout-v0'  
    'FrozenLake-v0'    
'''
environment_name = 'FrozenLake-v0'
batch_size = 50
max_episodes = 5000
epsilon = 1.0
epsilon_decay = 0.9999
epsilon_min = 0.001
gamma = .95
learning_rate = 0.001
learning_tick = 1
q_copy_tick = 2000
memory_max = 100000

empty_batch = None

def create_network(environment):
    if environment.spec.id == 'Breakout-v0':        
        state_dim1 = env.observation_space.shape[0]
        state_dim2 = env.observation_space.shape[1]
        state_dim3 = env.observation_space.shape[2]
        action_dim = env.action_space.n 

        input_states = tf.placeholder(tf.float32, [batch_size, state_dim1, state_dim2, state_dim3])
        input_next_states = tf.placeholder(tf.float32, [batch_size, state_dim1, state_dim2, state_dim3])
        input_rewards = tf.placeholder(tf.float32, [None, ])
        input_not_terminal = tf.placeholder(tf.float32, [None, ])
        input_actions_filter = tf.placeholder(tf.int32, [None], name="actions_filter_oh")

        actions_filter_oh = tf.one_hot(input_actions_filter, action_dim)
  
        network_inputs = [input_states, input_next_states, input_rewards, input_not_terminal, input_actions_filter]  
      
        kernel1 = tf.Variable(tf.random_uniform([4, 4, state_dim3, 24]))
        b1 = tf.Variable(tf.random_uniform([24], -0.1, 0.1))

        kernel2 = tf.Variable(tf.random_uniform([4, 4, 24, 24]))
        b2 = tf.Variable(tf.random_uniform([24], -0.1, 0.1))

        conv1 = tf.nn.conv2d(input_states, kernel1, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

        conv2 = tf.nn.conv2d(conv1, kernel2, [1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

        conv2_shape = conv2.get_shape().as_list()
        reshape = tf.reshape(conv2, [batch_size, -1])
        dim1 = reshape.get_shape()[1].value

        w3 = tf.Variable(tf.random_uniform([dim1, 128], -0.1, 0.1))
        b3 = tf.Variable(tf.random_uniform([128], -0.1, 0.1))

        w4 = tf.Variable(tf.random_uniform([128, action_dim], -0.1, 0.1))
        b4 = tf.Variable(tf.random_uniform([action_dim], -0.1, 0.1))

        hidden_3 = tf.nn.relu(tf.matmul(reshape, w3) + b3)

        kernel1_ = tf.Variable(kernel1.initialized_value())
        b1_ = tf.Variable(b1.initialized_value())

        kernel2_ = tf.Variable(kernel2.initialized_value())
        b2_ = tf.Variable(b2.initialized_value())

        conv1_ = tf.nn.conv2d(input_next_states, kernel1_, [1, 1, 1, 1], padding='SAME')
        conv1_ = tf.nn.relu(tf.nn.bias_add(conv1_, b1_))

        conv2_ = tf.nn.conv2d(conv1_, kernel2_, [1, 1, 1, 1], padding='SAME')
        conv2_ = tf.nn.relu(tf.nn.bias_add(conv2_, b2_))

        reshape_ = tf.reshape(conv2_, [batch_size, -1])

        w3_ = tf.Variable(w3.initialized_value())
        b3_ = tf.Variable(b3.initialized_value())  

        w4_ = tf.Variable(w4.initialized_value())
        b4_ = tf.Variable(b4.initialized_value())  

        hidden_3_ = tf.nn.relu(tf.matmul(reshape_, w3_) + b3_) 

        update_kernel1_ = kernel1_.assign(kernel1)
        update_b1_ = b1_.assign(b1)

        update_kernel2_ = kernel2_.assign(kernel2)
        update_b2_ = b2_.assign(b2)

        update_w3_ = w3_.assign(w3)
        update_b3_ = b3_.assign(b3)

        update_w4_ = w4_.assign(w4)
        update_b4_ = b4_.assign(b4)

        update_q_ = [
            update_kernel1_, 
            update_kernel2_, 
            update_w3_,
            update_w4_, 
            update_b1_, 
            update_b2_, 
            update_b3_, 
            update_b4_]  

        Q = tf.matmul(hidden_3, w4) + b4
        Q_ = tf.stop_gradient(tf.matmul(hidden_3_, w4_) + b4_)

        Q_filtered = tf.reduce_sum(tf.mul(Q, actions_filter_oh), reduction_indices=1)
        y_Q_ = input_rewards + gamma * tf.mul(tf.reduce_max(Q_, reduction_indices=1), input_not_terminal)

        loss = tf.reduce_mean(tf.square(y_Q_ - Q_filtered))
        training = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return (Q, training, update_q_, network_inputs)
    
    if environment.spec.id == 'CartPole-v0' or environment.spec.id == 'FrozenLake-v0': 
        if isinstance(environment.observation_space, gym.spaces.box.Box):
            state_dim = environment.observation_space.shape[0]
    
        else:
            state_dim = environment.observation_space.n

        action_dim = env.action_space.n 

        input_states = tf.placeholder(tf.float32, [None, state_dim])
        input_next_states = tf.placeholder(tf.float32, [None, state_dim])
        input_rewards = tf.placeholder(tf.float32, [None, ])
        input_not_terminal = tf.placeholder(tf.float32, [None, ])
        input_actions_filter = tf.placeholder(tf.int32, [None], name="actions_filter_oh")
        actions_filter_oh = tf.one_hot(input_actions_filter, action_dim)

        network_inputs = [input_states, input_next_states, input_rewards, input_not_terminal, input_actions_filter]  

        w1 = tf.Variable(tf.random_uniform([state_dim, 48], -0.1, 0.1))
        b1 = tf.Variable(tf.random_uniform([48], -0.1, 0.1))

        w2 = tf.Variable(tf.random_uniform([48, 48], -0.1, 0.1))
        b2 = tf.Variable(tf.random_uniform([48], -0.1, 0.1))

        w3 = tf.Variable(tf.random_uniform([48, 48], -0.1, 0.1))
        b3 = tf.Variable(tf.random_uniform([48], -0.1, 0.1))

        w4 = tf.Variable(tf.random_uniform([48, action_dim], -0.1, 0.1))
        b4 = tf.Variable(tf.random_uniform([action_dim], -0.1, 0.1))

        w1_ = tf.Variable(w1.initialized_value())
        b1_ = tf.Variable(b1.initialized_value())

        w2_ = tf.Variable(w2.initialized_value())
        b2_ = tf.Variable(b2.initialized_value())

        w3_ = tf.Variable(w3.initialized_value())
        b3_ = tf.Variable(b3.initialized_value())

        w4_ = tf.Variable(w4.initialized_value())
        b4_ = tf.Variable(b4.initialized_value())

        update_w1_ = w1_.assign(w1)
        update_b1_ = b1_.assign(b1)

        update_w2_ = w2_.assign(w2)
        update_b2_ = b2_.assign(b2)

        update_w3_ = w3_.assign(w3)
        update_b3_ = b3_.assign(b3)

        update_w4_ = w4_.assign(w4)
        update_b4_ = b4_.assign(b4)

        update_q_ = [
            update_w1_, 
            update_w2_, 
            update_w3_,
            update_w4_, 
            update_b1_, 
            update_b2_, 
            update_b3_, 
            update_b4_]

        hidden_1 = tf.nn.relu(tf.matmul(input_states, w1) + b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3) 

        hidden_1_ = tf.nn.relu(tf.matmul(input_next_states, w1_) + b1_)
        hidden_2_ = tf.nn.relu(tf.matmul(hidden_1_, w2_) + b2_)
        hidden_3_ = tf.nn.relu(tf.matmul(hidden_2_, w3_) + b3_)

        Q = tf.matmul(hidden_3, w4) + b4
        Q_ = tf.stop_gradient(tf.matmul(hidden_3_, w4_) + b4_)
     
        Q_filtered = tf.reduce_sum(tf.mul(Q, actions_filter_oh), reduction_indices=1)

        y_Q_ = input_rewards + gamma * tf.mul(tf.reduce_max(Q_, reduction_indices=1), input_not_terminal)

        loss = tf.reduce_mean(tf.square(y_Q_ - Q_filtered))
        training = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return (Q, training, update_q_, network_inputs)
    return None


def get_action(environment, Q, state, network_inputs):
    if environment.spec.id == 'Breakout-v0': 
        if empty_batch == None:
            dim1 = environment.observation_space.shape[0]
            dim2 = environment.observation_space.shape[1]
            dim3 = environment.observation_space.shape[2]
            empty_batch = np.zeros((batch_size, dim1, dim2, dim3))

        empty_batch[0] = state
        q = sess.run(Q, feed_dict={network_inputs[0]: empty_batch})
        return np.argmax(q[0]) 

    if environment.spec.id == 'CartPole-v0' or environment.spec.id == 'FrozenLake-v0':  
        if isinstance(environment.observation_space, gym.spaces.box.Box):
            q = sess.run(Q, feed_dict={network_inputs[0]: np.array([state])})

        else:
            empty_batch = np.zeros((environment.observation_space.n))
            empty_batch[state] = 1.0
            q = sess.run(Q, feed_dict={network_inputs[0]: np.array([empty_batch])})

        return np.argmax(q)
    return None

def get_batch_feed(environment, batch, network_inputs):
    batch_states = []
    batch_actions = []
    batch_next_states = []
    batch_rewards = []
    batch_not_terminal = []

    if environment.spec.id == 'FrozenLake-v0':
        for idx, batch_item in enumerate(batch):
            mem_state, mem_action, mem_reward, mem_next_state, mem_terminal = batch_item 

            mem_state_array = np.zeros((environment.observation_space.n))
            mem_next_state_array = np.zeros((environment.observation_space.n))

            mem_state_array[mem_state] = 1.0
            mem_next_state_array[mem_next_state] = 1.0
         
            batch_states.append(mem_state_array)
            batch_actions.append(mem_action)
            batch_next_states.append(mem_next_state_array)
            batch_rewards.append(mem_reward)
            batch_not_terminal.append(0 if mem_terminal else 1)

        feed = {
            network_inputs[0]: batch_states,
            network_inputs[1]: batch_next_states,
            network_inputs[2]: batch_rewards,
            network_inputs[3]: batch_not_terminal,
            network_inputs[4]: batch_actions
        }

        return feed 

    else:     
        for idx, batch_item in enumerate(batch):
            mem_state, mem_action, mem_reward, mem_next_state, mem_terminal = batch_item  
         
            batch_states.append(mem_state)
            batch_actions.append(mem_action)
            batch_next_states.append(mem_next_state)
            batch_rewards.append(mem_reward)
            batch_not_terminal.append(0 if mem_terminal else 1)

        feed = {
            network_inputs[0]: batch_states,
            network_inputs[1]: batch_next_states,
            network_inputs[2]: batch_rewards,
            network_inputs[3]: batch_not_terminal,
            network_inputs[4]: batch_actions
        }

        return feed 
    return None

if __name__ == '__main__':
    tick_count = 0  

    env = gym.make(environment_name)
    max_steps = env.spec.timestep_limit

    sess = tf.Session()
    Q, training, update_q_, network_inputs = create_network(env)
    sess.run(tf.initialize_all_variables())         

    D = []

    env.monitor.start('training_results_{}'.format(env.spec.id), force=True)

    for episode in xrange(max_episodes):  
        done = False
        reward = 0.0;
        state = env.reset() 
        next_state = state;
        episode_reward_sum = 0

        for step in xrange(max_steps):
            tick_count += 1

            if epsilon >= np.random.rand():
                action = env.action_space.sample()
            else:
                action = get_action(env, Q, state, network_inputs)

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            next_state, reward, done, _ = env.step(action)

            terminal = True if done and step + 1 < max_steps else False
            
            episode_reward_sum += reward               

            D.append([state, action, reward, next_state, terminal]) 

            if len(D) > memory_max:
                D.pop(0)

            if len(D) >= batch_size:
                batch = random.sample(D, batch_size)
                feed = get_batch_feed(env, batch, network_inputs) 

                if tick_count % learning_tick == 0:
                    sess.run([training], feed_dict=feed)           
                    
                if tick_count % q_copy_tick == 0:
                    sess.run(update_q_)  

            state = next_state 

            #if episode % 10 == 0:
            #    env.render() 

            if done or step + 1 == max_steps:
                print "[Episode - {}, Steps - {}, Rewards - {}]".format(episode, step, episode_reward_sum) 
                break

    env.monitor.close()
    sess.close()    
