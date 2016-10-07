import gym
import random
import numpy as np
import tensorflow as tf

nodes1 = 200
nodes2 = 200
nodes3 = 200
kernel1_1 = 5
kernel1_2 = 5
kernel1_4 = 64
kernel2_1 = 5
kernel2_2 = 5
kernel2_4 = 64
dropout1 = .5
dropout2 = .5
dropout3 = .5
batch_max = 30
max_episodes = 10000
max_steps = 500
epsilon = 1.0
epsilon_decay = 0.99991
epsilon_min = 0.01
gamma = (max_steps - 1) / float(max_steps)
learning_rate = 0.001
learning_tick = 10
q_copy_tick = 1000
memory_max = 100000
use_cnn = True
use_dropout = False
show_render = True

if __name__ == '__main__':

    ''' 
    Compatible environments:
        'CartPole-v0'
        'Breakout-v0'      
    '''
    env = gym.make('Breakout-v0')
    use_cnn = True
    #env.monitor.start('training_dir2', force=True)

    action_dimensions = env.action_space.n

    sess = tf.Session()

    if use_cnn:
        state_dimensions_1 = env.observation_space.shape[0]
        state_dimensions_2 = env.observation_space.shape[1]
        state_dimensions_3 = env.observation_space.shape[2]
        kernel1_3 = state_dimensions_3
        kernel2_3 = kernel1_4

        states = tf.placeholder(tf.float32, [batch_max, state_dimensions_1, state_dimensions_2, state_dimensions_3])
        states_ = tf.placeholder(tf.float32, [batch_max, state_dimensions_1, state_dimensions_2, state_dimensions_3])
        
        kernel1 = tf.Variable(tf.random_uniform([kernel1_1, kernel1_2, kernel1_3, kernel1_4]))
        b1 = tf.Variable(tf.random_uniform([kernel1_4], -0.1, 0.1))

        kernel2 = tf.Variable(tf.random_uniform([kernel2_1, kernel2_2, kernel2_3, kernel2_4]))
        b2 = tf.Variable(tf.random_uniform([kernel2_4], -0.1, 0.1))

        conv1 = tf.nn.conv2d(states, kernel1, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

        conv2 = tf.nn.conv2d(conv1, kernel2, [1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

        conv2_shape = conv2.get_shape().as_list()
        reshape = tf.reshape(conv2, [batch_max, -1])
        dim1 = reshape.get_shape()[1].value
   
        w3 = tf.Variable(tf.random_uniform([dim1, nodes3], -0.1, 0.1))
        b3 = tf.Variable(tf.random_uniform([nodes3], -0.1, 0.1))

        w4 = tf.Variable(tf.random_uniform([nodes3, action_dimensions], -0.1, 0.1))
        b4 = tf.Variable(tf.random_uniform([action_dimensions], -0.1, 0.1))
        
        hidden_3 = tf.nn.relu(tf.matmul(reshape, w3) + b3) 

        if use_dropout:
            hidden_3 = tf.nn.dropout(hidden_3, dropout3)
       
        kernel1_ = tf.Variable(kernel1.initialized_value())
        b1_ = tf.Variable(b1.initialized_value())

        kernel2_ = tf.Variable(kernel2.initialized_value())
        b2_ = tf.Variable(b2.initialized_value())

        w3_ = tf.Variable(w3.initialized_value())
        b3_ = tf.Variable(b3.initialized_value())  

        w4_ = tf.Variable(w4.initialized_value())
        b4_ = tf.Variable(b4.initialized_value())  

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

        conv1_ = tf.nn.conv2d(states_, kernel1_, [1, 1, 1, 1], padding='SAME')
        conv1_ = tf.nn.relu(tf.nn.bias_add(conv1_, b1_))

        conv2_ = tf.nn.conv2d(conv1_, kernel2_, [1, 1, 1, 1], padding='SAME')
        conv2_ = tf.nn.relu(tf.nn.bias_add(conv2_, b2_))

        reshape_ = tf.reshape(conv2_, [batch_max, -1])

        hidden_3_ = tf.nn.relu(tf.matmul(reshape_, w3_) + b3_) 

        Q = tf.matmul(hidden_3, w4) + b4
        Q_ = tf.stop_gradient(tf.matmul(hidden_3_, w4_) + b4_)

    else:
        state_dimensions = env.observation_space.shape[0]

        states = tf.placeholder(tf.float32, [None, state_dimensions])
        states_ = tf.placeholder(tf.float32, [None, state_dimensions])

        w1 = tf.Variable(tf.random_uniform([state_dimensions, nodes1], -0.1, 0.1))
        b1 = tf.Variable(tf.random_uniform([nodes1], -0.1, 0.1))

        w2 = tf.Variable(tf.random_uniform([nodes1, nodes2], -0.1, 0.1))
        b2 = tf.Variable(tf.random_uniform([nodes2], -0.1, 0.1))

        w3 = tf.Variable(tf.random_uniform([nodes2, nodes3], -0.1, 0.1))
        b3 = tf.Variable(tf.random_uniform([nodes3], -0.1, 0.1))

        w4 = tf.Variable(tf.random_uniform([nodes3, action_dimensions], -0.1, 0.1))
        b4 = tf.Variable(tf.random_uniform([action_dimensions], -0.1, 0.1))

        hidden_1 = tf.nn.relu(tf.matmul(states, w1) + b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)

        if use_dropout:
            hidden_1 = tf.nn.dropout(hidden_1, dropout1)
            hidden_2 = tf.nn.dropout(hidden_2, dropout2)
            hidden_3 = tf.nn.dropout(hidden_3, dropout3)

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

        hidden_1_ = tf.nn.relu(tf.matmul(states_, w1_) + b1_)
        hidden_2_ = tf.nn.relu(tf.matmul(hidden_1_, w2_) + b2_)
        hidden_3_ = tf.nn.relu(tf.matmul(hidden_2_, w3_) + b3_)

        Q = tf.matmul(hidden_3, w4) + b4
        Q_ = tf.stop_gradient(tf.matmul(hidden_3_, w4_) + b4_)

    actions_filter = tf.placeholder(tf.int32, [None], name="actions_filter_oh")
    actions_filter_oh = tf.one_hot(actions_filter, action_dimensions)

    rewards = tf.placeholder(tf.float32, [None, ])
    not_terminal = tf.placeholder(tf.float32, [None])
 
    Q_filtered = tf.reduce_sum(tf.mul(Q, actions_filter_oh), reduction_indices=1)
    y_Q_ = rewards + gamma * tf.reduce_max(Q_, reduction_indices=1) * not_terminal

    loss = tf.reduce_mean(tf.square(y_Q_ - Q_filtered))
    training = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    D = []

    sess.run(tf.initialize_all_variables())

    tick_count = 0

    if use_cnn:
        empty_batch = np.zeros((batch_max, state_dimensions_1, state_dimensions_2, state_dimensions_3)) 

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
                if use_cnn:
                    empty_batch[0] = state
                    feed = {states: empty_batch}
                else:
                    feed = {states: np.array([state])}

                q = sess.run(Q, feed_dict=feed)
                action = np.argmax(q)

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            next_state, reward, done, _ = env.step(action)

            terminal = True if done and step + 1 < max_steps else False
            
            episode_reward_sum += reward               

            D.append([state, action, reward, next_state, terminal]) 

            if len(D) > memory_max:
                D.pop(0)

            if len(D) >= batch_max:
                batch = random.sample(D, batch_max)
                batch_states = []
                batch_actions = []
                batch_next_states = []
                batch_rewards = []
                batch_not_terminal = []
            
                for idx, batch_item in enumerate(batch):
                    mem_state, mem_action, mem_reward, mem_next_state, mem_terminal = batch_item  
                 
                    batch_states.append(mem_state)
                    batch_actions.append(mem_action)
                    batch_next_states.append(mem_next_state)
                    batch_rewards.append(mem_reward)
                    batch_not_terminal.append(0 if mem_terminal else 1)

                feed = {
                    states: batch_states,
                    actions_filter: batch_actions,
                    rewards: batch_rewards,
                    states_: batch_next_states,
                    not_terminal: batch_not_terminal
                }

                if tick_count % learning_tick == 0:
                    loss_value, _ = sess.run([loss, training], feed_dict=feed)           
                    
                if tick_count % q_copy_tick == 0:
                    sess.run(update_q_)   

            state = next_state 

            if show_render and episode % 10 == 0:
                env.render() 

            if done or step + 1 == max_steps:
                reward_total = np.sum(rewards)
                print "[Episode - {}, Steps - {}, Rewards - {}]".format(episode, step, episode_reward_sum) 
                break

    #env.monitor.close()
    sess.close()      
