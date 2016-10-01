import gym
import numpy as np
import tensorflow as tf

batch_max = 500
max_episodes = 1000
max_steps = 400
epsilon = 1.0
epsilon_decay = 0.90
gamma = 0.99
memory_max = 5000
memory_num = 0
memory_fill = 0
train_num = 0

env = gym.make('CartPole-v0')
env.monitor.start('training_dir', force=True)

state_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.n

sess = tf.Session()

w1 = tf.Variable(tf.random_uniform([state_dimensions, 150], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([150], -1.0, 1.0))

w2 = tf.Variable(tf.random_uniform([150, 150], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([150], -1.0, 1.0))

w3 = tf.Variable(tf.random_uniform([150, 150], -1.0, 1.0))
b3 = tf.Variable(tf.random_uniform([150], -1.0, 1.0))

w4 = tf.Variable(tf.random_uniform([150, action_dimensions], -1.0, 1.0))
b4 = tf.Variable(tf.random_uniform([action_dimensions], -1.0, 1.0))

states = tf.placeholder(tf.float32, [None, state_dimensions])
actions = tf.placeholder(tf.float32, [None, action_dimensions])

hidden_1 = tf.nn.relu(tf.matmul(states, w1) + b1)
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)
action_values = tf.matmul(hidden_3, w4) + b4
Q = tf.reduce_sum(tf.mul(action_values, actions), reduction_indices=1)

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

rewards = tf.placeholder(tf.float32, [None, ])
states_ = tf.placeholder(tf.float32, [None, state_dimensions])

hidden_1_ = tf.nn.relu(tf.matmul(states_, w1_) + b1_)
hidden_2_ = tf.nn.relu(tf.matmul(hidden_1_, w2_) + b2_)
hidden_3_ = tf.nn.relu(tf.matmul(hidden_2_, w3_) + b3_)
action_values_ = tf.matmul(hidden_3_, w4_) + b4_
Q_ = rewards + gamma * tf.reduce_max(action_values_, reduction_indices=1)

loss = tf.reduce_mean(tf.square(Q_ - Q))
loss2 = tf.reduce_mean(tf.square(rewards - Q))
training = tf.train.AdamOptimizer(0.0001).minimize(loss)
training2 = tf.train.AdamOptimizer(0.0001).minimize(loss2)

mem_states = np.zeros((memory_max, state_dimensions));
mem_states_ = np.zeros((memory_max, state_dimensions));
mem_actions = np.zeros((memory_max, action_dimensions));
mem_rewards = np.zeros((memory_max));

sess.run(tf.initialize_all_variables())

for episode in xrange(max_episodes):  
    done = False
    reward = 0.0;
    state = env.reset() 
    next_state = state; 
    
    for step in xrange(max_steps):

        if epsilon >= np.random.rand():
            action = env.action_space.sample()
        else:
            action = sess.run(action_values, feed_dict={states: np.array([state])})
            action = np.argmax(action)

        epsilon = epsilon * epsilon_decay

        next_state, reward, done, _ = env.step(action)    

        if done and step + 1 < max_steps:
            reward = -500
           
        mem_states[memory_num] = np.array(state)      
        mem_states_[memory_num] = np.array(next_state)
        mem_rewards[memory_num] = reward
        mem_actions[memory_num] = np.zeros(action_dimensions)
        mem_actions[memory_num][action] = 1.0

        if(memory_num + 1 < memory_max):
            memory_num += 1
            if(memory_fill < memory_max):
                memory_fill += 1
        else:       
            memory_num = 0

        if memory_fill > 0:
            batch = batch_max if memory_fill > batch_max else memory_fill
            size = memory_fill if memory_fill < memory_max else memory_max
            i = np.random.choice(size, batch, replace=True)
            feed = {
                states: mem_states[i],
                states_: mem_states_[i],
                rewards: mem_rewards[i],
                actions: mem_actions[i]
            }
            
            if done and step + 1 < max_steps:
                loss_value, _ = sess.run([loss2, training2], feed_dict=feed)
            else:
                loss_value, _ = sess.run([loss, training], feed_dict=feed) 
            
            if train_num >= 100:
                sess.run(update_w1_)
                sess.run(update_b1_)
                sess.run(update_w2_)
                sess.run(update_b2_)
                sess.run(update_w3_)
                sess.run(update_b3_)
                sess.run(update_w4_)
                sess.run(update_b4_)
                train_num = 0
            else:
                train_num += 1        

        state = next_state    

        if done or step + 1 == max_steps:
            print "[Episode - {}, Steps - {}]".format(episode, step) 
            break

env.monitor.close() 
sess.close()      
