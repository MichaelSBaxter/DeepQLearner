import gym
import numpy as np
import tensorflow as tf

batch_size = 100
episodes = 1
steps = 100
epsilon = 0.3
gamma = 0.99
memory_max = 1000
memory_num = 0

env = gym.make('CartPole-v0')

space = env.observation_space
state_dimensions = space.shape[0]
action_dimensions = env.action_space.n

sess = tf.InteractiveSession()

w1 = tf.Variable(tf.random_uniform([state_dimensions, action_dimensions], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([action_dimensions], -1.0, 1.0))

previous_state = tf.placeholder(tf.float32, [None, state_dimensions])
previous_action_values = tf.squeeze(tf.nn.relu(tf.matmul(previous_state, w1) + b1))
previous_action_masks = tf.placeholder(tf.float, [None, action_dimensions])
previous_values = tf.reduce_sum(tf.mul(previous_action_values, previous_action_masks), reduction_indices=1)

previous_rewards = tf.placeholder(tf.float32, [None, ])
next_state = tf.placeholder(tf.float32, [None, state_dimensions])
next_action_values = tf.squeeze(tf.matmul(next_state, w1) + b1)
next_value = previous_rewards + gamma * tf.reduce_max(next_action_values, reduction_indices=1)

loss = tf.reduce_mean(tf.square(previous_value - next_values))
training = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Initialize replay memory D
memory_previous_states = np.zeros(memory_max, state_dimensions)
memory_next_states = np.zeros(memory_max, state_dimensions)
memory_rewards = np.zeros(memory_max)
memory_action_masks = np.zeros(memory_max, state_dimensions)

# Initialize action-value function Q
sess.run(tf.initialize_all_variables())

for episode in xrange(episodes):
    # Initialize sequence
    done = False
    reward = 0.0;
    observation = env.reset()    
    
    for step in xrange(steps):
        if done:
            break;
        else
            env.render()

        if epsilon >= np.random.rand():
            # With probability epsilon select a random action
            action = env.action_space.sample()
        else:
            # Otherwise select action = argmax(Q)
            states = np.array([observation])
            action_values = session.previous_action_values.eval(feed_dict={previous_sate: states})
            action = np.argmax(action_values)    
      
        # Store transition in D
        memory_num += 1;        
        memory_previous_states[memory_num] = np.array(previous_state)
        memory_next_states[memory_num] = np.array(next_state)
        memory_rewards[memory_num] = reward
        memory_action_masks[memory_num] = np.zeros(action_dimensions)
        memory_action_masks[memory_num][previous_action] = 1.0        

        # sample random minibatch of transitions from D
        imb = np.random.choice(memory_max, batch_size, replace=True)
        minibatch = { 
            previous_states: memory_previous_states[imb],
            previous_action_masks: memory_action_masks[imb],
            previous_rewards: memory_experinces_rewards[imb],
            next_states: memory_next_states[imb]
        }   

        # perform gradient descent step
        loss, _ = sess.run([loss, training], feed_dict=minibatch)


        observation, reward, done, info = env.step(action)

