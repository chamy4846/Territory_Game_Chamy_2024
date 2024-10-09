import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.flatten = layers.Flatten(input_shape=(state_size,))
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(action_size, activation='softmax')

    def call(self, state):
        x = self.flatten(state)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

class PPO:
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.policy_models = [self.build_policy_model() for _ in range(num_agents)]
        self.value_models = [self.build_value_model() for _ in range(num_agents)]
        self.policy_optimizers = [tf.keras.optimizers.Adam() for _ in range(num_agents)]
        self.value_optimizers = [tf.keras.optimizers.Adam() for _ in range(num_agents)]

    def build_policy_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model

    def build_value_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def get_action(self, state, agent_index):
        state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        probs = self.policy_models[agent_index](state)
        action = tf.random.categorical(probs, num_samples=1).numpy()[0, 0]
        return action, probs[0, action].numpy()

    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
            advantages[t] = last_gae_lam = delta + gamma * lam * last_gae_lam
        return advantages

    def train(self, states, actions, rewards, old_probs, advantages, agent_index):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        # Training the policy model
        with tf.GradientTape() as tape:
            new_probs = self.policy_models[agent_index](states, training=True)
            actions_one_hot = tf.one_hot(actions, self.action_size)
            selected_probs = tf.reduce_sum(new_probs * actions_one_hot, axis=1)
            ratio = selected_probs / old_probs
            clipped_ratio = tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        policy_grads = tape.gradient(policy_loss, self.policy_models[agent_index].trainable_variables)
        self.policy_optimizers[agent_index].apply_gradients(zip(policy_grads, self.policy_models[agent_index].trainable_variables))

        # Training the value model
        with tf.GradientTape() as tape:
            values = self.value_models[agent_index](states, training=True)
            value_loss = tf.reduce_mean(tf.square(rewards - values))
        
        value_grads = tape.gradient(value_loss, self.value_models[agent_index].trainable_variables)
        self.value_optimizers[agent_index].apply_gradients(zip(value_grads, self.value_models[agent_index].trainable_variables))

    


    def save(self, path):
        for i, model in enumerate(self.policy_models):
            save_path = f"{path}_policy_{i}.h5"
            print(f"Saving policy model {i} to {save_path}")  # 打印保存路径# Print save path
            model.save_weights(save_path)
            print(f"Policy model {i} saved successfully to {save_path}")
    
        for i, model in enumerate(self.value_models):
            save_path = f"{path}_value_{i}.h5"
            print(f"Saving value model {i} to {save_path}")  # 打印保存路径# Print save path
            model.save_weights(save_path)
            print(f"Value model {i} saved successfully to {save_path}")

    
    
    def load(self, path):
        for i, model in enumerate(self.policy_models):
            full_path = f"{path}_policy_{i}.h5"
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Model file {full_path} not found.")
            print(f"Loading model from: {full_path}")
            model.load_weights(full_path)
        for i, model in enumerate(self.value_models):
            full_path = f"{path}_value_{i}.h5"
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Model file {full_path} not found.")
            print(f"Loading model from: {full_path}")
            model.load_weights(full_path)











        