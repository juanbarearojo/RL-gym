import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm

def train_q_learning(
    env,
    n_training_episodes=25000,
    max_steps=99,
    learning_rate=0.7,
    gamma=0.95,
    max_epsilon=1.0,
    min_epsilon=0.05,
    decay_rate=0.005
):
    """
    Entrena un agente con Q-Learning para un entorno de Gymnasium.

    :param env: Entorno de Gymnasium (p. ej., "Taxi-v3").
    :param n_training_episodes: Número de episodios de entrenamiento.
    :param max_steps: Máximo de pasos por episodio.
    :param learning_rate: Tasa de aprendizaje (alpha).
    :param gamma: Factor de descuento.
    :param max_epsilon: Valor inicial de epsilon para la política epsilon-greedy.
    :param min_epsilon: Valor mínimo de epsilon.
    :param decay_rate: Tasa de decaimiento exponencial de epsilon.
    :return: La Q-table entrenada (numpy array).
    """

    # Espacios de estados y acciones
    state_space = env.observation_space.n
    action_space = env.action_space.n

    # Inicializamos la Q-table en ceros
    Q_table = np.zeros((state_space, action_space))

    for episode in tqdm(range(n_training_episodes), desc="Entrenando"):
        # Calculamos epsilon dinámicamente en cada episodio (exponential decay)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Reiniciamos el entorno
        state, info = env.reset()
        done = False

        for step in range(max_steps):
            # Política epsilon-greedy
            if random.uniform(0, 1) > epsilon:
                # Explotación
                action = np.argmax(Q_table[state, :])
            else:
                # Exploración
                action = env.action_space.sample()

            # Ejecutamos la acción elegida
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Actualización Q-Learning:
            # Q(s, a) ← Q(s, a) + α [ r + γ * max_a' Q(s', a') - Q(s, a) ]
            Q_table[state, action] += learning_rate * (
                reward + gamma * np.max(Q_table[new_state, :]) - Q_table[state, action]
            )

            state = new_state

            if done:
                break

    return Q_table


def evaluate_q_table(env, Q_table, n_eval_episodes=100, max_steps=99):
    """
    Evalúa la Q-table durante n_eval_episodes episodios
    y devuelve la recompensa promedio y su desviación típica.

    :param env: Entorno de Gymnasium
    :param Q_table: Q-table ya entrenada
    :param n_eval_episodes: Número de episodios de evaluación
    :param max_steps: Máximo de pasos por episodio
    :return: (media_de_recompensas, std_de_recompensas)
    """
    rewards = []

    for _ in range(n_eval_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        for _ in range(max_steps):
            # Acción basada en la política greedy (explotación pura)
            action = np.argmax(Q_table[state, :])
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = new_state
            if done:
                break

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    # ------------------------------------------------
    # 1. Configuración del entorno y parámetros
    # ------------------------------------------------
    env_id = "Taxi-v3"
    env = gym.make(env_id)  # Por defecto, sin render para entrenamiento

    # Hiperparámetros de Q-Learning
    n_training_episodes = 200000
    learning_rate = 0.7
    gamma = 0.95
    max_steps = 99
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.005

    # Número de episodios para evaluación
    n_eval_episodes = 100

    # ------------------------------------------------
    # 2. Entrenamiento
    # ------------------------------------------------
    print(f"Entrenando agente Q-Learning en {env_id}...")
    Q_table = train_q_learning(
        env=env,
        n_training_episodes=n_training_episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        max_epsilon=max_epsilon,
        min_epsilon=min_epsilon,
        decay_rate=decay_rate
    )

    # ------------------------------------------------
    # 3. Evaluación
    # ------------------------------------------------
    mean_reward, std_reward = evaluate_q_table(env, Q_table, n_eval_episodes, max_steps)
    print(f"Recompensa promedio tras evaluación: {mean_reward:.2f} +/- {std_reward:.2f}")

    # ------------------------------------------------
    # 4. Prueba visual (opcional)
    # ------------------------------------------------
    # Podemos hacer un render del entorno para ver cómo actúa el agente
    # con la política final (greedy)
    test_env = gym.make(env_id, render_mode="human")
    episodes_to_test = 5

    print("\n--- Prueba visual de la política entrenada ---\n")
    for ep in range(episodes_to_test):
        obs, info = test_env.reset()
        done = False
        total_rewards = 0

        while not done:
            action = np.argmax(Q_table[obs, :])
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_rewards += reward
            done = terminated or truncated

        print(f"Episodio {ep+1}/{episodes_to_test} - Recompensa total: {total_rewards:.2f}")

    test_env.close()
    print("Prueba finalizada.")


if __name__ == "__main__":
    main()
