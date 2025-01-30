"""
Script Q-Learning para Taxi-v3 con hyperparámetros, seeds de evaluación fijos
(y subida a la Hugging Face Hub). 

¡NO modifiques el array `eval_seed`! Esto asegura la misma evaluación
entre todos los compañeros de clase.
"""

import os
import gymnasium as gym
import numpy as np
import random
import pickle
import json
import datetime

from tqdm import tqdm
from huggingface_hub import HfApi, Repository

# --------------------------------------------------------------------------------
# Hiperparámetros y semilla para evaluación (NO MODIFICAR eval_seed)
# --------------------------------------------------------------------------------

# Parám. de entrenamiento
n_training_episodes = 25000  # Total training episodes
learning_rate = 0.7          # Learning rate

# Parám. de evaluación
n_eval_episodes = 100        # Número de episodios de evaluación

# NO MODIFICAR eval_seed:
eval_seed = [
    16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50,
    172, 100, 148, 146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7,
    54, 55, 161, 131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54,
    82, 45, 95, 89, 59, 95, 124, 9, 113, 58, 85, 51, 134, 121, 169, 105, 21, 30,
    11, 50, 65, 12, 43, 82, 145, 152, 97, 106, 55, 31, 85, 38, 112, 102, 168,
    123, 97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28, 148
]

# Parámetros del entorno
env_id = "Taxi-v3"     # Nombre del entorno
max_steps = 99         # Pasos máximos por episodio
gamma = 0.95           # Factor de descuento

# Parámetros de exploración
max_epsilon = 1.0      # Exploración inicial
min_epsilon = 0.05     # Exploración mínima
decay_rate = 0.005     # Tasa de decaimiento de epsilon


def train_q_learning(env, n_episodes, max_steps, lr, gamma_,
                     max_eps, min_eps, decay):
    """
    Entrena un agente con Q-Learning para un entorno tipo Taxi-v3 (Gymnasium).
    
    :param env: entorno Gymnasium (ej.: gym.make("Taxi-v3"))
    :param n_episodes: número de episodios de entrenamiento
    :param max_steps: máx. pasos por episodio
    :param lr: tasa de aprendizaje (learning rate)
    :param gamma_: factor de descuento
    :param max_eps: epsilon inicial (exploración)
    :param min_eps: epsilon mínimo
    :param decay: tasa de decaimiento de epsilon (exponencial)
    :return: Q-table entrenada (numpy array)
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))

    for episode in tqdm(range(n_episodes), desc="Entrenando"):
        # Cálculo dinámico de epsilon (exponential decay)
        epsilon = min_eps + (max_eps - min_eps) * np.exp(-decay * episode)

        state, info = env.reset()
        done = False

        for _step in range(max_steps):
            # Política epsilon-greedy
            if random.uniform(0, 1) > epsilon:
                # Explotación
                action = np.argmax(Q[state, :])
            else:
                # Exploración
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Actualización Q-Learning
            Q[state, action] += lr * (
                reward + gamma_ * np.max(Q[new_state, :]) - Q[state, action]
            )

            state = new_state
            if done:
                break
    return Q


def evaluate_q_table(env, Q, n_episodes_eval, max_steps_eval, seeds):
    """
    Evalúa la Q-table entrenada en varios episodios, usando cada seed de la lista.
    Retorna recompensa media y desviación estándar.
    
    :param env: entorno Gymnasium
    :param Q: Q-table entrenada
    :param n_episodes_eval: número de episodios de evaluación
    :param max_steps_eval: pasos máximos por episodio
    :param seeds: lista de semillas (eval_seed)
    :return: (media_recompensa, std_recompensa)
    """
    rewards = []
    for i in range(n_episodes_eval):
        # Cada episodio se inicia con una semilla distinta
        state, info = env.reset(seed=seeds[i])
        total_reward = 0
        done = False

        for _step in range(max_steps_eval):
            # Acción determinista (greedy)
            action = np.argmax(Q[state, :])
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = new_state
            if done:
                break
        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def push_to_hub(repo_id, model, env, local_repo_path="q-taxi-hub", commit_msg="Q-learning Taxi-v3 Upload"):
    """
    Sube el modelo y metadatos a Hugging Face Hub.
    
    :param repo_id: "Usuario/Repo" en Hugging Face
    :param model: diccionario con la Q-table y metadatos
    :param env: entorno Gymnasium (para reproducibilidad de nombre o desc)
    :param local_repo_path: carpeta local para clonar el repo
    :param commit_msg: mensaje de commit
    """

    # Creamos/Clonamos el repositorio local
    if not os.path.exists(local_repo_path):
        os.makedirs(local_repo_path)

    repo = Repository(local_dir=local_repo_path, clone_from=repo_id)
    
    # Guardar la Qtable en un pickle
    qtable_path = os.path.join(local_repo_path, "qtable.pkl")
    with open(qtable_path, "wb") as f:
        pickle.dump(model["qtable"], f)

    # Guardar metadatos en JSON
    metadata_path = os.path.join(local_repo_path, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(model, f, indent=4)

    # Hacer push al repo
    repo.push_to_hub(commit_message=commit_msg)
    print(f"\n¡Modelo subido con éxito! Mira tu repo aquí:\nhttps://huggingface.co/{repo_id}")


def main():
    # 1) Crear el entorno de Taxi-v3
    env = gym.make(env_id)

    # 2) Entrenar el agente Q-Learning
    print("Entrenando el agente Q-Learning...")
    Qtable_taxi = train_q_learning(
        env,
        n_episodes=n_training_episodes,
        max_steps=max_steps,
        lr=learning_rate,
        gamma_=gamma,
        max_eps=max_epsilon,
        min_eps=min_epsilon,
        decay=decay_rate
    )

    # 3) Evaluar el agente con seeds fijas (NO se modifican)
    print("\nEvaluando el agente con seeds fijas...")
    mean_reward, std_reward = evaluate_q_table(
        env, Qtable_taxi, n_eval_episodes, max_steps, eval_seed
    )
    print(f"Recompensa promedio: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 4) Preparar el diccionario del modelo
    model_dict = {
        "env_id": env_id,
        "max_steps": max_steps,
        "n_training_episodes": n_training_episodes,
        "n_eval_episodes": n_eval_episodes,
        "eval_seed": eval_seed,  # <- NO modificar
        "learning_rate": learning_rate,
        "gamma": gamma,
        "max_epsilon": max_epsilon,
        "min_epsilon": min_epsilon,
        "decay_rate": decay_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "qtable": Qtable_taxi.tolist()  # Para guardarlo en JSON sin problemas
    }

    # 5) Subir a la Hugging Face Hub (opcional)
    # Reemplaza con tu nombre de usuario y nombre de repo en HF
    username = "Barearojojuan"  # <--- ¡Cambia esto!
    repo_name = "q-learning-taxi-v3"  # <--- ¡Cambia esto!
    repo_id = f"{username}/{repo_name}"

    # Subida al Hub
    push_to_hub(repo_id=repo_id, model=model_dict, env=env)

    print("\nProceso completo. Puedes comparar en la leaderboard:\n"
          "https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard")


if __name__ == "__main__":
    main()
