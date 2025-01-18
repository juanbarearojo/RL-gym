"""
Script unificado: entrena un modelo PPO en LunarLander-v2, prueba localmente
y sube el modelo entrenado a la Hugging Face Hub.

PASOS PREVIOS:
--------------
1) Instala las dependencias (si no lo has hecho):
       pip install gymnasium box2d-py stable-baselines3 huggingface_hub huggingface_sb3

2) Inicia sesión en tu cuenta de Hugging Face con permiso de escritura:
       huggingface-cli login

3) Ajusta la variable 'repo_id' con tu usuario y nombre de repositorio
   en la Hugging Face Hub, por ejemplo "MiUsuario/ppo-LunarLander-v2"
"""

import gymnasium as gym
# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# Librería para subir modelos a Hugging Face
from huggingface_sb3 import package_to_hub

def main():
    # ------------------------------------------------------------------------
    # 1) ENTRENAMIENTO DEL MODELO
    # ------------------------------------------------------------------------
    env_id = "LunarLander-v2"
    vec_env = make_vec_env(env_id, n_envs=8)  # entorno vectorizado

    # Este entorno se usa para evaluación (registra métricas con Monitor)
    eval_env = Monitor(gym.make(env_id))

    # Definimos el modelo PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1
    )

    # Entrenamiento
    total_timesteps = 1_000_000
    print(f"Entrenando durante {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Guardamos el modelo (carpeta "lunar-lander" + nombre del zip)
    model_path = "lunar-lander/ppo-LunarLander-v2"
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}.zip")

    # ------------------------------------------------------------------------
    # 2) EVALUACIÓN DEL MODELO (cálculo de recompensa media)
    # ------------------------------------------------------------------------
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Recompensa media tras evaluación: {mean_reward:.2f} +/- {std_reward:.2f}")

    # ------------------------------------------------------------------------
    # 3) PRUEBA VISUAL LOCAL (con render_mode="human")
    # ------------------------------------------------------------------------
    test_env = gym.make(env_id, render_mode="human")

    episodes = 5
    for ep in range(episodes):
        obs, info = test_env.reset()
        done = False
        total_rewards = 0.0

        while not done:
            # Predice la acción de manera determinista
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_rewards += reward

        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_rewards:.2f}")

    test_env.close()
    print("Prueba local finalizada.")

    # ------------------------------------------------------------------------
    # 4) SUBIR EL MODELO A LA HUGGING FACE HUB
    # ------------------------------------------------------------------------
    # Si deseas omitir la subida, puedes comentar la sección siguiente

    # a) Ajusta tu repo en Hugging Face: "TU_USUARIO/algún_nombre"
    repo_id = "Barearojojuan/ppo-LunarLander-v2"
    commit_message = "Upload PPO LunarLander-v2 trained agent"

    # Creamos un DummyVecEnv para que package_to_hub pueda grabar un video de evaluación
    # Gymnasium necesita render_mode="rgb_array" para capturar fotogramas.
    hf_eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

    # Llamamos a package_to_hub, pasando el modelo ya entrenado en memoria
    # model_name: nombre "lógico" que se usará en la metadata (y para .zip)
    package_to_hub(
        model=model,
        model_name="ppo-LunarLander-v2",      # aparecerá como "ppo-LunarLander-v2.zip" en HF
        model_architecture="PPO",
        env_id=env_id,
        eval_env=hf_eval_env,                # se usará para grabar video de evaluación
        repo_id=repo_id,
        commit_message=commit_message
    )

    print("\n¡Modelo subido exitosamente a la Hugging Face Hub!")
    print(f"Revisa tu repositorio en: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
