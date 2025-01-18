import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env  # <-- Import desde env_util
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    """
    Ejemplo de entrenamiento, evaluación y prueba de un modelo PPO
    con el entorno LunarLander-v3 de Gymnasium, incluyendo renderizado.
    """

    # --------------------
    # 1) CREAR EL ENTORNO
    # --------------------
    env_id = "LunarLander-v3"
    # Creamos un entorno vectorizado con 2 procesos para entrenar
    vec_env = make_vec_env(env_id, n_envs=2)

    # Para evaluación con un monitor:
    eval_env = Monitor(gym.make(env_id))

    # ----------------------------
    # 2) DEFINIR EL MODELO (PPO)
    # ----------------------------
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

    # -------------------------
    # 3) ENTRENAR EL MODELO
    # -------------------------
    total_timesteps = 200_000  # Ajusta según tus recursos
    print(f"Entrenando durante {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # -------------------------
    # 4) GUARDAR EL MODELO
    # -------------------------
    model_name = "lunar-lander/ppo-LunarLander-v3"  # Renombrado para evitar confusión
    model.save(model_name)
    print(f"Modelo guardado como: {model_name}.zip")

    # -------------------------
    # 5) EVALUAR EL MODELO
    # -------------------------
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=30, 
        deterministic=True
    )
    print(f"Recompensa media tras evaluación: {mean_reward:.2f} +/- {std_reward:.2f}")

    # ---------------------------------------
    # 6) PRUEBA VISUAL CON RENDERIZACIÓN
    # ---------------------------------------
    # Crea un entorno normal (no vectorizado) con render_mode="human"
    test_env = gym.make(env_id, render_mode="human")

    episodes = 5
    for ep in range(episodes):
        obs, info = test_env.reset()  # Gymnasium: reset() devuelve (obs, info)
        done = False
        truncated = False
        total_rewards = 0

        while not (done or truncated):
            # Si quieres llamar manualmente al render (opcional):
            # test_env.render()

            # El modelo predice la acción dada la observación actual
            action, _ = model.predict(obs, deterministic=True)

            # Paso en el entorno con Gymnasium API:
            obs, reward, done, truncated, info = test_env.step(action)

            total_rewards += reward

        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_rewards:.2f}")

    test_env.close()
    print("Prueba finalizada.")

if __name__ == "__main__":
    main()
