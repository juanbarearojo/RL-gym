import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    env_id = "LunarLander-v2"
    # Crear entorno vectorizado
    vec_env = make_vec_env(env_id, n_envs=8)
    # Monitor para evaluación
    eval_env = Monitor(gym.make(env_id))

    model = PPO(
        "MlpPolicy",
        vec_env,
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

    # Guardar el modelo
    model_name = "lunar-lander/ppo-LunarLander-v2"
    model.save(model_name)
    print(f"Modelo guardado como: {model_name}.zip")

    # Evaluación
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Recompensa media tras evaluación: {mean_reward:.2f} +/- {std_reward:.2f}")

    # ---------------------------------------
    # PRUEBA VISUAL
    # ---------------------------------------
    # Lo correcto en Gymnasium es especificar render_mode:
    test_env = gym.make(env_id, render_mode="human")

    episodes = 5
    for ep in range(episodes):
        # En la nueva API, reset() devuelve (obs, info)
        obs, info = test_env.reset()

        done = False
        total_rewards = 0.0

        while not done:
            # model.predict() requiere solo obs (numpy array)
            action, _states = model.predict(obs, deterministic=True)

            # La nueva API de step() devuelve (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_rewards += reward

        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_rewards:.2f}")

    test_env.close()
    print("Prueba finalizada.")

if __name__ == "__main__":
    main()
