import os
import json
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

def objective(trial):
    """
    Función objetivo para Optuna.
    Se sugiere un conjunto de hiperparámetros, se entrena un modelo PPO durante un número reducido de timesteps
    y se evalúa el modelo retornando la recompensa media obtenida.
    """
    env_id = "LunarLander-v2"
    
    # Crear entorno vectorizado para entrenamiento y entorno monitorizado para evaluación
    vec_env = make_vec_env(env_id, n_envs=8)
    eval_env = Monitor(gym.make(env_id))
    
    # Sugerir hiperparámetros
    n_steps = trial.suggest_int("n_steps", 256, 2048, step=256)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    # Asegurar que n_steps sea múltiplo de batch_size (requisito de PPO)
    if n_steps % batch_size != 0:
        n_steps = batch_size * (n_steps // batch_size)
    
    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # Crear el modelo PPO con los hiperparámetros sugeridos y forzando el uso de la CPU
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        device="cpu",
        verbose=0
    )
    
    # Entrenar el modelo durante un número reducido de timesteps para la optimización
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluar el modelo en 5 episodios
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    
    # Cerrar los entornos para liberar recursos
    vec_env.close()
    eval_env.close()
    
    return mean_reward

def main():
    n_trials = 20
    study = optuna.create_study(direction="maximize")
    
    # Callback para mostrar el progreso de la optimización
    def print_progress(study, trial):
        print(f"Trial {trial.number + 1}/{n_trials} completado. Mejor recompensa hasta ahora: {study.best_value:.2f}")
    
    # Ejecutar la optimización con el callback
    study.optimize(objective, n_trials=n_trials, callbacks=[print_progress])
    
    # Mostrar los mejores hiperparámetros encontrados
    print("\nMejores parámetros encontrados:")
    best_trial = study.best_trial
    print(f"  Recompensa media: {best_trial.value:.2f}")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Guardar los mejores hiperparámetros en un archivo en la carpeta 'hiper_optimos'
    output_dir = "hiper_optimos"
    os.makedirs(output_dir, exist_ok=True)
    best_params_file = os.path.join(output_dir, "best_params.json")
    with open(best_params_file, "w") as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"\nLos mejores hiperparámetros se han guardado en: {best_params_file}")
    
    # Entrenar el modelo final con los mejores hiperparámetros
    best_params = best_trial.params
    n_steps = best_params["n_steps"]
    batch_size = best_params["batch_size"]
    if n_steps % batch_size != 0:
        n_steps = batch_size * (n_steps // batch_size)
    
    env_id = "LunarLander-v2"
    vec_env = make_vec_env(env_id, n_envs=8)
    eval_env = Monitor(gym.make(env_id))
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=best_params["n_epochs"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        ent_coef=best_params["ent_coef"],
        learning_rate=best_params["learning_rate"],
        device="cpu",  # Forzamos el uso de la CPU
        verbose=1
    )
    
    total_timesteps = 1_000_000
    print(f"\nEntrenando el modelo final durante {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Guardar el modelo final
    model_name = "lunar-lander/ppo-LunarLander-v2"
    model.save(model_name)
    print(f"Modelo guardado como: {model_name}.zip")
    
    # Evaluar el modelo final
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Recompensa media tras evaluación: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # ---------------------------------------
    # PRUEBA VISUAL
    # ---------------------------------------
    # Para visualizar en Gymnasium se debe especificar 'render_mode'
    test_env = gym.make(env_id, render_mode="human")
    
    episodes = 5
    for ep in range(episodes):
        obs, info = test_env.reset()
        done = False
        total_rewards = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_rewards += reward
        
        print(f"Episodio {ep+1}/{episodes} - Recompensa total: {total_rewards:.2f}")
    
    test_env.close()
    print("Prueba finalizada.")

if __name__ == "__main__":
    main()
