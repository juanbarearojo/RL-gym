"""
PASO PREVIO:
-----------
1) Instala huggingface_hub y huggingface_sb3 (si no lo has hecho):
    pip install huggingface_hub huggingface_sb3

2) Inicia sesión en tu cuenta de Hugging Face:
   - Si estás en un notebook (Colab/Jupyter):
        from huggingface_hub import notebook_login
        notebook_login()
   - O desde la terminal/VSCode:
        huggingface-cli login
   Pega tu token con permisos 'write'.
"""

import gymnasium as gym
from gymnasium.wrappers import Monitor

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Librería para subir modelos a Hugging Face
from huggingface_sb3 import package_to_hub

def main():
    """
    Ejemplo de cómo subir un modelo PPO entrenado en LunarLander a la Hugging Face Hub.
    """

    # --------------------------------------------------------------------------------
    # 1) CARGAR TU MODELO ENTRENADO
    # --------------------------------------------------------------------------------
    # Supongamos que entrenaste y guardaste tu modelo bajo el nombre 'ppo-LunarLander-v2'.
    # Si lo tienes en memoria (variable model), omite la carga.
    model_name = "ppo-LunarLander-v2"
    model = PPO.load(model_name)

    # --------------------------------------------------------------------------------
    # 2) DEFINIR CONFIGURACIONES DEL ENTORNO Y PARAMETROS
    # --------------------------------------------------------------------------------
    # Usaremos "LunarLander-v2" (gymnasium) o "LunarLander-v3" si tienes gymnasium con box2d.
    # Ajusta según tu entorno.
    env_id = "LunarLander-v2"  
    model_architecture = "PPO"

    # Creamos un DummyVecEnv que envuelve un entorno de gym con render en modo "rgb_array"
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

    # Este repo_id debe ser único y seguir el formato "usuario_de_HF/nombre_de_repo".
    # e.g. "tuUsuarioDeHF/ppo-LunarLander-v2"
    repo_id = "TU_USUARIO_HF/ppo-LunarLander-v2"

    # Mensaje de commit para la subida
    commit_message = "Upload PPO LunarLander-v2 trained agent"

    # --------------------------------------------------------------------------------
    # 3) SUBIR EL MODELO A HUGGING FACE
    # --------------------------------------------------------------------------------
    # package_to_hub generará:
    #   - un archivo README (model card),
    #   - un video de la evaluación,
    #   - un archivo de metadatos,
    #   - y subirá todo a tu repositorio en la Hugging Face Hub.
    package_to_hub(
        model=model,                       # Tu modelo entrenado (PPO, DQN, etc.)
        model_name=model_name,             # Nombre del archivo del modelo (sin .zip)
        model_architecture=model_architecture,  # Arquitectura: PPO, DQN, A2C...
        env_id=env_id,                     # Nombre del entorno
        eval_env=eval_env,                 # Entorno que se usará para grabar video y evaluar
        repo_id=repo_id,                   # ID del repo en HF: "usuario/nombre-repo"
        commit_message=commit_message      # Mensaje para el commit
    )

    print("\n¡Modelo subido exitosamente a la Hugging Face Hub!")
    print(f"Revisa tu repositorio en: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
