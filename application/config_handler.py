class ConfigHandler:
    def __init__(self):
        self.config = {
            "task_name": "walk",
            "domain_name": "walker",
            "image_size": 64,
            "action_repeat": 2,
            "latent_size": 1024,
            "stoch_size": 32,
            "deter_size": 1024,
            "max_steps": 1000000,
            "collect_steps": 500,
            "batch_size": 16,
            "batch_length": 50,
            "horizon": 15,
            "world_lr": 0.0003,
            "actor_lr": 8e-05,
            "critic_lr": 8e-05,
            "kl_free": 1.0,
            "kl_scale": 1.0,
            "actor_ent": 0.0001,
            "discount": 0.99,
            "lambda_": 0.95,
            "buffer_capacity": 25000,  # Reduced to fit memory
            "log_dir": "./logs",
            "experiment_name": "dmc_walker_walk"
        }

    def get_config(self):
        return self.config

    def update_config(self, updates):
        self.config.update(updates)