class Config:
    def __init__(self, model_name="default_model"):
        self.set_config(model_name)

    def config_obstacles_10_2(self):
        self.nr_inner_iter = 60
        self.nr_initial_iter = 400
        self.extraction_epochs = 501

    def config_obstacles_8_5(self):
        self.nr_inner_iter = 70
        self.nr_initial_iter = 400
        self.extraction_epochs = 501

    def config_avoid(self):
        self.nr_inner_iter = 25
        self.nr_initial_iter = 200
        self.extraction_epochs = 501

    def config_rover(self):
        self.nr_inner_iter = 100
        self.nr_initial_iter = 400
        self.extraction_epochs = 501

    def config_network(self):
        self.nr_inner_iter = 150
        self.nr_initial_iter = 400
        self.extraction_epochs = 501

    def config_drone(self):
        self.nr_inner_iter = 150
        self.nr_initial_iter = 400
        self.extraction_epochs = 5001

    def config_maze(self):
        self.nr_inner_iter = 150
        self.nr_initial_iter = 400
        self.extraction_epochs = 501

    def config_default_model(self):
        self.nr_inner_iter = 50
        self.nr_initial_iter = 400
        self.extraction_epochs = 501

    def config_obstacles_10_5(self):
        self.nr_inner_iter = 100
        self.nr_initial_iter = 400
        self.extraction_epochs = 501

    def config_avoid_large_old(self):
        self.nr_inner_iter = 35
        self.nr_initial_iter = 200
        self.extraction_epochs = 501

    def config_avoid_large(self):
        self.nr_inner_iter = 35
        self.nr_initial_iter = 250
        self.extraction_epochs = 5001

    def config_moving_obstacles(self):
        self.nr_inner_iter = 50
        self.nr_initial_iter = 400
        self.extraction_epochs = 5001

    def translation_model_name_dict(self):
        return {
            "obstacles-10-2": "obstacles_10_2",
            "obstacles-8-5": "obstacles_8_5",
            "obstacles-10-5": "obstacles_8_5",
            "avoid": "avoid",
            "avoid-large": "avoid_large",
            "rover": "rover",
            "network": "network",
            "drone-2-6-1": "drone",
            "maze-10": "maze",
            "moving-obstacles": "moving_obstacles",
        }

    def set_config(self, model_name="default_model"):
        print(model_name)
        translation_dict = self.translation_model_name_dict()
        if model_name in translation_dict:
            method_name = f"config_{translation_dict[model_name]}"
        else:
            method_name = f"config_default_model"
        method = getattr(self, method_name)
        method()
