from leopard_em.pydantic_models.managers import MatchTemplateManager

#YAML_CONFIG_PATH = "/home/shaharsandhaus/Leopard-EM/15426374/xenon_216_000_0.0_DWS_config.yaml"
#YAML_CONFIG_PATH = "/home/shaharsandhaus/Leopard-EM/data/benchmark_configuration_torch_old.yaml"

#YAML_CONFIG_PATH = "/home/shaharsandhaus/Leopard-EM/data/data2/benchmark_configuration_torch.yaml"
YAML_CONFIG_PATH = "/home/shaharsandhaus/Leopard-EM/data/data2/vk_config.yaml"

ORIENTATION_BATCH_SIZE = 2

def main():
    mt_manager: MatchTemplateManager = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    mt_manager.run_match_template(ORIENTATION_BATCH_SIZE)
    mt_manager.results_to_dataframe().to_csv("results.csv")

# NOTE: invoking from `if __name__ == "__main__"` is necessary
# for proper multiprocessing/GPU-distribution behavior
if __name__ == "__main__":
    main()