from tt2dtm.models import MatchTemplateManager

INPUT_YAML_PATH = "/Users/mgiammar/Documents/benchmark_template_matching_methods/benchmark_configuration_torch_local.yaml"


if __name__ == "__main__":
    match_template_manager = MatchTemplateManager.from_yaml(INPUT_YAML_PATH)
    match_template_manager.run_match_template()
    
    print("Done!")