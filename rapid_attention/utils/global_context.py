import json
import rapid_attention as RA


class rapid_attention_config_obj:
    pass


class rapid_attention_global_context:
    project_root = RA.utils.common.find_project_root()
    config_file_path = project_root / "rapid_attention_config.json"

    @staticmethod
    def load_config():
        if not rapid_attention_global_context.config_file_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {rapid_attention_global_context.config_file_path}"
            )
        config_file_content = json.load(
            open(rapid_attention_global_context.config_file_path, "r")
        )
        for key in config_file_content.keys():
            # 为每个配置项创建一个属性
            setattr(rapid_attention_global_context, key, rapid_attention_config_obj())
            for sub_key, value in config_file_content[key].items():
                setattr(getattr(rapid_attention_global_context, key), sub_key, value)
        rapid_attention_global_context.common_config.dataset_path = (
            rapid_attention_global_context.project_root
            / rapid_attention_global_context.common_config.datasets_relative_path
        )
        rapid_attention_global_context.tokenizer_config.tokenizer_train_data_path = (
            rapid_attention_global_context.common_config.dataset_path
            / rapid_attention_global_context.tokenizer_config.tokenizer_train_data_relative_path
        )
        rapid_attention_global_context.tokenizer_config.tokenizer_dir = (
            rapid_attention_global_context.project_root
            / rapid_attention_global_context.tokenizer_config.tokenizer_relative_dir
        )
        rapid_attention_global_context.train_config.output_dir = (
            rapid_attention_global_context.project_root
            / rapid_attention_global_context.train_config.output_relative_dir
        )
        rapid_attention_global_context.train_config.train_data_path = (
            rapid_attention_global_context.common_config.dataset_path
            / rapid_attention_global_context.train_config.train_data_relative_path
        )

rapid_attention_global_context.load_config()