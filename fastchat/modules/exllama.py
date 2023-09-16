class ExllamaConfig:
    max_seq_len:int
    gpu_split:str


def load_exllama_model(model_path, exllama_config: ExllamaConfig):
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Config,
        ExLlamaV2Tokenizer,
    )
    exllamav2_config = ExLlamaV2Config()
    exllamav2_config.model_dir = model_path
    exllamav2_config.prepare()
    exllamav2_config.max_seq_len = exllama_config.max_seq_len

    model = ExLlamaV2(exllamav2_config)
    tokenizer = ExLlamaV2Tokenizer(exllamav2_config)
    # TODO: Add GPU split later
    model.load()
    return model, tokenizer

def init_exllama_cache(model):
    from exllamav2 import ExLlamaV2Cache
    return ExLlamaV2Cache(model)