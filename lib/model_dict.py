# CACHE_DIR_BASE = "/data/locus/llm_weights"
CACHE_DIR_BASE = "/data/locus/project_data/project_data3/abair"

MODEL_DICT_LLMs = {

    "gpt2": {
        "model_id": "openai-community/gpt2",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights"
    },

    # "gemma-2-27b-it":{
    #     "model_id": "google/gemma-2-27b-it",
    #     "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    # },

    "mistral-7b-v3-it": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },

    "pythia-70m": {
        "model_id": "EleutherAI/pythia-70m-deduped",
        "cache_dir": CACHE_DIR_BASE,
    },

    "pythia-6.9b": {
        "model_id": "EleutherAI/pythia-6.9b-deduped",
        "cache_dir": CACHE_DIR_BASE,
    },

    "qwen2.5-7b-it": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    }, 

    "llama3.1-8b-base": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "cache_dir": CACHE_DIR_BASE,
    },

    "llama3.1-70b-it": {
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },

    "gemma-2-9b-base": {
        "model_id": "google/gemma-2-9b",
        "cache_dir": CACHE_DIR_BASE,
    },

    "gemma-2-2b-base": {
        "model_id": "google/gemma-2-2b",
        "cache_dir": CACHE_DIR_BASE,
    },

    "gemma-2-9b-it": {
        "model_id": "google/gemma-2-9b-it",
        "cache_dir": CACHE_DIR_BASE,
    },

    "gemma-2-2b-it": {
        "model_id": "google/gemma-2-2b-it",
        "cache_dir": CACHE_DIR_BASE,
    },

    "mistral-7b-v3": {
        "model_id": "mistralai/Mistral-7B-v0.3",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },


    "llama3.1-8b-it": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "cache_dir": "/data/locus/llm_weights",
    },

    "llama3-8b-it-fp16": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "cache_dir": "/data/locus/llm_weights",
    },


    "llama3.1-8b-it-fp8": {
        "model_id": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },

    "llama3.1-8b-it-int4": {
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },

    "llama3.1-8b-it-int4-gptq": {
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },

    "llama3.1-8b-it-int4-bnb": {
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },


    ######################################################################
    "llama3.1-8b-it": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "cache_dir": "/data/locus/project_data/project_data3/abair",
    },

    "llama3.1-8b-it-fp8": {
        "model_id": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },

    "llama3.1-8b-it-int4-awq": {
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights",
    },

    ########################################################################################
    "olmo-7b": {
        "model_id": "allenai/OLMo-7B",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights/",
    },

    ### llama3 model:
    "llama3-8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "cache_dir": "/data/models/huggingface/meta-llama/Meta-Llama-3-8B",
    },

    "llama3-70b": {
        "model_id": "meta-llama/Meta-Llama-3-70B",
        "cache_dir": "/data/models/huggingface/meta-llama/Meta-Llama-3-70B",
    },
    "llama3-70b-instruct": {
        "model_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "cache_dir": "/data/models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct",
    },

    # "llama3.1-8b-instruct": {
    #     "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     "cache_dir": CACHE_DIR_BASE,
    # },

    ### phi-3 model 
    "phi-3-small-8k": {
        "model_id": "microsoft/Phi-3-small-8k-instruct",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights/",
    },
    "phi-3-mini-4k": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights/",
    },
    "phi-3-medium-4k": {
        "model_id": "microsoft/Phi-3-medium-4k-instruct",
        "cache_dir": "/data/locus/project_data/project_data3/llm_weights/",
    },

    ### llama2 model
    "llama2_7b": {
        "model_id": "meta-llama/Llama-2-7b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_13b": {
        "model_id": "meta-llama/Llama-2-13b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_70b": {
        "model_id": "meta-llama/Llama-2-70b-hf", 
        "cache_dir": CACHE_DIR_BASE
    },

    ### llama2 chat model
    "llama2_7b_chat": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 
    "llama2_13b_chat": {
        "model_id": "meta-llama/Llama-2-13b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 
    "llama2_70b_chat": {
        "model_id": "meta-llama/Llama-2-70b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 

    ### mistral model 
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe": {
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_7b_instruct":{
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe_instruct": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    }, 

    ### phi-2
    "phi-2": {
        "model_id": "microsoft/phi-2",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### falcon model
    "falcon_7b": {
        "model_id": "tiiuae/falcon-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon_40b": {
        "model_id": "tiiuae/falcon-40b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### mpt model 
    "mpt_7b": {
        "model_id": "mosaicml/mpt-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mpt_30b": {
        "model_id": "mosaicml/mpt-30b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### opt model 
    "opt_125m": {
        "model_id": "facebook/opt-125m", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_350m": {
        "model_id": "facebook/opt-350m", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_1.3b": {
        "model_id": "facebook/opt-1.3b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_2.7b": {
        "model_id": "facebook/opt-2.7b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_7b": {
        "model_id": "facebook/opt-6.7b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_13b": {
        "model_id": "facebook/opt-13b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_30b": {
        "model_id": "facebook/opt-30b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_66b": {
        "model_id": "facebook/opt-66b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### gpt2 model 
    "gpt2_medium": {
        "model_id": "gpt2-medium",
        "cache_dir": CACHE_DIR_BASE
    },
    "gpt2_large": {
        "model_id": "gpt2-large",
        "cache_dir": CACHE_DIR_BASE
    },
    "gpt2_xl": {
        "model_id": "gpt2-xl",
        "cache_dir": CACHE_DIR_BASE
    },
}