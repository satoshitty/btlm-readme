## Loading the BTLM Model from Huggingface

When loading the BTLM model from Huggingface, it's important to use a consistent and compatible environment, especially when working with the `bitsandbytes` package. It's recommended to use a fresh `conda` or `pyenv` environment to avoid any conflicts.

### Step-by-step Guide:

1. **Setting up the Environment**:
    - If you're using `conda`, create a new environment:
        ```bash
        conda create --name btlm_env python=3.8
        conda activate btlm_env
        ```
    - If you're using `pyenv`, create a new virtual environment:
        ```bash
        pyenv virtualenv 3.8.10 btlm_env
        pyenv activate btlm_env
        ```

2. **Install Required Packages**:
    Run the following commands in your terminal:
    ```bash
    pip install -q -U bitsandbytes
    pip install -q -U scipy
    pip install -q -U git+https://github.com/huggingface/transformers.git
    pip install -q -U git+https://github.com/huggingface/peft.git
    pip install -q -U git+https://github.com/huggingface/accelerate.git
    ```

3. **Load the Model in Python**:
    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        'cerebras/btlm-3b-8k-base', 
        load_in_4bit=True, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Check if the model is in 4bit
    print(model.transformer.h[3].attn.c_attn)
    ```

4. **Note**: In case you face any issues, please ensure that you're in the correct environment and all packages are installed correctly.

Good luck, and happy coding! 🚀
