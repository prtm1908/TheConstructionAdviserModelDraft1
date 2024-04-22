import torch
import torch.nn as nn
import transformers
from peft import (
    PeftConfig,
    PeftModel,
    get_peft_model
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import streamlit as st


@st.cache_resource
def load_model():
    PEFT_MODEL = "prtm/ConstructionAdviserModelDraft1"

    config = PeftConfig.from_pretrained(PEFT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_4bit = True

    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, PEFT_MODEL, device_map="auto")

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    return alpaca_prompt, model, tokenizer


def inference(alpaca_prompt, model, tokenizer, instruction, context):
    print("Hi")
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction,
            context,
            "",
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 100, use_cache = True)
    outputs = tokenizer.batch_decode(outputs)

    response_index = outputs[0].find("### Response:")
    response = outputs[0][response_index + len("### Response:"):].strip()

    return response


def main():
    st.title("The Construction Adviser Model Draft 1")

    alpaca_prompt, model, tokenizer = load_model()
    
    with st.sidebar:
        # Input fields aligned to the left in the sidebar
        instruction = st.text_input("Enter instruction: ")
        context = st.text_input("Enter context: ")
    
    # Button to trigger processing
    if st.button("Process"):
        # Call the processing function
        with st.spinner("Inferencing..."):
            result = inference(alpaca_prompt, model, tokenizer, instruction, context)
        
        # Display the result
        st.write(result)

if __name__ == "__main__":
    main()
