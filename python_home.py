import json
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import spacy
from spacy.matcher import Matcher

# Load your home configuration from a JSON file
def load_home_config():
    with open("home_configuration.json", "r") as config_file:
        return json.load(config_file)

# Save the updated home configuration to the JSON file
def save_home_config(home_config):
    with open("home_configuration.json", "w") as config_file:
        json.dump(home_config, config_file, indent=4)

# Initialize the smaller language model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Set pad_token_id to eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# Initialize spaCy with the English model
nlp = spacy.load("en_core_web_sm")

# Define a spaCy Matcher for intent and entity recognition
matcher = Matcher(nlp.vocab)

# Define patterns for intent and entity recognition
intent_patterns = [
    {"LOWER": "turn", "POS": {"IN": ["VERB"]}},
    {"LOWER": {"IN": ["on", "off"]}, "POS": {"IN": ["ADP"]}},
]
entity_patterns = [
    {"POS": "NOUN"},
]

matcher.add("INTENT", [intent_patterns])
matcher.add("ENTITY", [entity_patterns])

def process_prompt(prompt_text):
    # Generate a response based on the user's input prompt
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Create an attention mask of ones
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Process the response and take action based on recognized intent and entity
    intent, entity = recognize_intent(prompt_text)
    perform_action(intent, entity)

def recognize_intent(user_input):
    doc = nlp(user_input)
    intent = None
    entity = None

    # Use spaCy Matcher to identify intent and entity
    matches = matcher(doc)

    for match_id, start, end in matches:
        match_text = doc[start:end].text
        if match_id == nlp.vocab.strings["INTENT"]:
            intent = match_text.lower()
        elif match_id == nlp.vocab.strings["ENTITY"]:
            entity = match_text.lower()

    return intent, entity

def perform_action(intent, entity):
    home_config = load_home_config()  # Load the current configuration

    # Implement logic to perform the corresponding action based on the recognized intent and entity
    if intent == "turn on" and entity:
        entity = entity.lower()  # Convert to lowercase for case-insensitive matching
        found_device = False

        for room_name, room_info in home_config["rooms"].items():
            for device_name, device_info in room_info["devices"].items():
                if entity in device_name.lower() and device_info.get("power_socket"):
                    power_socket_name = device_info["power_socket"]
                    if power_socket_name in room_info["power_sockets"]:
                        if room_info["power_sockets"][power_socket_name]["state"] == "off":
                            room_info["power_sockets"][power_socket_name]["state"] = "on"
                            print(f"Turned on {entity} in {room_name}.")
                            found_device = True
                        else:
                            print(f"{entity} in {room_name} is already turned on.")
                            found_device = True
                        break
        
        if not found_device:
            print(f"I couldn't find {entity} or its socket in any room.")
    elif intent == "turn off" and entity:
        entity = entity.lower()  # Convert to lowercase for case-insensitive matching
        found_device = False

        for room_name, room_info in home_config["rooms"].items():
            for device_name, device_info in room_info["devices"].items():
                if entity in device_name.lower() and device_info.get("power_socket"):
                    power_socket_name = device_info["power_socket"]
                    if power_socket_name in room_info["power_sockets"]:
                        if room_info["power_sockets"][power_socket_name]["state"] == "on":
                            room_info["power_sockets"][power_socket_name]["state"] = "off"
                            print(f"Turned off {entity} in {room_name}.")
                            found_device = True
                        else:
                            print(f"{entity} in {room_name} is already turned off.")
                            found_device = True
                        break
        
        if not found_device:
            print(f"I couldn't find {entity} or its socket in any room.")
    else:
        print("I'm not sure how to respond to that.")
    
    save_home_config(home_config)  # Save the updated configuration

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        process_prompt(user_input)

