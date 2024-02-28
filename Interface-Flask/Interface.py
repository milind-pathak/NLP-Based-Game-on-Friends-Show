from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
from flask import Flask, render_template, request, jsonify
import spacy
from difflib import SequenceMatcher

app = Flask(__name__)

# Load the SpaCy NER model
ner_model = spacy.load("/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv3/Spacy")

# Load the GPT-2 Narrator model and tokenizer
model_narrator = GPT2LMHeadModel.from_pretrained("/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv3/gpt2-finetuned-narrator")
tokenizer_narrator = GPT2Tokenizer.from_pretrained("/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv3/gpt2-finetuned-narrator")

# Load the GPT-2 Dialogue model and tokenizer
model_dialogue = GPTNeoForCausalLM.from_pretrained("/Users/YouShallNotPass/Desktop/Friends_Chatbot/1.3B Chatbot/Model-1.3-Full/Model-Neo1.3B")
tokenizer_dialogue = GPT2Tokenizer.from_pretrained("/Users/YouShallNotPass/Desktop/Friends_Chatbot/1.3B Chatbot/Model-1.3-Full/Model-Neo1.3B")

# model_dialogue = GPT2LMHeadModel.from_pretrained("/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv2/gpt2-finetuned-dialogue")
# tokenizer_dialogue = GPT2Tokenizer.from_pretrained("/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv2/gpt2-finetuned-dialogue")


def remove_repeated_sentences(text):
    sentences = text.split('. ')
    # Store the index of sentences to remove
    remove_indices = set()
    # Compare each sentence to every other sentence
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if j in remove_indices:
                # Skip sentences already marked for removal
                continue
            similarity = similar(sentences[i], sentences[j])
            # If two sentences are very similar, mark one for removal
            if similarity > 0.9:  # Threshold can be adjusted
                remove_indices.add(j)

    # Rebuild the text without the repeated sentences
    return '. '.join([s for i, s in enumerate(sentences) if i not in remove_indices])

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Modify the generate_dialogue function to include post-processing
def generate_dialogue(input_text):
    inputs = tokenizer_dialogue.encode(input_text, return_tensors='pt')
    outputs = model_dialogue.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=1, early_stopping=True, temperature=0.9, top_k=50, top_p=0.65)
    raw_response = tokenizer_dialogue.decode(outputs[0], skip_special_tokens=True)
    processed_response = remove_repeated_sentences(raw_response)
    return processed_response

# Modify the generate_narration function to include post-processing
def generate_narration(input_text):
    inputs = tokenizer_narrator.encode(input_text, return_tensors='pt')
    outputs = model_narrator.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=1, early_stopping=True, temperature=0.9, top_k=50, top_p=0.65)
    raw_response = tokenizer_narrator.decode(outputs[0], skip_special_tokens=True)
    processed_response = remove_repeated_sentences(raw_response)
    return processed_response

import re

def extract_character_and_dialogue(generated_text):
    # Define a regular expression pattern to match the character name and dialogue format
    pattern = r"Character: (.*?) Dialogue: (.*)"
    match = re.match(pattern, generated_text)

    if match:
        character_name = match.group(1).strip()
        dialogue = match.group(2).strip()
        return character_name, dialogue
    else:
        # If the pattern is not found, return defaults or log an error
        return "Unknown", generated_text

# Example scene data (you should load this from your datasets)
scenes = {
    'cafe': "You're at Central Perk, the coffee aroma fills the air."
}

@app.route('/')
def home():
    # Start by presenting the scene choices (cafe or apartment)
    return render_template('index.html', scenes=scenes.keys())

@app.route('/set_scene', methods=['POST'])
def set_scene():
    # Set the selected scene and provide a description
    scene_choice = request.form['scene']  # Corrected from 'scene_choice' to 'scene'
    scene_description = scenes.get(scene_choice, "Scene not found.")
    return jsonify(description=scene_description)


@app.route('/perform_action', methods=['POST'])
def perform_action():
    # Player performs an action
    player_input = request.form['action']
    dialogue_prompt = f"Chandler Bing: {player_input}"
    # Check if the player's input is an interaction with a character
    if "talk to" in player_input:
        # Generate dialogue with the mentioned character
        generated_text = generate_dialogue(dialogue_prompt)
        character_name, dialogue = extract_character_and_dialogue(generated_text)
        return jsonify(character=character_name, dialogue=dialogue, narration="")
    ner_action = ner_model(player_input)
    narration_prompt = f"Chandler Bing: {ner_action}"
    # Otherwise, treat it as an action and narrate the outcome
    narration = generate_narration(narration_prompt)
    # Optionally, generate a dialogue from another character commenting on the action
    generated_text = generate_dialogue(dialogue_prompt)
    character_name, followup_dialogue = extract_character_and_dialogue(generated_text)
    
    return jsonify(character=character_name, dialogue=followup_dialogue, narration=narration)

if __name__ == '__main__':
    app.run(debug=True)
