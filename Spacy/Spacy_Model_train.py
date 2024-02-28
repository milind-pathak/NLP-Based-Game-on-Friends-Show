import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import string

def adjust_entity_span(text, start, end):
    # Ensure start and end are within the string bounds
    if start < 0 or end > len(text):
        return None, None

    # Adjust the end index to exclude trailing punctuation
    while end > start and text[end - 1] in string.punctuation:
        end -= 1
    # Adjust the start index to exclude leading punctuation
    while start < end and text[start] in string.punctuation:
        start += 1

    return start, end


# Load your CSV data
data = pd.read_csv('/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Datasets/Action_Response_Pairs.csv')

# Preprocess data to SpaCy format
TRAIN_DATA = []
for _, row in data.iterrows():
    text = row['Action_Description']
    start = text.find(row['Player_Input'])
    if start != -1:
        end = start + len(row['Player_Input'])
        # Adjust for punctuation before adding to TRAIN_DATA
        start, end = adjust_entity_span(text, start, end)
        if start is not None and end is not None and start < end:
            TRAIN_DATA.append((text, {"entities": [(start, end, "ACTION")]}))


# Load a pre-existing SpaCy model or create a blank one
# If you have a model already, use spacy.load('en_core_web_sm')
# If not, start with a blank model: spacy.blank('en')
# Create a blank English Language model
nlp = spacy.blank('en')
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner')
ner = nlp.get_pipe('ner')
ner.add_label('ACTION')

# Disable other pipeline components during training
with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "ner"]):
    optimizer = nlp.begin_training()
    for itn in range(100):  # Example: 30 iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        # Batch up the examples using SpaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
        print(losses)

# Save the trained model to a directory
model_to_save = '/Users/YouShallNotPass/Desktop/NLP Project/Friends Game/Modelsv3/Spacy'
nlp.to_disk(model_to_save)

# Load the trained model from the directory
nlp = spacy.load(model_to_save)

# Test sentences
test_sentences = [
    "I want to hug Joey",
    "I am feeling very excited today",
    "Chandler decides to play a prank on Ross"
]

# Process the test sentences and print the entities
for sentence in test_sentences:
    doc = nlp(sentence)
    entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    print(f"Entities in '{sentence}': {entities}")
