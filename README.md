# NLP-Based-RPG-on-Friends-Show
The project's scope is to create an interactive narrative experience where players can input text, and the system responds with contextually relevant narrative and dialogues based on the TV show "Friends."

Key Takeaways:
● We need to build a world or multiple scenes which players can
choose as their starting point.
● The game should be able to understand and incorporate user’s
actions.
● Since this is a text-based game, the scenes have to be narrated.
● Interactions with other characters should be relevant to player’s
input.

Models Used:
● Player Input Processing (SpaCy NER Model)
● Narration Generation (GPT-2 Model/ T5)
● Character Dialogue Generation (GPT-2-Neo-1.3B)

Models Interaction Flow:
● SpaCy NER Model: Identifies and classifies actions within player input.
● GPT-2 Model (Narrator): Receives the classified action as a prompt and generates narrative text.
● GPT-2-Neo-1.3B Model (Character Dialogue): Takes the generated narrative and produces character-specific dialogue.

The project demonstrates the potential of NLP to create new, innovative and engaging experiences in gaming industry. Once trained properly with an appropriate dataset, we will be able to create a whole world where the players will be free to move freely to different locations and interact with different characters and perform multiple actions.
 
