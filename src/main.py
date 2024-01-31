import numpy as np
from nlp.text_processor import DictionaryProcessor, messageProcessor
from database.database_manager import DatabaseManager


def main():
    dir = 'txt_subs' # Directory that contains txt files that needed to be converted to csv
    db = 'database.sql'
    db_manager = DatabaseManager()
    processor = messageProcessor()



def decode_text(encoded_text, word_to_index):
    # Replace each integer in the encoded text with its corresponding word
    decoded_text = [word_to_index.get(index, 'UNK') for index in encoded_text]

    return ' '.join(decoded_text)


def encode_text(text):
    # Split the text into words
    words = text.split()

    # Create a dictionary to map each word to a unique integer
    word_to_index = {}
    for i, word in enumerate(words):
        word_to_index[word] = i

    # Replace each word in the text with its corresponding integer
    encoded_text = [word_to_index[word] for word in words]

    return encoded_text, word_to_index

if __name__ == '__main__':
    main()