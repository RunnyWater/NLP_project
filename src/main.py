from nlp.text_processor import DictionaryProcessor, messageProcessor
from database.database_manager import DatabaseManager
from ai.artificial_intelligence import IntentClassifier, ResponseGenerator


def main():
    # dir = 'txt_subs' # Directory that contains txt files that needed to be converted to csv
    # db_manager = DatabaseManager()
    # Dictionary_processor = DictionaryProcessor(dir, db_manager)
    # message_processor = messageProcessor()
    bert = IntentClassifier()
    gpt2 = ResponseGenerator()
    user_input = "I need help with my code"
    intent_id = bert.classify(user_input)
    print(f'Intent ID: {intent_id}')

    intent = "help"
    response = gpt2.generate_response(intent)
    print(f'Generated response: {response}')

if __name__ == '__main__':
    main() 