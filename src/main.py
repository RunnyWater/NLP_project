from nlp.text_processor import DictionaryProcessor, messageProcessor
from database.database_manager import DatabaseManager
from ai.artificial_intelligence import IntentClassifier


def main():
    # dir = 'txt_subs' # Directory that contains txt files that needed to be converted to csv
    # db_manager = DatabaseManager()
    # Dictionary_processor = DictionaryProcessor(dir, db_manager)
    # message_processor = messageProcessor()
    ai = IntentClassifier()
    print(ai.classify("Hello, how are things going?"))

if __name__ == '__main__':
    main() 