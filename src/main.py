import numpy as np
from nlp.text_processor import DictionaryProcessor, messageProcessor
from database.database_manager import DatabaseManager


def main():
    dir = 'txt_subs' # Directory that contains txt files that needed to be converted to csv
    db = 'database.sql'
    db_manager = DatabaseManager()
    processor = messageProcessor()


if __name__ == '__main__':
    main()