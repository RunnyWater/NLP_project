import sqlite3
import os, csv
import numpy as np
from nlp.text_processor import messageProcessor



class DatabaseManager:
    def __init__(self):
        # directory where the database will be stored, it will be created if it doesn't exist
        self.make_dir('databases')
        # database is a file name where the db stores
        self.database = 'databases/database.sql'
        self.create_table_if_not_exists_all()


    def create_table_if_not_exists_all(self):
        conn = sqlite3.connect(self.database)
        c = conn.cursor()
        # enteties
        c.execute('''
            CREATE TABLE IF NOT EXISTS entities(
                id INTEGER PRIMARY KEY,
                entity TEXT UNIQUE,
                label TEXT
            )
        ''')  
        # pos tags
        c.execute('''
                CREATE TABLE IF NOT EXISTS pos_tags(
                    id INTEGER PRIMARY KEY,
                    text TEXT UNIQUE,
                    tag TEXT,
                    dep TEXT,
                    head TEXT
                )
            ''') 
        # tokenized words
        c.execute("""
            CREATE TABLE IF NOT EXISTS tokenized_words (
                token INTEGER PRIMARY KEY,
                word TEXT NOT NULL UNIQUE
            )
        """)
        # user message
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_messages (
                message_id INTEGER UNIQUE PRIMARY KEY,
                message TEXT NOT NULL
                theme_id INTEGER,
                answer_message_id INTEGER,
                user_id INTEGER,
            )
        """)
        # AI messages
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_message (
                message_id INTEGER UNIQUE PRIMARY KEY,
                message BLOB
                theme_id INTEGER,
                question_message_id INTEGER,
            )
        """)
        # themes
        c.execute("""
                CREATE TABLE IF NOT EXISTS themes (
                theme_id INTEGER PRIMARY KEY,
                theme_text TEXT NOT NULL,
                message_id INTEGER UNIQUE PRIMARY KEY,
                message BLOB
            )
        """)
        conn.commit()
        conn.close()

    def save_entities_to_sql(self, entities):
        # Connect to the SQLite database
        conn = sqlite3.connect(self.database)
        c = conn.cursor()    
        # Insert the entities into the table
        for entity in entities:
            c.execute("INSERT OR IGNORE INTO entities (entity, label) VALUES (?, ?)", ( entity[0], entity[1]))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()


    def save_pos_tags_to_sql(self, pos_tags):
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(self.database)
            c = conn.cursor()      
            # Insert POS tags into the table
            for tag in pos_tags:
                c.execute("INSERT OR IGNORE INTO pos_tags (text, tag, dep, head) VALUES (?, ?, ?, ?)", tag)

            # Commit the changes and close the connection
            conn.commit()
            conn.close()
        except Exception as e:
            print(f'An error occured while saving the POS tags: {e}')


    def csv_files_to_sql(self):
        dir = 'csv_files'
        conn = sqlite3.connect(self.database)
        c = conn.cursor()
        # Will check for table and create it if it doesn't exist
        
        # go through each csv file to insert into the table
        for file_name in os.listdir(dir):
            print(f'{os.path.join(dir, file_name)} <- processed')
            path = f'{dir}/{file_name}'
            # insert data from csv into the table
            with open(path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    c.execute("""INSERT OR IGNORE INTO tokenized_words (word) VALUES (?)""", (row[0],))
            conn.commit()
        conn.close()

    def clearing_table(self, database, table=None):
        # clear the table without deleting it, just emptying it
        if table is None: print('please provide table name'); exit()
        conn = sqlite3.connect(self.database)
        c = conn.cursor()
        c.execute(f"""DELETE FROM {table}""")
        conn.commit()
        conn.close()
    

    def make_dir(self, name):
        # if doesn't exist make directory
        if not os.path.isdir(name): os.mkdir(name)


    def save_message_to_sql(self, theme_id, answer_message_id, message_id, user_id, message):
        conn = sqlite3.connect(self.database)
        c = conn.cursor()
        c.execute("INSERT user_messages (theme_id, answer_message_id, message_id, user_id, message) VALUES (?, ?, ?, ?, ?)", (theme_id, answer_message_id, message_id, user_id, message)) 
        conn.commit()
        conn.close()


    def select_user_message_by_id(self, message_id):
        conn = sqlite3.connect(self.database)
        c = conn.cursor()
        c.execute("SELECT message FROM user_messages WHERE message_id = ?", (message_id,))
        rows = c.fetchall()
        conn.close()
        return rows