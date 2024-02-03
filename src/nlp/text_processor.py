import speech_recognition as sr
from textblob import TextBlob
import pandas as pd
import os, string, spacy
from gensim import corpora, models

class Processor():
    def __init__(self):
        try: 
            self.nlp = spacy.load('en_core_web_lg')
        except Exception as e:
            print(f'An error occured while loading the spacy model: {e}')
            print('Make sure you have this model downloaded: https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.2.0/en_core_web_lg-2.2.0.tar.gz')
        self.translator = str.maketrans('', '', string.digits)
        self.translator_punct = str.maketrans('', '', string.punctuation)


class DictionaryProcessor(Processor):
    def __init__(self, dir, database_manager):
        super().__init__()
        self.dir_exists_handle(dir)
        self.make_dir('csv_files')
        self.database_manager = database_manager


    def dir_exists_handle(self, dir):
        if not os.path.isdir(dir): print("Directory doesn't exist, will initiate exit()"); exit(1); print('exit didn\'t work, check the function dir_exists_handle')
        self.dir = dir


    def csv_file_exists(self, file_name):
        file_name = file_name.replace('.txt', '.csv')
        if os.path.isfile(f'csv_files/{file_name}'): print(f"{file_name} already has a csv file"); return True
        return False


    def make_dir(self, name):
        # if doesn't exist make directory
        if not os.path.isdir(name): os.mkdir(name)
    

    def to_csv(self):
        filtered_tokens = []
        for file_name in os.listdir(self.dir):
            print(f'{os.path.join(file_name)} <- processing')
            if self.csv_file_exists(file_name): continue
            file_path = f'{self.dir}/{file_name}'
            txt = ''
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        # lower case
                        line = line.lower()
                        # remove numbers
                        line = line.translate(self.translator)
                        # remove punctuations
                        line = line.translate(self.translator_punct)
                        
                        # Named Entity Recognition
                        entities = [(ent.text, ent.label_) for ent in doc.ents]
                        pos_tags = [(token.text, token.tag_, token.dep_, token.head.text) for token in doc]
                        # tokenize with spacy
                        doc = self.nlp(line)
                        # remove stopwords and lemmatize in one step using spacy
                        filtered_tokens = [token.lemma_ for token in doc if not token.is_stop]
                        # add to return
                        txt = txt + ' '.join(filtered_tokens)
                        # If you want to save entities to a separate file or database
                        # self.save_entities_to_csv(file_path, entities)
                        self.database_manager.save_entities_to_sql(entities)
                        self.database_manager.save_pos_tags_to_sql(pos_tags)
                self.save_data_to_csv(file_path, txt)
                self.delete_txt_file(file_path)
            except Exception as e:
                print(f'An error occured while processing the file: {e}')
                return None
        
    def delete_txt_file(self, file_path):
        os.remove(file_path)
        print(f'{file_path} was deleted')


    def csv_to_sql_handler(self):
        for file_name in os.listdir(self.dir):
            print(f'{os.path.join(file_name)} <- processing')
            file_path = f'{self.dir}/{file_name}'
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # lower case
                    line = line.lower()
                    # remove numbers
                    line = line.translate(self.translator)
                    # remove punctuations
                    line = line.translate(self.translator_punct)
                    # tokenize with spacy
                    doc = self.nlp(line)
                    # Named Entity Recognition
                    entities = [(ent.text, ent.label_) for ent in doc.ents]
                    pos_tags = [(token.text, token.tag_, token.dep_, token.head.text) for token in doc]
                    # Save entities to a separate file or database
                    self.save_entities_to_csv(file_path, entities)
                    self.database_manager.save_entities_to_sql(entities)
                    self.database_manager.save_pos_tags_to_sql(pos_tags)


    def save_entities_to_csv(self, file_path, entities):
        self.make_dir('entities')
        # new path where to save the csv file after processing 
        new_path = file_path.replace(".txt", ".csv")
        new_path = new_path.replace("txt_subs", "entities")
        try:
            with open(new_path, 'w', encoding='utf-8') as file:
                for entity in entities:
                    file.write(entity[0] + ',' + entity[1] + '\n')
        except Exception as e:
            print(f'An error occured while saving the entities to csv file: {e}')
            return None


    def save_data_to_csv(self, file_path, txt):
        # new path where to save the csv file after processing 
        new_path = file_path.replace(".txt", ".csv")
        new_path = new_path.replace("txt_subs", "csv_files")
        try:
            with open(new_path, 'w', encoding='utf-8') as file:
                for word in txt.split():
                    file.write(word + '\n')
        except Exception as e:
            print(f'An error occured while saving the txt data to csv file: {e}')
            return None      


    def topic_modeling(self): 
        for file_name in os.listdir('csv_files'):
            print(f'{os.path.join(file_name)} <- processing')
            file_path = f'csv_files/{file_name}'
            df = pd.read_csv(file_path, encoding='utf-8')
            all_text = ' '.join(df.stack().tolist())
        
            doc = self.nlp(all_text)
            data = [{'id': 0, 'text': [token.text for token in doc]}]

            # Now create a dictionary from the texts
            # Note that we wrap data[0]['text'] in another list to create an iterable of documents
            dictionary = corpora.Dictionary([data[0]['text']])

            # Convert the text to a corpus that we can use for the topic modeling
            # Note that we pass the entire document as a single list to doc2bow
            corpus = [dictionary.doc2bow(data[0]['text'])]

            # Create the LDA model
            lda_model = models.LdaModel(corpus=corpus, num_topics=1, id2word=dictionary)

            # Print the topics
            for idx, topic in lda_model.print_topics(-1):
                print('Words: {}'.format(topic))

class messageProcessor(Processor):

    def __init__(self):
        super().__init__()


    def speech_recognizing():
        r = sr.Recognizer()
        # capturing microphone input
        # change the device_index if needed 
        with sr.Microphone(device_index=2) as source:
            r.adjust_for_ambient_noise(source)
            print("Speak Anything :")
            audio = r.listen(source)

        # recognize speech using Google Speech Recognition
        try:
            '''
            message = r.recognize_houndify(audio, client_id="Pz38Yvr01KU7avT1L_lEGw==" client_key="Bb4N1XOzeSTeXZ-lYS4UfZewwIKgCsr032WT_KNKgF4EJxeXldBy-IazTRQ5sa6zWCGAo1CMV4PxCkfr3jAF-w==")
            '''
            # change language if needed
            return r.recognize_google(audio, language='en-EN')
        except:
            print("There were a problem with recognizing speech")
            print("Try changing the microphone or language")
            return "Could not recognize what you've said"



    def semantic_similarity(self, initial_text, final_text):
        try:
            initial_text = self.nlp(initial_text)
            final_text = self.nlp(final_text)
            similarity = initial_text.similarity(final_text)
            print(similarity)
            return(similarity)
        except Exception as e:
            print(f'An error occured while calculating semantic similarity: {e}')


    def emotions_evaluation(text):
        # evaluate emotions in the text
        blob = TextBlob(text)
        sentiment = blob.sentiment
        if sentiment.polarity > 0:
            print("Positive")
        elif sentiment.polarity == 0:
            print("Neutral")
        else:
            print("Negative")

            

    def filter_message_to_list(self, txt):
        try:
            # lower case
            line = self.preprocess_txt(txt)
            # tokenize with spacy
            doc = self.nlp(line)
            # remove stopwords and lemmatize in one step using spacy
            filtered_tokens = [token.lemma_ for token in doc if not token.is_stop]
            # If you want to save entities to a separate file or database
            return filtered_tokens
        except Exception as e:
            print(f'An error occured while processing the file: {e}')
            return None



    def preprocess_txt(self, txt):
        # lower case
        txt = txt.lower()
        # remove numbers
        txt = txt.translate(self.translator)
        # remove punctuations
        txt = txt.translate(self.translator_punct)
        return txt
