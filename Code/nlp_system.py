
#https://www.gutenberg.org/ebooks/search/?query=instruct&submit_search=Go%21
from genericpath import exists
import numpy
import scipy
import os
from urllib import request
import string
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.util import ngrams
from joblib import dump, load
# import spacy
# from spacy import displacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('names')
nltk.download('vader_lexicon')

runChatbotFlag = True
printStats = True

# from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import random
from datetime import date
from nltk.corpus import names

#nltk.download_gui ()

import re
#  Intent matching 
# Intent matching is a necessary part of any system which aims to do more than one task. It consists in 
# taking in a user input and predicting the intent of the user, i.e., which function of the system the user 
# is intending to use. Such functions could be storing/showing the user’s name, outputting the current 
# date and time, starting a transaction dialogue, or anything else described in this document. Intent 
# detection can be done at different degrees of complexity, from simple text matching to similarity based and machine learning-based approaches.

# • Identity management 
# Identity management refers to being able to acquire, memorise, and use the name of the person 
# currently using the system. It consists in functions such as detecting name statements (e.g., “my name
# is X”, “call me X”, etc.) and storing the name to be used either in basic statements addressing the user, 
# or in explicit name output functions (e.g., “what is my name?”).

# • Small talk 
# A small talk function refers to the ability to have short discussions about shallow topics, such as “hello”, 
# “how are you”, “how’s the weather”, etc. There is no expectation of long conversations due to them 
# being one-off topics, but they can either be single turn (question, answer) or multi-turn (question, 
# answer, and one or more follow-ups). 

# • Transactions 
# A transactional dialogue system is a tool that allows for an artificial system to dialogue with a human 
# operator in order to achieve a specific set of goals, which are transactional in nature, such as
# booking a train/plane ticket, a table in a restaurant, ordering an item from a store, etc.
# Here is an example of a turn-by-turn dialogue between an operator and a travel booking system, to 
# illustrate the range of tasks that such system would need to accomplish to properly function.

# Turn Author Text
# 1 Bot Hello, welcome to the Skynet Travel Agency, how may I help?
# 2 User I would like to book a flight from London to Paris
# 3 Bot Would that be a return trip or a single flight?
# 4 User A return trip please
# 5 Bot When were you thinking of going?
# 6 User From the 10th to the 19th?
# 7 Bot I’m sorry, I am going to need more details than that – please use the precise 
# dates of your trip and whether you are flexible in your dates
# 8 User 10/11/2021 to 19/11/2021, and I am flexible
# 9 Bot Thank you, I will search my database for available tickets. Please wait.
# 10 Bot I have one flight available on the 10th from London to Paris, but no flight from 
# Paris to London on the 19th, would you like to change the return date or book a 
# single flight?
# 11 User Change the date to the 20th

# • Information retrieval & question answering 
# A conversational information retrieval system is an information retrieval tool that interprets a query
# from a dialogue with a user and returns a set of one or more results which are deemed to be relevant, 
# such as direct answers, webpages, books, recipes, songs, movies, etc. It requires having some form 
# of database of documents and an indexing system. A question answering system can be seen as a 
# special case of information retrieval where the documents are question-answer pairs, and only the 
# answer is retrieved.



#Basic Plot:
#1. meet a crow who asks for name (Identity management)
#2. Illusion of choice (Intent matching)
#3. See a cabin and descriptions (Information retrieval & question answering)
#4. talk with witch inside (Small talk)
#5. barter for a magical item (Transactions)
tokenizer = nltk.RegexpTokenizer(r"\w+")

def calculateCosineSimularity(x, y, ignoreStopwordsFlag = False, stemArguments = False):
    x = x.lower()
    y = y.lower()
        
    # tokenization
    X_list = tokenizer.tokenize(x) 
    Y_list = tokenizer.tokenize(y)
    
    if stemArguments:
        # p_stemmer = PorterStemmer()
        p_stemmer = SnowballStemmer('english')
        
        
        newX = []
        newY = []
        #print("Quack")
        for token in X_list:
            newX.append(p_stemmer.stem(token))
            
            # print(p_stemmer.stem(token))
            # print(sb_stemmer.stem(token))
            # print('---')
            
        for token in Y_list:
            newY.append(p_stemmer.stem(token))
            
            # print(p_stemmer.stem(token))
            # print(sb_stemmer.stem(token))
            # print('---')
            
        X_list = newX
        Y_list = newY
            
        
    
    l1 =[];l2 =[]
    
    if not ignoreStopwordsFlag:
        # sw contains the list of stopwords
        english_stopwords = stopwords.words('english')
        
        # remove stop words from the string
        X_set = {word for word in X_list if word not in english_stopwords} 
        Y_set = {word for word in Y_list if word not in english_stopwords}
    else:
        X_set = {word for word in X_list}
        Y_set = {word for word in Y_list}
        
    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: 
            l1.append(1) # create a vector
        else: 
            l1.append(0)
        if w in Y_set: 
            l2.append(1)
        else: 
            l2.append(0)
    c = 0
    
    # cosine formula 
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float(numpy.sqrt(sum(l1)*sum(l2)))
    # print("similarity: ", cosine)
    
    return cosine
    
def generatePotion():
    
    potionData = loadpotionsDictionary()
    newPotion =  {}

    for potion in potionData:
        randomOptionindex = random.randint(0, len(potion['options']) - 1)
        
        newPotion[potion['characteristic'].lower() ] = {
            'descriptor': potion['descriptor'],
            'option': potion['options'][randomOptionindex]
        }
        
        
    return newPotion

def generateUserPotion(userInput = 'vial of everlasting life'):
    #potionData = all potions data seporated out
    
    # import time
    # start_time = time.time()
    
    potionData = loadpotionsDictionary()
    newPotion =  {}

    # lemmentisation - no change growth --> growth. grow --> grow
    for potion in potionData:
        options = []
        for option in potion['options']:
            options.append(
                    {
                        'option': option, 
                        'score': calculateCosineSimularity(option, userInput, False, True)
                    }
                ) 
            
        options.sort(key=scoreSorter, reverse=True)
        
        if options[0]['score'] == 0.0:
            generatedOption = potion['options'][random.randint(0, len(potion['options']) - 1)] 
        else:
            generatedOption = options[0]['option']
        
        newPotion[potion['characteristic'].lower()] = {
            'descriptor': potion['descriptor'],
            'option': generatedOption
        }
        
    # print("--- %s seconds ---" % (time.time() - start_time))
    return newPotion
        

def main():
    N_PARAM = 2 
    

    # corpus_name = 'Potions.joblib'
    # if exists(corpus_name):
    #     corpus = load(corpus_name)
    # else:
    #     # corpus = loadCorpus(r'Coursework 1\Code\Data\potion-data-raw.html', 'Potions')
    #     corpus = loadpotionsCorpus()
    #     #print(corpus)
    
    # corpus = loadpotionsDictionary()
    

    # #data = loadBookResource("http://www.gutenberg.org/files/84/84-0.txt")
    # tokenized_text = tokenizeCorpus(corpus)
    # print(tokenized_text)
    
    # a = []
    # a = tokenized_text.split('d')
    
    # z = 0
    
    
    #intent matching
    #Should be able to handle:
    #"My Name is John"
    #"John is my name"
    #"John"
    #"My name is John Smith."
    
    #1. As is:
    # print(f"You are a hunter by the name of:")
    # name = input('> ')
    
    #     CC: It is the conjunction of coordinating
    # CD: It is a digit of cardinal
    # DT: It is the determiner
    # EX: Existential
    # FW: It is a foreign word
    # IN: Preposition and conjunction
    # JJ: Adjective
    # JJR and JJS: Adjective and superlative
    # LS: List marker
    # MD: Modal
    # NN: Singular noun
    # NNS, NNP, NNPS: Proper and plural noun
    # PDT: Predeterminer
    # WRB: Adverb of wh
    # WP$: Possessive wh
    # WP: Pronoun of wh
    # WDT: Determiner of wp
    # VBZ: Verb
    # VBP, VBN, VBG, VBD, VB: Forms of verbs
    # UH: Interjection
    # TO: To go
    # RP: Particle
    # RBS, RB, RBR: Adverb
    # PRP, PRP$: Pronoun personal and professional


    #region 3)
    # Crawling a website:
        # from urllib import request
        # # url = "http://example.org"
        # url = "https://www.reddit.com/r/DnDBehindTheScreen/comments/4btnkc/random_potions_table/"
        # raw = request.urlopen(url).read().decode('utf8')
        # print(raw) # html code of the page
    #endregion


    # python -m spacy download en
    #NER = spacy.load("en_core_web_sm")
    # qa_model = pipeline("question-answering")


    runTask1 = True
    runTask2 = True
    runTask3 = False
    runTask4 = True
    runTask5 = True
    
    name = 'Adventurer'

    #print(f"Name:")
    placename = ['North']
    huntingName = ['rabbits']
    #region Task 1:
    #https://huggingface.co/dslim/bert-base-NER?text=My+name+is+Sarah+and+I+live+in+London
    if (runTask1):
        print(f"Welcome to RPGAI!")
        print(f"You were travelling the {placename[0]} Roads, while you were hunting {huntingName[0]} you mistakenly stepped onto a consealed pit fall trap and dropped into a deep dark cave.")
        print(f"You awaken some time later in a pile of bones, to the sound of a shrill voice calling out to you. A crow sits on a protruding femur and calls out:")
        print(f"\"You're lucky to survive that fall... SQUAWK!... Who are you?... SQUAWK!\".")
        
        
        
        
        #check me
        #version 1
        #N.B. ner not allowed
        nameText = input('> ').replace('-',' ').title()
        # stuff = tokenizeText(nameText)
        
        english_stopwords = stopwords.words('english')
        namesData = {word for word in nameText.split() if word not in english_stopwords}
        
        
        acceptedNames = []
        for word in namesData:
            if word in names.words():
                acceptedNames.append(word)
                
        
        if len(acceptedNames) > 0:
            name = " ".join(acceptedNames) 
        
        a = 0
        
        #version 3 
        #Nltk names
        # 
            
        # newText = []
        # partsOfSpeechTags = []
        # for sentance in nameInputs:
        #     newText.append([])
        #     partsOfSpeechTags.append([])
        #     for word in sentance.split():
        #         if word not in english_stopwords:
        #             newText[len(newText) - 1 ].append(word)
        #             partsOfSpeechTags[len(newText) - 1 ] = nltk.pos_tag(newText[len(newText) - 1 ])
        
        
        
        
        
        
        
        
        # text_tokenized = [word_tokenize(sentence) for sentence in stuff]
        # text_padded = [list(sentence_tokenized)
        #             for sentence_tokenized in text_tokenized]
        # flat_text_padded = list(flatten(text_padded))
        # partsOfSpeechTags = nltk.pos_tag(flat_text_padded)
        
        # if len(partsOfSpeechTags) > 1:
        #     name = (' '.join([x for (x,y) in partsOfSpeechTags if y in ('JJ', 'NN')])).title() #extract adjectives...
        # else:
        #     name = (' '.join([x for (x,y) in partsOfSpeechTags if y in ('NN')])).title() 
            
        # if len(name.split(" ")) == 1:
        #     name = 'Adventurer'

        #version 2
        # Named Entity Recognition (NER) 
        # nameText = input('> ')
        # text1 = NER(nameText)
        # for word in text1.ents:
        #     print(word.text,word.label_)
        #     name = word.text.title()
        
        ##Check for dodgy names and call the player adventurer instead?
        
        
        print(f"\"So your name is '{name}'... SQUAWK!... Best hurry, my Master grows bored of waiting... SQUAWK!\" - as the crow takes flight.")  
    #endregion
    
    #region Task 2:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (runTask2):
        #IMPROVEMENTS: Capture any adjectives they used in close proximity to door or left? E.G. I open the door slowly --> You open the dark wood door 'slowly'
        #cannot capture 'I ignore the tunnel and instead take the door'

        print(f"You stand as see three exits from this room. A door left, a ladder downwards or a tunnel right. You pick the...:")
        decision = input('> ')
        
        vocab1 = 'door left'
        vocab2 = 'ladder downwards'
        vocab3 = 'tunnel right'
        
        simularityScore_left = calculateCosineSimularity(vocab1, decision)
        simularityScore_down = calculateCosineSimularity(vocab2, decision)
        simularityScore_right = calculateCosineSimularity(vocab3, decision)
            
        if simularityScore_left > simularityScore_down and simularityScore_left > simularityScore_right:
            print(f"You open the dark wood door and enter into a small barracks. There are a number of cots, tables and leftover food in the room. Exitting out the back door, you walk down a hedge lined road until you see it...")

        elif simularityScore_right > simularityScore_left and simularityScore_right > simularityScore_down:
            print(f"You move briskly down the tunnel, but before long you spot a flickering light ahead of you. As you move towards it you see that the light is coming from it...")

        elif simularityScore_down > simularityScore_left and simularityScore_down > simularityScore_right:
            print(f"You climb down the ladder into knee deep water, which flows slowly meandering down to the right.")
            print(f"Travelling down the river path, you eventually come across a stone carved staircase built into one of the walls, while the river takes a plunge into the depths below. You climb up these stairs and enter into a gardens a short distance from it...")
            
        else:
            print(f"Umm... Im confused which one you picked... Try again")
            
            
        print(f"A cottage built on a hill in the cave.")
        print(f"The cottage has a path leading through a front garden to the door. The structure is three stories tall with several windows but a single front door.")
    
    #endregion
    
    
    #region Task 3:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #https://github.com/huggingface/transformers
    # if (runTask3):
    #     #https://www.kdnuggets.com/2020/04/simple-question-answering-systems-text-similarity-python.html
    #     context = """
    #     The cottage has a front garden of multicoloured flowers in every shape and size, a large left-leaning tree sparsely covered in leaves and a small path leading to the front door. The path to the door displays wealth with marble columns standing firm along it and with a pair of fountains barely dripping water into multilayered basins. The path leads to a short staircase of blue stone blocks before reaching the door. 
        
    #     The cottage is three stories tall and has a loft conversion. The walls are made of white-washed timber. The roof is red slate and covered in lichen and moss, with vines crawling up the walls to reach it. The single chimney is leaning precariously. The structure has 5 windows on the ground floor and 2 in the roof. The door is a large ashen double door which looks weathered. When you look at the door it slowly opens.
        
    #     The building looks old, but not abandoned. The cottage appears occupied, two windows are lit and the smoke stack is releasing occassional puffs of grey smoke."""
        
    #     sentences = context.split('.')
        
    #     tokenizedSentences = [word_tokenize(sentence) for sentence in sentences]
    #     posSentences = [nltk.pos_tag(tokenizedSentence) for tokenizedSentence in tokenizedSentences]
        
    #     #https://www.youtube.com/watch?v=DkY_RZrOoqY - 49:00 onwards
    #     # grammar = "NP: {<DT>?<JJ>*<NN>}"
    #     # grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    #     # grammar = "NP: {<DT|PP\$>?<JJ>*<NN>}"
    #     # grammar = "NP: {<NNP>+}"
    #     # grammar = r"""
    #     # NP: 
    #     #     {<.*>+}
    #     #     }<VBD|IN>+{
    #     # """
    #     grammar = r"NP: {<[CDJNP].*>+}"
    #     # text = '''
    #     # he PRP B-NP
    #     # accepted VBD B-VP
    #     # the DT B-NP
    #     # position NN I-NP
    #     # of IN B-PP
    #     # vice NNP B-NP
    #     # lord NNP I-NP
    #     # of IN B-PP
    #     # cheese NNP B-NP
    #     # '''
    #     # from nltk import conllstr2tree
        
    #     # nltk.chunk.conllstr2tree(text, ['NP']).draw()
        
    #     # grammar = r"""
    #     #                 VP: {<ADJ_SIM><V_PRS>}
    #     #                 VP: {<ADJ_INO><V.*>}
    #     #                 VP: {<V_PRS><N_SING><V_SUB>}
    #     #                 NP: {<N_SING><ADJ.*><N_SING>}
    #     #                 NP: {<N.*><PRO>}
    #     #                 VP: {<N_SING><V_.*>}
    #     #                 VP: {<V.*>+}
    #     #                 NP: {<ADJ.*>?<N.*>+ <ADJ.*>?}
    #     #                 DNP: {<DET><NP>}
    #     #                 PP: {<ADJ_CMPR><P>}
    #     #                 PP: {<ADJ_SIM><P>}
    #     #                 PP: {<P><N_SING>}
    #     #                 PP: {<P>*}
    #     #                 DDNP: {<NP><DNP>}
    #     #                 NPP: {<PP><NP>+}
    #     #                 """
        
    #     cp = nltk.RegexpParser(grammar)
        
    #     # from nltk import conll2000
    #     # unigramChunker = UnigramChunker()
        
    #     result = [cp.parse(posSentence) for posSentence in posSentences]
    #     #result[4].draw()
        
    #     #---------------
        
        
    #     lm = createLanguageModel(N_PARAM, context)
    #     question = input('> ')
    #     perp = calculatePerplexity(question, lm, 2)


        #---------------
        

        # print(f"The structure has many unique features, but the question most on your mind about it is...")
        
        # while True:
        #     question = input('> ')
            
        #     if question == 'done':
        #         break
            
        #     if not question.endswith('?'):
        #         question += '?'
            

            
        #     # answer = qa_model(question = question, context = context)
             
            
        #     # if answer['score'] <= 0.1:
        #     #     #if certainty is low assume that there is no effective answer
        #     #     print(f"Unfortunately, from this distance you are unable to determine that.")
        #     # else:
        #     #      #change this to say 'you see that...' or 'The...'
        #     #     print(f"{answer['answer']}")    
        

        #     print(f"If you have no more questions enter 'done' to open the cottage door, or keep asking:")
        
        
    #endregion
    
    #region Task 4:-------------
    witchName = "The Witch"
    if (runTask4):
        #4. talk with witch inside (Small talk)
        # random.randint(0,1)
        crowName = 'Fluff'
        
        print(f"You push open the creaky double doorway and enter into an open-plan room. A large rug covers much of the floor and upon it sits a large oak table strewn with magical items of many shapes and sizes. A grotesque old woman - large boils adorning her nose and one eye twice the size as the other - lounges in an armchair infront of a roaring fire on the far side of the room.")
        print(f"As you enter, you spot a large crow perched on the top of the armchair which Squawks loudly.")
        print(f"The woman announces \"I'm so glad you could finally arrive, uh... um...\", as the crow abruptly screeches \"{name}\".")
        print(f"You ask: ")

        #'who are you', 'what are you', 'where are we', 'when are we', 'why are you doing this'
        # {'question': '-', 'score': -1.0}
        vocabList = [
            {'question': 'who are you', 'score': -1.0},
            {'question': 'what are you', 'score': -1.0},
            {'question': 'where are we', 'score': -1.0},
            {'question': 'when are we', 'score': -1.0},
            {'question': 'why are you doing this', 'score': -1.0},
            ]
        
        
        while True:
            question = input('> ')
            if question == 'done':
                break
            
            for i in range(len(vocabList)):
                vocabList[i]['score'] = calculateCosineSimularity(vocabList[i]['question'], question, True)
                
            vocabList.sort(key=scoreSorter, reverse=True)
            
            if vocabList[0]['question'] == 'who are you':
                print(f"\"I am Ubaba the Witch of course! Muhahahaha!\"")
                print(f"... Oh, and that's {crowName}.")
                witchName = 'Ubaba'
                
                vocabList.append({'question': 'what is a witch', 'score': -1.0})                
                vocabList.append({'question': f'who is {crowName}', 'score': -1.0})
                
            if vocabList[0]['question'] == 'what are you':
                print(f"\"What? I am what they call a witch - and might I say is rather beautiful one at that... Wouldn't you agree?\"")
                
                vocabList.append({'question': 'what is a witch', 'score': -1.0})
                
                sentimentStatement = input('> ')
                sid = SentimentIntensityAnalyzer()
                sentiment = sid.polarity_scores(sentimentStatement)
                
                if sentiment['compound'] > 0:
                    print(f"\"I'm glad you agree.\". She winks.")
                if sentiment['compound'] <= 0:
                    print(f"\"Maybe I should just turn you into a toad... That would shut you up.\"")
                    
            if vocabList[0]['question'] == 'where are we':
                print(f"\"We are in my quaint little cottage. I teleport it when one location gets... samey\"")
                
                vocabList.append({'question': 'You can teleport', 'score': -1.0})
            if vocabList[0]['question'] == 'when are we':

                print(f"\"Thats an odd question. A day later than it was yesterday. How's that?\"")
                
                vocabList.append({'question': 'what is the date', 'score': -1.0})
                
            if vocabList[0]['question'] == 'why are you doing this':
                print(f"\"I am here to provide aid of course... for a price.\"")
                
                vocabList.append({'question': 'what aid', 'score': -1.0})
                vocabList.append({'question': 'what price', 'score': -1.0})   
            
            #appended questions added from the small talk options above:
            if vocabList[0]['question'] == 'what is a witch':
                print(f"\"A beautiful woman, a spellcaster and enchantress, who in this case who is willing to trade some of their wares...\"")
                vocabList.append({'question': 'wares', 'score': -1.0})
                vocabList.append({'question': 'trade', 'score': -1.0})
                
                
            if vocabList[0]['question'] == f'who is {crowName}':
                print(f"\"Surely you've heard my familiar... He likes to Squawk\"")
                print(f"*{crowName} the Crow screeches \"SQUAWK\" loudly*")
            if vocabList[0]['question'] == 'You can teleport':
                print(f"\"Of course I can. It's not the only thing I can do. I also sell many magical wares\"")
                
                vocabList.append({'question': 'wares', 'score': -1.0})
                vocabList.append({'question': 'trade', 'score': -1.0})
                
            if vocabList[0]['question'] == 'what is the date':
                currentDate = date.today().strftime("it is the %d of %B %Y I believe.")
                print(f"\"'{name}', you are a weird one... Uhh, it's {currentDate}.\"")
            if vocabList[0]['question'] == 'what aid':
                print(f"\"I have many magical wares for sell!\"")
                
                vocabList.append({'question': 'wares', 'score': -1.0})
                vocabList.append({'question': 'trade', 'score': -1.0})
                
            if vocabList[0]['question'] == 'what price':
                print(f"\"A small payment, but we will get to all that momentarily.\"")
                
                vocabList.append({'question': 'wares', 'score': -1.0})
                vocabList.append({'question': 'trade', 'score': -1.0})
                
            if vocabList[0]['question'] == 'wares' or vocabList[0]['question'] == 'trade':
                print(f"\"Ah, I thought that might interest you...\"")
                break
                
                

             
        
        # greeting, weather, function, occupation, who what where when why
        
        
        
    #endregion
    
    #region Task 5
    # 5. barter for a magical item (Transactions)
    if (runTask5):
        random.randint(0,1)
        notDone = True
        
        # potion1 = generatePotion()
        # potion2 = generatePotion()
        
        
        # potion1Description = f"{potion1['container']['option'][0:-1]} containing a {potion1['appearance 1']['option'][0:-1]} liquid with {potion1['appearance 2']['option']}"
        # potion2Description = f"Middle: {potion2['container']['option'][0:-1]} containing a {potion2['appearance 1']['option'][0:-1]} liquid with {potion2['appearance 2']['option']}"
        
        print(f"The table in front of {witchName} suddenly becomes covered in smoke, which slowly dissipates to reveal potions, a set of alchemist's supplies and a bundle of ingredients.")
        print(f"\"To create a potion for you '{name}', I first will need inspiration. Tell me what you want and I'll see what I can conjure!\".")
        #print(f"{witchName} cackles \"I have two potions I made earlier here, or you can help me create a new one.\", she gestures a {potion1Description}, then a {potion2Description}, then the alchemy set.")

        while notDone:
            userInput = input('> ')
                        
            potionUser = generateUserPotion(userInput)
            potionUserDescription = f"{potionUser['container']['option'][0:-1]} containing a {potionUser['appearance-1']['option'][0:-1]} liquid with {potionUser['appearance-2']['option']}"
            print(f"{witchName} stirs a cauldron spewing a sickly stench. She adds ingredients; a raven's egg, the blood of a hen, eyeballs of a crocodile. After a pungent explosion forming a purple cloud she states \"Ahh, finished\"")
            print(f"She holds out a {potionUserDescription} \"This potion's effect is {potionUser['effect']['option'][0:-1]}... Is that what you wanted? A simple 'yes' would do.\"")
            
            userInput = input('> ')
            if (userInput == "yes"):
                notDone = False
            else:
                print(f"\"I'll take that as a no then... typical. Tell me again:\"")
                
        print(f"\"Now, let us cover the price...\"")
        costString = 'cost price how much worth trade'
        haggleString = 'haggle negotiate yes ok agreed'
        expectedLetterCount = 2
        notDone = True
        while notDone:
            userInput = input('> ')
            
            if calculateCosineSimularity(costString, userInput, True) > 0.0:
                print(f"\"The price will be... TWO LETTERS OF YOUR NAME!!! Wuhahahahahaha\"")
                print(f"\"I am of course willing to negotiate... \"")
                
                userInput = input('> ')
                if calculateCosineSimularity(haggleString, userInput, True) > 0.0:
                    #haggling
                    print(f"\"In that case, why don't you remind me how beautiful I am!?\"")
                    
                    sentimentStatement = input('> ')
                    sid = SentimentIntensityAnalyzer()
                    sentiment = sid.polarity_scores(sentimentStatement)
                    
                    if sentiment['compound'] > 0.3:
                        print(f"\"How I love to hear that.\". She winks. \"Alright, only ONE letter instead!\"")
                        expectedLetterCount = 1
                    elif sentiment['compound'] < 0.3 and sentiment['compound'] > -0.3:
                        print(f"\"Hmm. Not good enough. TWO letters still!\"")
                    elif sentiment['compound'] <= -0.3:
                        expectedLetterCount = 3
                        print(f"\"Not funny. I think I want THREE letters now.\"")
                
                
                print(f"\"Time to choose, which will you pick to lose forever! Muhahahaha!\"") 
                
                while notDone:
                    userInput = input('> ')
                    
                    chosenCharacters = re.split('[^a-zA-Z]', userInput)
                    
                    characters = [c for c in chosenCharacters if len(c) == 1] 
                    
                    allCharactersValid = True

                    
                    if len(characters) != expectedLetterCount:
                        print(f"\"Not the right number of letters, try again:\"") 
                    elif CharactersInvalid(characters, name):
                        print(f"\"Im afaid they must be letters actually IN your name. Try again:\"") 
                    else:
                        for character in characters:
                            name = name.replace(character, '')
                        print(f"\"Perfect. I thank you. But alas our time together draws to an end. Goodbye '{name}'\"") 
                        notDone = False
                
                    
        print(f"Abruptly, you hear a thunderclap, as you teleport to the end of the {placename} Roads, your potion in hand.")        
        print(f"You feel a moment of bravado and announce \"I am {name}, and I'm ready for the next adventure!\" ")    
        print(f"        To be continued...")  
            
    #endregion

def CharactersInvalid(charList, name):
    for character in charList:
        if character not in name:
            return True
    return False

def scoreSorter(e):
    return e['score']

def loadBookResource(url):
    # Get Frankenstein eBook
    book = request.urlopen(url).read().decode('utf8', errors='ignore')

    # Get first chapter, Beautiful Soup would have been useful to remove all the tags.
    text = book.split("Chapter 1\r\n\r\n")[1].split("Chapter 2\r\n\r\n")[0]
    
    return text

def createLanguageModel(n_parameter, data):  
    # Remove punctuation
    sentences = tokenizeText(data)
    # sentences = lowerData.split('.')

    text_tokenized = [word_tokenize(sentence) for sentence in sentences]
    text_padded = [list(sentence_tokenized) for sentence_tokenized in text_tokenized]
    # text_padded = [list(pad_both_ends(sentence_tokenized, n=n_parameter)) for sentence_tokenized in text_tokenized]

    if False:
        print(f"Number of sentences: {len(data)}")
        n_tokens = 0
        for sentence in text_tokenized:
            n_tokens += len(sentence)
        print(f"Number of tokens: {n_tokens}")
        print(f"Average number of tokens per sentence: {n_tokens/len(data)}")

        flat_text_padded = list(flatten(text_padded))
        unigrams = flat_text_padded
        bigrams = list(ngrams(flat_text_padded, 2))
        trigrams = list(ngrams(flat_text_padded, 3))
        
        print(f"Most frequent unigrams:\n{nltk.FreqDist(unigrams).most_common(10)}")
        print(f"Most frequent bigrams:\n{nltk.FreqDist(bigrams).most_common(10)}")
        print(f"Most frequent trigrams:\n{nltk.FreqDist(trigrams).most_common(10)}")
        
        #All unique words
        multiples = flat_text_padded
        singles = list(set(flat_text_padded))

        #all parts of speech
        print(nltk.pos_tag(multiples))      
        print()

        #extract all nouns
        is_noun = lambda pos: pos[:2] == 'NN'
        nouns_singles = [word for (word, pos) in nltk.pos_tag(singles) if is_noun(pos)]
        nouns_multiples = [word for (word, pos) in nltk.pos_tag(singles) if is_noun(pos)]
        print(f'All Nouns = {nouns_singles}')
        print(f'Most Common Nouns = {nltk.FreqDist(nouns_multiples).most_common(10)}')

       
        text = nltk.Text(word for word in multiples)
        a = text
        print(a)
        # print(f'text...! = {a.similar("dragon")}')

    # Create language model:
    # Language model
    corpus, vocab = padded_everygram_pipeline(n_parameter, text_padded)
    # lm = MLE(N_PARAM)  # non-smoothed
    languageModel = Laplace(n_parameter)  # smoothed
    languageModel.fit(corpus, vocab)
    print(list(languageModel.vocab))
    return languageModel
    
def tokenizeText(text):
    text_string = text.lower()

    # Remove punctuation
    string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
    string.punctuation = string.punctuation.replace('.', '')  # keep "." so that can split sentences with NLTK
    text_filtered = "".join([char for char in text_string if char not in string.punctuation])
    text_sentences = sent_tokenize(text_filtered)
    return text_sentences
    
def tokenizeCorpus(corpus):
    string.punctuation = string.punctuation + '“' + '”' + '’' + '‘' + '—'
    string.punctuation = string.punctuation.replace('.', '') # keep "." so that can split sentences with NLTK
    # string.punctuation = string.punctuation.replace('<s>', '')  # keep "." so that can split sentences with NLTK
    # string.punctuation = string.punctuation.replace('</s>', '')  # keep "." so that can split sentences with NLTK
    text_filtered = ""
    s = ""
    for value in list(corpus.values()):
        s =  value.lower()
        text_string = s.replace('-', ' ')

        # Remove punctuation

        text_filtered += "".join([char for char in text_string if char not in string.punctuation])
    text_sentences = sent_tokenize(text_filtered)
    return text_sentences

def calculatePerplexity(text, languageModel, N_PARAM):
    # **Lower perplexity = more chance of plagiarism**
    # **Higher perplexity = more chance of having generated text**
    # We can do this whole process again with the MLE language model (`lm = MLE(N_PARAM)` instead of `lm = Laplace(N_PARAM)`) to see if a non-smoothed language model produces different perplexity results.
    text_tokenized = nltk.word_tokenize(text)
    text_padded = list(pad_both_ends(text_tokenized, n=N_PARAM))
    text_ngrams = list(ngrams(text_padded, N_PARAM))
    return languageModel.perplexity(text_ngrams)
    #print(text_ngrams)
    #print(f"Perplexity: {languageModel.perplexity(text_ngrams)}")

def loadpotionsCorpus():
    corpus = {}
    fp = r'Coursework 1\Code\Data\potion-data-raw.html'
    with open(fp, encoding='utf8', errors='ignored', mode='r') as document:
        contents = content = BeautifulSoup(document.read(), 'html.parser').get_text() 
        corpus['potion-data-raw'] = contents
    dump(corpus, fr'Coursework 1\Code\Potions.joblib')
    return corpus

def loadpotionsDictionary():
    fp = r'Data\potion-data-processed.txt'
    with open(fp, encoding='utf8', errors='ignored', mode='r') as document:
        text = document.read()
        sections = re.split('d\d+ ', text)[1:]
        
        d = []
        
        for section in sections:
            characteristic = section.split('. ')[0]
            descriptor = re.search('. (.*),', section).group(1)
            options = section.split('\n')[1:-1]

            d.append(
                {
                    'characteristic': characteristic,
                    'descriptor': descriptor,
                    'options': options
                }
            )
        
        return d
       
def loadCorpus(document_path, corpus_name):
    corpus = {}
    for file in os.listdir(document_path):
        fp = document_path + os.sep + file
        with open(fp, encoding='utf8', errors='ignored', mode='r') as document:
            contents = document.read()
            document_id = file
            corpus[document_id] = contents
    dump(corpus, fr'Coursework 1\Code\{corpus_name}.joblib')
    return corpus


if __name__ == "__main__":
    main()
    
    