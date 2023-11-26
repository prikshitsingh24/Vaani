import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def preprocess_sentence(sentence):
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)
    print(words)
    

    # Remove punctuation
    words = [word for word in words if word.isalpha() or word.isalnum() or word.isnumeric() or  word.isdigit()]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove stop words based on their impact in the sentence
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in words if word.lower() not in stop_words or word.lower() in ['my', 'your', 'his', 'her', 'its', 'our', 'their', 'who', 'are', 'you','out']]
   
    # Split each word into characters, convert to uppercase, and join with commas
    characters_list = [word.upper() for word in meaningful_words]

    # Join the character lists with commas
    # processed_sentence = characters_list.split(',')
    print(characters_list)

    return characters_list

# Get input from the user
# user_input = input("Enter a sentence: ")
# output_result = preprocess_sentence(user_input)
# print("Processed Result:", output_result)

def wordPreproccessor(glosses):
    
    # ascii_values = sum(ord(char) for char in word)
    # print(f"ASCII values for '{word}': {ascii_values}")
    userInput=[]
   
    baseValue=10
    for i in glosses:
        for j in i:
            
            if (ord(j) >=65 and ord(j) <91 ):
                gen=ord(j)-65
                userInput.append( int(gen+baseValue) ) 
            else:
                userInput.append(int(j)) 
    return userInput
def inverseWordpreproccessor(index):
    baseValue=10
    if (index>=10):
        character=(index-baseValue)+65
    else:
        character=index
      
    return chr(character)  
        
    
    
    