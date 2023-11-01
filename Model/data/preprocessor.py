
def wordPreproccessor(glosses):
    
    # ascii_values = sum(ord(char) for char in word)
    # print(f"ASCII values for '{word}': {ascii_values}")
    userInput=[]
    baseValue=10
    for i in glosses:
        
        if (ord(i) >=65 and ord(i) <91 ):
            gen=ord(i)-65
            userInput.append( int(gen+baseValue) ) 
        else:
             userInput.append(int(i)) 
    return userInput
