from customData import CustomDataset
word = input("enter alphabet or number:")  #priyanshu ALIML algorithm
ascii_values = sum(ord(char) for char in word)

print(f"ASCII values for '{word}': {ascii_values}")
