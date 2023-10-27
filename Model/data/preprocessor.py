from customData import CustomDataset
def wordPreproccessor(word):
    ascii_values = sum(ord(char) for char in word)
    print(f"ASCII values for '{word}': {ascii_values}")


