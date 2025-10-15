import re

moderate_content = "This IS TEST FILE NOW WITH long"

characters_in_string = 0
capitalized_characters = 0
for c in moderate_content:
    if c.isalpha():
        characters_in_string = characters_in_string + 1
        if c.isupper():
            capitalized_characters = capitalized_characters + 1
        else:
            pass
    else:
        pass

capitalize_ratio = capitalized_characters / characters_in_string

print(capitalize_ratio)

if characters_in_string > 15:
    if capitalize_ratio > 0.7:
        print("too much")
    else:
        print("okay, long")
else:
    print("okay, short")
