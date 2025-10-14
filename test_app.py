import re

moderate_content = "This is a link www.google.com that will be removed"

pattern = re.compile(r'(https?://[^\s]+|www\.[^\s]+)', re.IGNORECASE)
url_matches = pattern.findall(moderate_content)
replace_text = "[link removed]"

if url_matches:
    moderate_content = re.sub(pattern, replace_text,
                              moderate_content)
    print(moderate_content)
else:
    print(moderate_content)
