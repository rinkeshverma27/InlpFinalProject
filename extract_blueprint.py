import zipfile
import re

try:
    with zipfile.ZipFile('master_blueprint.docx') as z:
        xml_content = z.read('word/document.xml').decode('utf-8')
        # Remove all xml tags
        text = re.sub('<[^<]+>', '', xml_content)
        # Clean up multiple spaces
        text = re.sub(' +', ' ', text)
        with open('blueprint_text.txt', 'w') as f:
            f.write(text)
except Exception as e:
    with open('blueprint_text.txt', 'w') as f:
        f.write(str(e))
