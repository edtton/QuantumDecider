import pytesseract
from PIL import Image
import re
from google import genai

client = genai.Client(api_key="AIzaSyDn6WqMdfh0tojInAmpAAVf5Ms2BI-jPIM")

def extract_menu_items(image_path):
    # text = pytesseract.image_to_string(Image.open("Bitcamp2025/tlj.jpg"))
    # lines = text.split("\n")
    # items = []

    # for line in lines:
    #     line = line.strip()
    #     # Skip empty lines or prices
    #     if len(line) < 2 or re.search(r"\$\d", line):
    #         continue
    #     # Remove prices from end
    #     item = re.sub(r"\$[\d\.]+", "", line).strip()
    #     items.append(item)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Explain how AI works in a few words"
    )

    print(response.text)

    # return items

def main(): 
    extract_menu_items("tlj.jpg")
    # items = extract_menu_items("tlj.jpg")
    # print(items)

if __name__ == "__main__":
    main()

