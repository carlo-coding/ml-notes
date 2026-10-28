
import datasets
from pathlib import Path

ds = datasets.load_dataset("jncarlo/tinymix", "1M")

text = "\n\n".join([item["text"] for item in ds["train"]])


validation_splits = ["validation_dialogue", "validation_expository", "validation_narrative"]
validation_text = "\n\n".join([item["text"] for split in validation_splits for item in ds[split]] )

print("Num of chars in train dataset: ", len(text))
print("Num of chars in validation dataset: ", len(validation_text))


chars = sorted(list(set(text+validation_text)))
# See all the chars
print(chars)

# It shows a lot of junk after certain point. So I'm just getting rid of those characters
print("\nUSEFULL: ", chars[:chars.index("£")]) # Usefull characters
chars_to_remove = chars[chars.index("£"):]
print("\nJUNK: ", chars_to_remove) # Junk

cleaned_text = "".join([char for char in text if char not in chars_to_remove])

cleaned_validation_text = "".join([char for char in validation_text if char not in chars_to_remove])

chars_cleaned = sorted(list(set(cleaned_text+cleaned_validation_text)))
# See all the chars
print("\n--- AFTER CLEANING ---")
print(chars_cleaned)


vocab_size = len(chars_cleaned)
print("vocab size: ",vocab_size)

with open(str(Path(__file__).parent)+"/data_train.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

with open(str(Path(__file__).parent)+"/data_val.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_validation_text)
