from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


MODEL_PATH = ""
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)




def analyze_relation_sentiment(relation_name):
    inputs = tokenizer(
        relation_name,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=-1).tolist()[0]

    label_mapping = {0: -1, 1: -1, 2: 0, 3: 1, 4: 1}
    emotion_label = label_mapping[pred_label]
    confidence = round(max(probabilities) * 100, 2)

    return emotion_label, confidence



def read_rel_file():
    rel_file_path = ""
    rel_id2name = {}
    try:
        with open(rel_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    rel_id = int(parts[0])
                    rel_uri = " ".join(parts[1:])
                    rel_name = rel_uri.split("/")[-1]
                    rel_id2name[rel_id] = rel_name
        print(f"Successfully read {len(rel_id2name)} relations\n")
        return rel_id2name
    except FileNotFoundError:
        print(f"Error: Relation file '{rel_file_path}' not found")
        exit()
    except UnicodeDecodeError:
        print(f"Error: Failed to decode relation file '{rel_file_path}'")
        exit()



if __name__ == "__main__":

    rel_id2name = read_rel_file()

    output_file = ""

    with open(output_file, "w", encoding="utf-8") as f:
        count = 0
        for rel_id, rel_name in rel_id2name.items():
            emotion_label, confidence = analyze_relation_sentiment(rel_name)

            f.write(f"{rel_id} {emotion_label} {confidence}\n")

            if count < 30:
                print(
                     f"Relation ID: {rel_id:4d} | Name: {rel_name:25s} | Emotion Label: {emotion_label:^3d} | Confidence: {confidence:5.2f}%")
            count += 1

    print(f"\nSuccessfully generated emotion mapping file: {output_file}")
    print(f"Total relations processed: {count}")