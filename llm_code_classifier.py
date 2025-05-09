import os
import openai
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------
# OpenAI GPT-4 CLASSIFIER
# ------------------------

def classify_function_with_gpt(code_snippet: str, api_key: str) -> str:
    openai.api_key = api_key
    prompt = f"""You are a security expert. Analyze the C function below and determine if it is vulnerable.

    Function:
    ```c
    {code_snippet}
    ```

    Does this function contain a known or potential vulnerability? Answer only \"Yes\" or \"No\".
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response['choices'][0]['message']['content'].strip()
    return answer

# ------------------------
# CodeBERT EMBEDDING + CLASSIFIER
# ------------------------

print("[INFO] Loading CodeBERT tokenizer and model...")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()

def embed_function_code(code_snippet: str) -> torch.Tensor:
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # CLS token

# Example usage for batch classification with CodeBERT

def train_codebert_classifier(function_codes: list, labels: list):
    print("[INFO] Embedding function codes with CodeBERT...")
    X = torch.stack([embed_function_code(code) for code in function_codes]).numpy()
    y = labels

    print("[INFO] Training logistic regression classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("[RESULTS] Classification Report:")
    print(classification_report(y_test, y_pred))

    return clf

# ------------------------
# Example Main Usage
# ------------------------
if __name__ == "__main__":
    example_code = """
    int vulnerable_function(char *input) {
        char buffer[64];
        strcpy(buffer, input);
        return 0;
    }
    """

    # GPT-4 example
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    verdict = classify_function_with_gpt(example_code, api_key)
    print("[GPT-4] Vulnerable?", verdict)

    # CodeBERT + Logistic Regression example
    dummy_functions = [example_code, "int safe_function() { return 1; }"]
    dummy_labels = [1, 0]
    train_codebert_classifier(dummy_functions, dummy_labels)
