import json

def main():
    with open("outputs/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("=== Test Set Evaluation ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
