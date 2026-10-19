import pandas as pd


def main():
    catalogue = pd.read_csv("services.csv")
    provider_catalogue = pd.read_csv("provider_services.csv")
    answer_key = pd.read_csv("answer_key.csv")

    correct_matches = 0
    
    for catalogue_index,catalogue_row in catalogue.iterrows():
        target_name = catalogue_row["service_name"]
        target_id = catalogue_row["service_id"]
        print(f"{target_id}: {target_name}")
        best_score = 0
        best_candidate = None
        correct_row = answer_key[answer_key["target_id"] == target_id]
        correct_provider_id = correct_row["correct_provider_id"].iloc[0]
        

        for provider_index, provider_row in provider_catalogue.iterrows():
            candidate_name = provider_row["service_name"]
            score = word_overlap_score(target_name, candidate_name)
            if score > best_score:
                best_candidate = candidate_name
                best_score = score
                best_candidate_id = provider_row["service_id"]

        if best_candidate is None:
            print("No match found.")
            print(f"Correct prediction --> {correct_provider_id}")
            print("WRONG")
        else:
            print(f"Predicted --> {best_candidate_id}: {best_candidate}")
            print(f"Correct prediction --> {correct_provider_id}")
            if best_candidate_id == correct_provider_id:
                print("CORRECT")
                correct_matches += 1
            else:
                print("WRONG")
        print()

    print(f"Accuracy: {correct_matches}/{len(catalogue["service_name"])}")


def word_overlap_score(target, candidate):
    target_words = set(target.lower().split())
    candidate_words = set(candidate.lower().split())
    return len(target_words & candidate_words)


if __name__ == "__main__":
    main()
