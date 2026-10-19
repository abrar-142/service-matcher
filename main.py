import pandas as pd


def main():
    catalogue = pd.read_csv("services.csv")
    provider_catalogue = pd.read_csv("provider_services.csv")

    for target in catalogue["service_name"]:
        print(f"Target: {target}")
        best_score = 0
        best_candidate = None

        for candidate in provider_catalogue["service_name"]:
            score = word_overlap_score(target, candidate)
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_candidate is None:
            print("No match found.")
        else:
            print(f"Candidate: {best_candidate}")
        print()


def word_overlap_score(target, candidate):
    target_words = set(target.lower().split())
    candidate_words = set(candidate.lower().split())
    return len(target_words & candidate_words)


if __name__ == "__main__":
    main()
