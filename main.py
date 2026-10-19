import pandas as pd

def main():
    catalogue = pd.read_csv("services.csv")
    provider_catalogue = pd.read_csv("provider_services.csv")
    for target in catalogue["service_name"]:
        print(f"Target: {target}")
        best_score = 0
        for candidate in provider_catalogue["service_name"]:
            matches = word_overlap_score(target, candidate)
            if matches > best_score:
                best_candidate = candidate
                best_score = matches
        if best_score == 0:
            print("No match found.")
        else:
            print(f"Candidate: {best_candidate}")
        print()


def word_overlap_score(target, candidate):
    target_words = set(target.lower().split())
    candidate_words = set(candidate.lower().split())
    matches = len(target_words & candidate_words)
    return matches




main()