import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    catalogue = pd.read_csv("services.csv")
    provider_catalogue = pd.read_csv("provider_services.csv")
    answer_key = pd.read_csv("answer_key.csv")

    evaluate_word_overlap(catalogue, provider_catalogue, answer_key)
    print()
    evaluate_tfidf(catalogue, provider_catalogue, answer_key)
    print()
    evaluate_embeddings(catalogue, provider_catalogue, answer_key)


def evaluate_word_overlap(catalogue, provider_catalogue, answer_key):
    correct_matches = 0

    for _, catalogue_row in catalogue.iterrows():
        target_name = catalogue_row["service_name"]
        target_id = catalogue_row["service_id"]
        print(f"{target_id}: {target_name}")
        best_score = 0
        best_candidate = None
        correct_row = answer_key[answer_key["target_id"] == target_id]
        correct_provider_id = correct_row["correct_provider_id"].iloc[0]

        for _, provider_row in provider_catalogue.iterrows():
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

    print(f"Accuracy: {correct_matches}/{len(catalogue['service_name'])}")


def word_overlap_score(target, candidate):
    target_words = set(target.lower().split())
    candidate_words = set(candidate.lower().split())
    return len(target_words & candidate_words)


def evaluate_tfidf(catalogue, provider_catalogue, answer_key):
    all_service_list = (
        catalogue["service_name"].tolist()
        + provider_catalogue["service_name"].tolist()
    )
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(all_service_list)
    num_targets = len(catalogue)
    target_matrix = matrix[:num_targets]
    candidate_matrix = matrix[num_targets:]
    similarity_matrix = cosine_similarity(target_matrix, candidate_matrix)

    correct_matches = 0
    for target_index, target_similarities in enumerate(similarity_matrix):
        target = catalogue["service_name"].iloc[target_index]
        target_id = catalogue["service_id"].iloc[target_index]
        correct_row = answer_key[answer_key["target_id"] == target_id]
        correct_provider_id = correct_row["correct_provider_id"].iloc[0]

        print(f"{target_id}: {target}")

        top_indices = np.argsort(target_similarities)[-3:][::-1]
        for rank, candidate_index in enumerate(top_indices, start=1):
            score = target_similarities[candidate_index]
            candidate = provider_catalogue.iloc[candidate_index]
            print(
                f"{rank}. {candidate['service_id']}:"
                f"{candidate['service_name']} ({score:.3f})"
            )
            if rank == 1:
                best_candidate_id = candidate["service_id"]

        print(f"Correct prediction --> {correct_provider_id}")
        if best_candidate_id == correct_provider_id:
            print("CORRECT")
            correct_matches += 1
        else:
            print("WRONG")
        print()
    print(f"Accuracy: {correct_matches}/{len(catalogue['service_name'])}")


def evaluate_embeddings(catalogue, provider_catalogue, answer_key):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    target_embeddings = model.encode(catalogue["service_name"].tolist())
    candidate_embeddings = model.encode(provider_catalogue["service_name"].tolist())
    similarity_matrix = cosine_similarity(target_embeddings, candidate_embeddings)
    
    correct_matches = 0
    for target_index, target_similarities in enumerate(similarity_matrix):
        
        target = catalogue["service_name"].iloc[target_index]
        target_id = catalogue["service_id"].loc[target_index]
        correct_provider_id = answer_key["correct_provider_id"].iloc[target_index]
        
        print(f"{target_id}: {target}")
        top_indices = np.argsort(target_similarities)[-3:][::-1]
        for rank, candidate_index in enumerate(top_indices, start=1):
            score = target_similarities[candidate_index]
            candidate = provider_catalogue.iloc[candidate_index]
            print(f"{rank}. {candidate["service_id"]}:{candidate['service_name']} ({score:.3f})")
            if rank == 1:
                best_candidate = candidate["service_name"]
                best_candidate_id = candidate["service_id"]
        print(f"Correct prediction --> {correct_provider_id}")
        if best_candidate_id == correct_provider_id:
            print("CORRECT")
            correct_matches += 1
        else:
            print("WRONG")
        print()
    print(f"Accuracy: {correct_matches}/{len(catalogue['service_name'])}")



if __name__ == "__main__":
    main()
