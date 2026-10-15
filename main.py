import pandas as pd

def main():
    df = pd.read_csv("services.csv")
    services = df["service_name"].tolist()
    print("Available services:")
    print()
    for number, service in enumerate(services, start = 1):
        print(f"{number}. {service}")
    user_service = input("What service are you looking for? ").lower()
    if user_service == "":
        print("No input.")
        return
    match_found = False
    for number, service in enumerate(services, start = 1):
        if user_service in service.lower():
            print(f"{number}. {service}")
            match_found = True
    if not match_found:
        print("No matches found")               
main()