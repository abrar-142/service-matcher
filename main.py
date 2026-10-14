def main():
    services = [
        "Diabetes Annual Review", 
        "Asthma Annual Review", 
        "Blood Pressure Check", 
        "Flu vaccination", 
        "Physiotherapy Initial Assessment"
    ]
    print("Available services:")
    print()
    for number, service in enumerate(services, start = 1):
        print(f"{number}. {service}")


main()