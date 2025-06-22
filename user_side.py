# This script should "simulate" the logic of the thing using the API


import requests


if __name__ == "__main__":

    # API endpoint
    base_url = "http://127.0.0.1:8000"

    submit_query_url = f"{base_url}/submit/"
    booking_url = f"{base_url}/book/"

    user_input = input("Input: ")

    # Data to send
    payload = {"query": user_input}

    # Make the POST request
    response = requests.post(submit_query_url, json=payload)

    # Print the result
    if response.status_code == 200:
        submit_data = response.json()
        print("Response:", submit_data["response"])
        print("Found:", submit_data["found"])
        print("Shop:", submit_data["shop_tuple"])

        if submit_data["found"]:
            print(f"Do you want to book a table")

            user_input = input("Input: ")

            # Data to send
            payload = {
                "user_input": user_input,
                "shop_tuple": submit_data["shop_tuple"],
            }

            # Make the POST request
            response = requests.post(booking_url, json=payload)

            if response.status_code == 200:

                booking_data = response.json()

                print("Here is what we got:")
                print("Booking Status:", booking_data["booked"])
                print("Details:", booking_data["entry"])

            else:
                print("Error:", response.status_code, response.text)

    else:
        print("Error:", response.status_code, response.text)
