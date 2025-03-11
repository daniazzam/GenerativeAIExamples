import requests
import os

class FlightsAPIClient:
    def __init__(self):
        self.api_url = "https://sky-scrapper.p.rapidapi.com/api/v2/flights/searchFlightEverywhere"
        self.api_key = os.getenv("RAPIDAPI_KEY") 
        self.api_host = "sky-scrapper.p.rapidapi.com"

    def search_flights(self, travelDate, returnDate=None, originEntityId=95673320, cabinClass="economy", journeyType="one_way", currency="USD"):
        """Fetch flight details from the API."""
        query_params = {
            "originEntityId": originEntityId,
            "travelDate": travelDate,
            "cabinClass": cabinClass,
            "journeyType": journeyType,
            "currency": currency,
        }

        if returnDate:
            query_params["returnDate"] = returnDate  # Optional field

        headers = {
            "x-rapidapi-key": "17d87c7187msha4597d54e7512fbp17daa4jsnc28771aae1f5",
            "x-rapidapi-host": self.api_host,
        }

        try:
            response = requests.get(self.api_url, headers=headers, params=query_params, timeout=10)
            response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None


# client = FlightsAPIClient()
# print(client.search_flights(travelDate="2025-03-04"))