import json
import requests
import logging

logger = logging.getLogger(__name__)

class FlightAPIClient:
    def __init__(self, api_key: str, host: str = "sky-scrapper.p.rapidapi.com"):
        self.api_key = api_key
        self.host = host
        self.flight_search_url = f"https://{host}/api/v2/flights/searchFlightEverywhere"
        self.airport_search_url = f"https://{host}/api/v1/flights/searchAirport"

    def get_entity_id_for_city(self, city: str) -> str:
        """
        Look up the airport for the given city and return the first entityId.
        If the lookup fails, returns the original city value.
        """
        querystring = {"query": city, "locale": "en-US"}
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        try:
            response = requests.get(self.airport_search_url, headers=headers, params=querystring)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("API request for city %s failed: %s", city, e)
            return city  # Fallback to the city name if lookup fails

        data = response.json()
        if data.get("status") and data.get("data") and len(data.get("data")) > 0:
            first_result = data.get("data")[0]
            return first_result.get("entityId", city)
        return city

    def search_flights(self, api_params: dict) -> dict:
        """
        Call the flight search API using the provided parameters.
        """
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        try:
            response = requests.get(self.flight_search_url, headers=headers, params=api_params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("Flight search API request failed: %s", e)
            return {}

    def extract_flight_contents(self, api_response: dict) -> list:
        """
        Extracts the 'content' field from each result in the API response.
        Returns a list of these content objects.
        """
        try:
            results = api_response.get("data", {}).get("results", [])
            extracted_contents = [result.get("content", {}) for result in results if result.get("content")]
            return extracted_contents
        except Exception as e:
            logger.error("Error extracting flight contents: %s", e)
            return []
