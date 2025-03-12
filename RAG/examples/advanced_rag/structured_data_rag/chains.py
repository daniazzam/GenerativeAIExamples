# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM Chains for executing Retrival Augmented Generation."""
import logging
import os
from typing import Generator, List

import pandas as pd
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers.string import StrOutputParser

from RAG.src.chain_server.base import BaseExample
from RAG.src.chain_server.utils import get_config, get_llm, get_prompts

import json
import datetime
from RAG.examples.advanced_rag.structured_data_rag.flight_api import FlightAPIClient

logger = logging.getLogger(__name__)
settings = get_config()

INGESTED_CSV_FILES_LIST = "ingested_csv_files.txt"

class APIChatbot(BaseExample):
    """RAG example showcasing API parsing"""

    def ingest_docs(self, filepath: str, filename: str):
        """Ingest documents to the VectorDB."""

        if not filename.endswith(".csv"):
            raise ValueError(f"{filename} is not a valid CSV file")

        with open(INGESTED_CSV_FILES_LIST, "a+", encoding="UTF-8") as f:

            ref_csv_path = None

            try:
                f.seek(0)
                ref_csv_path = f.readline()
            except Exception as e:
                # Skip reading reference file path as this is the first file
                pass

            if not ref_csv_path:
                f.write(filepath + "\n")
            else:
                raise ValueError(
                    f"Columns of the file {filepath} do not match the reference columns of {ref_csv_path} file."
                )

        logger.info("Document %s ingested successfully", filename)


    def llm_chain(self, query: str, chat_history: List["Message"], addPhrase: bool = False, **kwargs) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above."""

        logger.info("Using llm to generate response directly without knowledge base.")
        chat_history = []

        system_message = [("system", get_prompts().get("prompts").get("chat_template"))]
        conversation_history = [(msg.role, msg.content) for msg in chat_history]
        user_input = [("user", "{input}")]

        # Checking if conversation_history is not None and not empty
        prompt = (
            ChatPromptTemplate.from_messages(system_message + conversation_history + user_input)
            if conversation_history
            else ChatPromptTemplate.from_messages(system_message + user_input)
        )

        logger.info(f"Using prompt for response generation: {prompt.format(input=query)}")
        chain = prompt | get_llm(**kwargs) | StrOutputParser()
        if addPhrase:
            yield "The following response is based solely on my general knowledge, as no flight API search was needed."
        yield from chain.stream({"input": query})

    def rag_chain(self, query: str, chat_history: List["Message"], **kwargs) -> Generator[str, None, None]:
        """
        Execute a RAG chain that:
        1. Uses the LLM to extract flight search parameters from the user query.
        2. Uses FlightAPIClient to look up entity IDs and call the flight search API.
        3. Uses the API response and the original query to generate a final answer.
        """
        logger.info("Starting RAG API chain with dynamic API call generation.")
        chat_history = []  # Disabling chat history for consistency

        # Get today's date for the prompt
        today = datetime.date.today()

        # --- Step 1: Extract API Parameters from the User Query ---
        prompts_config = get_prompts().get("prompts")
        parameter_prompt_template = PromptTemplate(
            template=prompts_config.get("flight_parameter_extraction_template"),
            input_variables=["query", "today"],
        )

        llm = get_llm(**kwargs)
        parameter_chain = parameter_prompt_template | llm | StrOutputParser()
        param_results = "".join(parameter_chain.stream({"query": query, "today": today}))
        logger.info("Extracted API parameters: %s", param_results)

        try:
            api_params = json.loads(param_results)
            logger.info("Loaded API parameters: %s", api_params)
            
            # If API search is not needed, call llm_chain with an introductory note.
            if not api_params.get("useAPI"):
                logger.info("The user query does not require a flight API search. Using general knowledge.")
                return self.llm_chain(query, chat_history, addPhrase=True, **kwargs)
        
        except Exception as e:
            logger.error("Failed to parse API parameters: %s", e)
            return iter(["Failed to parse API parameters."])
        
        # --- Step 2: Convert City Names to Entity IDs Using FlightAPIClient ---
        flight_api_key = os.environ.get("FLIGHT_API_KEY")
        if not flight_api_key:
            logger.info("FLIGHT_API_KEY not set in environment")
            return iter(["It seems the Flight API key is not set. Please contact the administrator."])
        
        flight_client = FlightAPIClient(api_key=flight_api_key)

        origin_city = api_params.get("originEntityId", "Berlin")
        origin_id = flight_client.get_entity_id_for_city(origin_city)
        logger.info("Converted origin city '%s' to entityId '%s'", origin_city, origin_id)
        api_params["originEntityId"] = origin_id

        if api_params.get("destinationEntityId", "null") != "null":
            dest_city = api_params["destinationEntityId"]
            dest_id = flight_client.get_entity_id_for_city(dest_city)
            logger.info("Converted destination city '%s' to entityId '%s'", dest_city, dest_id)
            api_params["destinationEntityId"] = dest_id
        else:
            logger.info("Destination city is 'null'; removing from parameters")
            api_params.pop("destinationEntityId", None)

        # --- Step 3: Call the Flight Search API ---
        api_response = flight_client.search_flights(api_params)
        logger.info("API response: %s", api_response)

        # Check if the API response contains data
        if not api_response.get("data"):
            logger.info("API response contains no data.")
            return iter(["No flight data available for the given query."])

        # --- Step 4: Extract Relevant Flight Content ---
        extracted_contents = flight_client.extract_flight_contents(api_response)
        filtered_api_response_str = json.dumps(extracted_contents, indent=2)
        logger.info("Extracted and filtered API data: %s", filtered_api_response_str)

        # --- Step 5: Generate Final Answer Using the Filtered API Response ---
        response_template_str = prompts_config.get("flight_response_template_new")
        answer_prompt_template = PromptTemplate(
            template=response_template_str,
            input_variables=["api_response", "query"],
        )
        answer_chain = answer_prompt_template | llm | StrOutputParser()
        return answer_chain.stream({
            "api_response": filtered_api_response_str,
            "query": query
        })

    def get_documents(self) -> List[str]:
        """Retrieves filenames stored in the vector store."""
        decoded_filenames = []
        if os.path.exists(INGESTED_CSV_FILES_LIST):
            with open(INGESTED_CSV_FILES_LIST, "r", encoding="UTF-8") as file:
                for csv_file_path in file.read().splitlines():
                    decoded_filenames.append(os.path.basename(csv_file_path))
        return decoded_filenames

    def delete_documents(self, filenames: List[str]):
        """Delete documents from the vector index."""
        logger.error("delete_documents not implemented")
        return True
