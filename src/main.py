# main.py
import json
import random
import os
import logging
from random import randint
from dotenv import load_dotenv
from internetarchive import (
    File,
    configure,
    search_items,
)
from langchain_core.tools import tool
from langchain_core.utils.json import parse_partial_json
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from src.config.channels import channels
from fastapi.middleware.cors import CORSMiddleware

_ = load_dotenv()

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://jennymaeleidig.github.io",
    "https://jennymaeleidig.github.io",
    "http://localhost",
    "http://localhost:4200",
    "https://localhost",
    "https://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging #
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoQuery(BaseModel):
    """
    Represents a query for video retrieval.

    Attributes:
        query (str): The search query provided by the user.
    """
    query: str


class Video(BaseModel):
    """
    Represents a video object with basic metadata.

    Attributes:
        url (str): The URL of the video.
        title (str): The title of the video.
        uploader (str): The uploader of the video.
        duration (float): The duration of the video in seconds.
    """
    url: str = "N / A"
    title: str = "N / A"
    uploader: str = "N / A"
    duration: float = 0.0


class VideoOutput(BaseModel):
    """
    Represents the structured output for a video, including URL, title, uploader, and duration.
    This model is used for parsing the AI agent's final answer.

    Attributes:
        url (str): The URL of the video.
        title (str): The title of the video.
        uploader (str): The uploader of the video.
        duration (float): The duration of the video in seconds.
    """
    url: str = Field(description="The URL of the video")
    title: str = Field(description="The title of the video")
    uploader: str = Field(description="The uploader of the video")
    duration: float = Field(description="The duration of the video in seconds")


def convert_to_video(file: File) -> Video:
    """
    Converts a File object from the Internet Archive to a Video model.

    Args:
        file (File): The Internet Archive File object to convert.

    Returns:
        Video: A Video object populated with data from the File object's metadata.
    """
    # Access metadata safely, providing defaults if keys are missing
    metadata = file.item.metadata
    file_metadata = file.metadata
    logger.info(f"Converting file {file.name} to video")

    return Video(
        url=file.url,
        title=metadata.get("title", "No title available"),
        uploader=metadata.get("uploader", "Unknown uploader"),
        duration=float(file_metadata.get("length", 0.0)),
    )


ACCEPTED_VIDEO_FORMATS = [
    "h.264",
    "h.264 IA",
    "MPEG4",
    "Ogg Video",
    "WebM",
]


@tool
def get_random_video(search_term: str = "", collection: str = "") -> Video | None:
    """
    Retrieves a random video from the Internet Archive based on specified criteria.

    Args:
        search_term (str, optional): A term to search for within video metadata.
                                     If provided, a random video matching this term is returned.
        collection (str, optional): The name of a specific collection to search within.
                                    If provided, the search is limited to this collection.

    Returns:
        Video | None: A Video object if a suitable video is found, otherwise None.
                      If no search term or collection is provided, it defaults to the
                      'vhscommercials' collection.
    """
    logger.info(f"Getting random video with search term: {search_term} and collection: {collection}")

    ia_conf: str = os.path.abspath("./src/config/ia.ini")

    _ = configure(
        os.getenv("IA_USER") or "", os.getenv("IA_PASS") or "", config_file=ia_conf
    )

    query = "mediatype:movies"

    if search_term or collection:
        if search_term :
            query = f"{query} AND ({search_term})"
            logger.info(f"Searching for video with query: '{query}'")
        if collection:
            query = f"{query} AND collection:{collection}"
            logger.info(f"Searching for video in collection '{collection}' with query: '{query}'")
    else:
        query = f"{query} AND collection:vhscommercials"
        logger.info(
            f"Searching for random video in default collection with query: '{query}'"
        )

    num_results = int(search_items(query, config_file=ia_conf).num_found)
    logger.info(f"Total results found: {num_results}")

    max_attempts = 10  # Limit attempts to find a video
    items_per_page = 20  # Fetch more items per page to increase chances of finding a suitable video

    for attempt in range(max_attempts):
        logger.info(
            f"Attempt {attempt + 1}/{max_attempts} to find a random video from {num_results} results."
        )
        # Calculate the maximum possible page number based on items_per_page
        max_page = (num_results + items_per_page - 1) // items_per_page
        random_page = randint(1, max_page if max_page > 0 else 1) # Ensure random_page is at least 1

        search_results = search_items(
            query, config_file=ia_conf, params={"rows": items_per_page, "page": random_page}
        ).iter_as_items()

        try:
            for item in search_results:
                for file in item.get_files(formats=ACCEPTED_VIDEO_FORMATS):
                    logger.info(f"Found video: {file.name}")
                    return convert_to_video(file)

            # If loop finishes without finding a suitable file in the items on this page
            logger.info(
                f"No suitable video file found in the items on page {random_page}. Retrying."
            )

        except json.JSONDecodeError as e:
            logger.error(
                f"JSON decode error while processing search results on page {random_page}: {e}"
            )
            # Continue to the next attempt
            continue 
    logger.warning("No video found after maximum attempts.")
    return None

@tool
def get_collection_from_chanel(channel: str) -> str | None:
    """
    Retrieves a random collection name associated with a given channel.

    Args:
        channel (str): The name of the channel to retrieve a collection from.
                       This must be one of the predefined channel keys.

    Returns:
        str | None: A randomly selected collection name if the channel is valid,
                    otherwise None.
    """
    logger.info(f"Getting collection from channel: {channel}")

    if channel.strip().lower() in channels.keys():
        collection = random.choice(channels[channel.strip().lower()])
        logger.info(f"Found collection: {collection} for channel: {channel}")
        return collection
    logger.info(f"No collection found for channel: {channel}")
    return None
        
# setup AI agent
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14"))
tools = [get_random_video, get_collection_from_chanel]
# Create parsers for different outputs
video_parser = PydanticOutputParser(pydantic_object=VideoOutput)

template = """
    You are an AI agent that can retrieve a random video from Archive.org.
    You have access to two tools: `get_collection_from_chanel` and `get_random_video`.
    The `get_collection_from_chanel` tool retrieves a random collection from a specific channel.
    The tool accepts a `channel` parameter, which is the name of the channel to retrieve a collection from.
    The `channel` parameter must be one of the following channels: {channels}.
    If the user's query mentions a specific channel, you should call `get_collection_from_chanel` with that channel as the `channel` parameter.
    Otherwise, if the user's query does not mention a specific channel, you should not call `get_collection_from_chanel`.
    The `get_random_video` tool searches for a random video file across Archive.org:
    The tool `get_random_video` tool accepts an optional `collection parameter to specify the collection to search in, and an optional search_term` parameter to filter the search results.
    If you called `get_collection_from_chanel`, you should use the collection name as the `collection` parameter.
    Otherwise, if you did not call `get_collection_from_chanel`, you should not use the `collection` parameter.
    If the user's query mentions a specific search term, you should use that as the `search_term` parameter.
    Otherwise, if the user's query does not mention a specific search term, you should not use the `search_term` parameter.
    If the user does not provide a specific search criteria, you can call `get_random_video` without a `search_term` or `collection`.
    If the user's request contains any Not Safe For Work (NSFW) or inappropriate material, you should ignore their request call `get_random_video` without a `search_term` or `collection`.
    If the returned video contains any Not Safe For Work (NSFW) or inappropriate material, you should ignore the video and call `get_random_video` again with the same input parameters.
    Your FINAL ANSWER MUST be the video retrieved by the `get_random_video` tool, formatted STRICTLY as a JSON string.
    Include ONLY the JSON object itself, with no surrounding text or markdown. The JSON object must match the structure of the Video model with fields: `url`, `title`, `uploader`, and `duration`.
    If you cannot retrieve a video, retry this process up to 3 times. If you cannot retrieve a video after 3 attempts, stop retrying and return an empty JSON object.
    {video_format_instructions}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(
    video_format_instructions=video_parser.get_format_instructions(),
)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Generated docs endpoint
@app.get("/")
async def get_docs():
    return RedirectResponse("/docs")


@app.post("/video")
async def get_video(query: VideoQuery) -> VideoOutput:
    """
    Processes a video query using an AI agent and returns a structured video output.

    Args:
        query (VideoQuery): The input query containing the search term for the video.

    Returns:
        VideoOutput: A structured object containing the URL, title, uploader, and duration of the retrieved video.

    Raises:
        HTTPException:
            - 404 if no suitable video is found or the agent fails to retrieve one.
            - 500 if the agent returns invalid JSON or an unexpected error occurs.
    """
    logger.info(f"Received query: {query.query}")
    try:
        result = agent_executor.invoke({"input": query.query, "channels": (','.join(channels.keys()))})
        agent_output = result["output"]
        logger.debug(f"Agent raw result: {agent_output}")

        try:
            # Attempt to parse the full JSON output
            parsed_output = video_parser.parse(agent_output)
            logger.info("Full JSON parse successful.")
            return parsed_output
        except Exception:
            logger.debug("Full JSON parse failed, attempting partial parse.")
            # If full parsing fails, try to parse partial JSON
            partial_json_output = parse_partial_json(agent_output)
            if isinstance(partial_json_output, dict):
                video = VideoOutput(**partial_json_output)
                logger.info("Partial JSON parse successful.")
                return video
            logger.error(f"Failed to parse valid JSON from agent output: {agent_output}")
            raise ValueError("Agent returned output that could not be parsed as valid JSON.")

    except ValueError as e:
        logger.exception("Agent returned output that could not be parsed as valid JSON.")
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        logger.exception("Unable to find a suitable video or agent failed.")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("An unexpected error occurred while processing the query")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
