from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document


def extract_video_id(url: str) -> str:
  
    parsed_url = urlparse(url)

    
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

   
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    raise ValueError("Invalid YouTube URL.")


def load_transcript(url: str):
    

    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError("Could not extract Video ID.")

    try:
        ytt_api = YouTubeTranscriptApi()

        transcript = ytt_api.fetch(video_id, languages=["en"])

        transcript_text = " ".join(
        chunk.text for chunk in transcript
)   

        document = Document(
            page_content=transcript_text,
            metadata={
                "video_id": video_id,
                "source": url
            }
        )

        return document,video_id

    except Exception as e:
        raise RuntimeError(f"Failed to load transcript: {e}")