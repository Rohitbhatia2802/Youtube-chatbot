from urllib.parse import urlparse, parse_qs
import yt_dlp
import os
import tempfile
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

        with tempfile.TemporaryDirectory() as temp_dir:

            ydl_opts = {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "skip_download": True,
                "subtitleslangs": ["en"],
                "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
                "quiet": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            transcript_text = ""

            for file in os.listdir(temp_dir):
                if file.endswith(".vtt"):
                    with open(
                        os.path.join(temp_dir, file),
                        encoding="utf-8"
                    ) as f:

                        lines = f.readlines()

                    transcript_text = " ".join(
                        line.strip()
                        for line in lines
                        if "-->" not in line
                        and not line.startswith("WEBVTT")
                        and line.strip()
                    )

                    break

            if not transcript_text:
                raise RuntimeError("English subtitles not found.")

            document = Document(
                page_content=transcript_text,
                metadata={
                    "video_id": video_id,
                    "source": url,
                },
            )

            return document, video_id

    except Exception as e:
        raise RuntimeError(f"Failed to load transcript: {e}")