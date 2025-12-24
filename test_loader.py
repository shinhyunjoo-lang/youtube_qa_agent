from langchain_community.document_loaders import YoutubeLoader
import traceback

def test_load():
    url = "https://www.youtube.com/watch?v=kCc8FmEb1nY" # DeepMind's "Walking with Missives" or any safe video
    # url = "https://www.youtube.com/watch?v=jNQXAC9IVRw" # "Me at the zoo" (very old, short)
    
    print(f"Testing URL: {url}")
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        docs = loader.load()
        print("Successfully loaded!")
        print(docs[0].metadata)
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
