from gradio_client import Client


client = Client("YouLearn/faster-whisper")

response = client.predict("tests/audio.mp3", 'tiny')

transcript = response['transcript']

for chunk in transcript:
    start = chunk['start']
    end = chunk['end']
    text = chunk['text']
    print(f"{start} -> {end} | {text}\n")