import whisper
import json
import os
#import sys
#sys.stdout.reconfigure(encoding='utf-8')

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios:
    print(audio)
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number,title)
        #result = model.transcribe(audio = f"audios/{audio}.mp3",
        result = model.transcribe(audio = f"audios/{audio}",
                                   language = "hi",
                                    task = "transcribe",
                                    word_timestamps = False )

        chunks = []
        for segment in result["segments"]:
            chunks.append({"number": number, "title": title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})


        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        #print(chunks)

        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata,f, ensure_ascii=False,indent=4)
    