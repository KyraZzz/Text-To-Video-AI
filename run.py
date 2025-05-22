from groq import Groq
import edge_tts
import asyncio
import os
import whisper_timestamped as whisper

groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 1. Generate a script from Google NotebookLLM for the given learning topic
# 
script_from_gllm = """
Have you ever followed a recipe, put together furniture using instructions, or even just followed a map? If so, you've used an algorithm!
An algorithm is essentially a sequence of steps that you carry out to perform a task. Think of it as a clear, step-by-step guide to solve a problem or complete a task.
In computer science, we use algorithms as the solution to a problem, expressed as a sequence of defined steps.
The text gives an example of a recipe for a cake, like "Mix together all the ingredients," as a simple algorithm. It's a set of steps you follow to get a specific result: a cake! Computer scientists are always looking for "good solutions" – algorithms that give correct results, use resources efficiently like memory and time, and are also clear and easy to understand.
"""

# response = groq.chat.completions.create(
#     model="llama3-8b-8192",
#     messages=[{"role": "user", "content": script_from_gllm}],
# )

# print(response)

async def generate_audio(text,outputFilename):
    communicate = edge_tts.Communicate(text,"en-AU-WilliamNeural")
    await communicate.save(outputFilename)
SAMPLE_FILE_NAME = "output.wav"
asyncio.run(generate_audio(script_from_gllm, SAMPLE_FILE_NAME))


audio = whisper.load_audio(SAMPLE_FILE_NAME)
model = whisper.load_model("base", device="cpu")
result = whisper.transcribe(model, audio, language="en", task="transcribe")

print(result)

# gen = transcribe_timestamped(WHISPER_MODEL, audio_filename, verbose=False, fp16=False)
# timed_captions = generate_timed_captions(SAMPLE_FILE_NAME)
# print(timed_captions)