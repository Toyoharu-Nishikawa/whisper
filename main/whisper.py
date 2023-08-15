from faster_whisper import WhisperModel
#model_size = "base"
model_size = "medium"
#model_size = "large-v2"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

#segments, info = model.transcribe("../audio/sampleTokyo.wav", beam_size=5, language='ja')
segments, info = model.transcribe("../audio/001-sibutomo.mp3", beam_size=5, language='ja')
results = list(segments)
print(' '.join([result.text for result in results]))

