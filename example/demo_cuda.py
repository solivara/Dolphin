import dolphin

waveform = dolphin.load_audio("data/audio.wav")
model = dolphin.load_model("small", "/workspace/models/DataoceanAI/dolphin-small", "cuda")

result1 = model(waveform)

# Specify language and region
result2 = model(waveform, lang_sym="zh", region_sym="CN")

print(result1.text)
print(result2.text)
