from flask import Flask, request, jsonify
import dolphin

app = Flask(__name__)

# 加载 Dolphin 模型
model = dolphin.load_model("small", "/workspace/models/DataoceanAI/dolphin-small", "cuda")


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    接收 POST 请求，上传音频文件，并返回识别后的文本。
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        # 将上传的音频文件保存到临时文件
        temp_audio_path = "temp_audio.wav"
        audio_file.save(temp_audio_path)

        # 加载音频文件并进行转录
        waveform = dolphin.load_audio(temp_audio_path)
        result = model(waveform)

        # 返回转录后的文本
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50050)
