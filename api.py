import argparse

from flask import Flask, request, jsonify
import dolphin
from dolphin.model import TranscribeResult

app = Flask("Dolphin API")


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
        result: TranscribeResult = model(waveform)

        # 返回转录后的文本
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行 Dolphin Flask 服务。")
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式。')
    parser.add_argument('--host', default='0.0.0.0', help='服务监听的 IP 地址。')
    parser.add_argument('--port', default="50050", help='服务监听的端口号。')
    parser.add_argument('--model', default='small', help='使用的模型 (small 或 large)。')
    parser.add_argument('--model_dir', default='/workspace/models/DataoceanAI/dolphin-small', help='模型的路径。')
    args = parser.parse_args()

    # 加载 Dolphin 模型
    model = dolphin.load_model(args.model, args.model_dir)

    app.run(debug=args.debug, host=args.host, port=int(args.port))
