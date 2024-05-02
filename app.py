# app.py
from flask import Flask, request, jsonify
from word_generator import WordGenerator
from sentence_generator import SentenceGenerator

app = Flask(__name__)
word_generator = WordGenerator()
sentence_generator = SentenceGenerator()


@app.route("/generate_word", methods=["POST"])
def generate_word():
    if request.is_json:
        data = request.get_json()
        step_no = data.get("step_no")
        n_words = data.get("n_words")
        kind = data.get("kind")

        if not step_no or not n_words:
            return jsonify({"error": "난이도와 생성할 단어 개수를 입력하세요."}), 400

        try:
            step_no = int(step_no)
            n_words = int(n_words)
            kind = int(kind)
        except ValueError:
            return (
                jsonify(
                    {
                        "error": "잘못된 입력입니다. 난이도와 생성할 단어 개수는 정수값이어야 합니다."
                    }
                ),
                400,
            )

        if kind == 1:
            generated_words = word_generator.generate_word(step_no, n_words)
        else:
            generated_words = word_generator.generate_word_with_gpt(step_no, n_words)

        return jsonify({"generated_words": generated_words})
    else:
        return jsonify({"error": "Request must be JSON"}), 415


@app.route("/generate_sentence", methods=["POST"])
def generate_sentence():
    if request.is_json:
        data = request.get_json()
        step_no = data.get("step_no")
        n_sentences = data.get("n_sentences")
        kind = data.get("kind")

        if not step_no or not n_sentences:
            return jsonify({"error": "난이도와 생성할 문장 개수를 입력하세요."}), 400

        try:
            step_no = int(step_no)
            n_sentences = int(n_sentences)
            kind = int(kind)
        except ValueError:
            return (
                jsonify(
                    {
                        "error": "잘못된 입력입니다. 난이도와 생성할 문장 개수는 정수값이어야 합니다."
                    }
                ),
                400,
            )

        if kind == 1:
            generated_sentences = sentence_generator.generate_sentence(
                step_no, n_sentences
            )
        else:
            generated_sentences = sentence_generator.generate_sentences_with_gpt(
                step_no, n_sentences
            )

        return jsonify({"generated_sentences": generated_sentences})
    else:
        return jsonify({"error": "Request must be JSON"}), 415


if __name__ == "__main__":
    app.run(debug=True)
