from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfFileReader
from transformers import pipeline
from flask import Response
import json

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return Response(str(e), status=500, content_type='text/plain')



@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if 'file' not in request.files:
            return Response(json.dumps({'error': 'No file part'}), status=400, content_type='application/json')

        file = request.files['file']

        if file.filename == '':
            return Response(json.dumps({'error': 'No selected file'}), status=400, content_type='application/json')

        if file:
            # Save the uploaded PDF file
            pdf_path = os.path.join('uploads', file.filename)
            file.save(pdf_path)

            # Use ChatGPT for summarization
            summarizer = pipeline("summarization", model="Falconsai/text_summarization")

        
            pdf_content = extract_text_from_pdf(pdf_path)
        try:
            summary = summarizer(pdf_content, max_length=146, min_length=50, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2, top_k=50)

            return jsonify({'summary': summary[0]['summary']})
        except Exception as e:
            print(f"Exception during summarization: {str(e)}")
            return Response(json.dumps({'error': 'Invalid request'}), status=400, content_type='application/json')

    except Exception as e:
        print("Exception:", str(e))
        return Response(json.dumps({'error': 'Internal server error'}), status=500, content_type='application/json')



def extract_text_from_pdf(pdf_path):
    try:
        import fitz
        with fitz.open(pdf_path) as doc:
            text = ''
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
        return text
    except Exception as e:
        print("PDF Extraction Error:", str(e))
        return ""



def generate_summary(text):
    summarizer = pipeline('summarization', model="Falconsai/text_summarization")
    summary = summarizer(text, max_length=146, min_length=50, length_penalty=2.0, num_beams=4)
    return summary[0]['summary']


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
