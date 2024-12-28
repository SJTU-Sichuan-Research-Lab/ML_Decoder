import os

from flask import Flask, jsonify, render_template
from flask_cors import CORS  # Import the CORS extension
from api.nas import nas_api  # Import the nas API blueprint

app = Flask(__name__)

# Enable CORS for all domains, including localhost and 0.0.0.0
CORS(app)  # allow all domains
# CORS(app, origins=["http://localhost", "http://127.0.0.1", "http://0.0.0.0"]) # specify the allowed origins:

# Register the nas API blueprint
app.register_blueprint(nas_api)

# Modify the js file, replace the API_BASE with env parameter when starting app
api_base = os.environ.get('API_BASE', 'localhost:8889')
js_file_path = 'static/js/home.js'
with open(js_file_path, 'r') as file:
    content = file.read()
    content = content.replace('{{API_BASE}}', api_base)
    file.close()
with open(js_file_path, 'w') as file:
    file.write(content)
    file.close()


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8889)
