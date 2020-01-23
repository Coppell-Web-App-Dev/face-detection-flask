from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post_image', methods=['POST', 'GET'])
def post_image():
    if request.method == 'POST':
        return render_template('display.html')

if __name__ == "__main__":
    app.run(debug=True)