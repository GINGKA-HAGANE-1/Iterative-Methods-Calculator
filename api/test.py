from flask import Flask, jsonify
import sys
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Hello World"

@app.route('/api/test', methods=['GET'])
def test():
    try:
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Error: {str(e)}")
    print(traceback.format_exc())
    return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=False)