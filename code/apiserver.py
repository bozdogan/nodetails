import sys
from flask import Flask, request, jsonify
import nodetails as nd
from nodetails import engines

app = Flask(__name__)

abs_model_dir = "../data/_models"

available_configurations = {
    # "abs-food_reviews-engine": engines.AbstractiveEngine(
    #     "nodetails-food", model_dir=abs_model_dir,
    #     model_name="nodetails--food_reviews--80-10--None.model"),
    # "abs-news_summary-engine": engines.AbstractiveEngine(
    #     "nodetails-news1", model_dir=abs_model_dir,
    #     model_name="nodetails--news_summary--150-20--None.model"),
    "abs-wikihow-engine": engines.AbstractiveEngine(
        "nodetails-how", model_dir=abs_model_dir,
        model_name="nodetails--wikihow--120-20--None.model"),
    "ext-engine": engines.ExtractiveEngine(
        "nodetails-extractive", length=10, preset="wikipedia"),
    "integrated-engine": engines.IntegratedEngine(
        "nodetails-integraged", model_dir=abs_model_dir,
        model_name="nodetails--wikihow--120-20--None.model", length=10, preset="article")
}

@app.route("/")
def index():
    return '{"info": "Use /sum route for summarization."}'


@app.route("/sum", methods=["POST"])
def get_summary():
    engine_name = request.form["engine"]
    text = request.form["text"]

    summary = {"engine": engine_name}

    if engine_name in available_configurations:
        engine = available_configurations[engine_name]
        engine.load()
        summary.update(engine.summarize(text, keep_references=True))
    else:
        return jsonify(
            {"error": "Requested engine not found. For a list of "
                      "available configurations call /list endpoint."})

    return jsonify(summary)


@app.route("/list")
def list_available_configurations():
    return jsonify(list(available_configurations))


@app.after_request
def add_cors_header(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    host = None
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("host="):
            host = sys.argv[1][len("host="):]
    
    nd.enable_vram_growth()
    app.run(port=5000, debug=nd.is_debug(), host=host)

# END OF apiserver.py
