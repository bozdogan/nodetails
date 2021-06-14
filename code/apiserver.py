import os.path as osp
from flask import Flask, request, jsonify
import nodetails as nd
from nodetails import ndabs, ndext
from nodetails import engines

app = Flask(__name__)

abs_model_dir = "../data/_models"
abs_model_name = None
abs_model = None

available_configurations = {
    "abs-food_reviews-engine" : engines.AbstractiveEngine(
        "nodetails-food", model_dir=abs_model_dir,
        model_name="nodetails--food_reviews--80-10--None.model"),
    "ext-engine" : engines.ExtractiveEngine(
        "nodetails-extractive", length=10, preset="wikipedia"),
    "integrated-engine" : engines.IntegratedEngine(
        "nodetails-integraged", model_dir=abs_model_dir,
        model_name="", length=12, preset="article")
}

@app.route("/")
def index():
    return '{"info": "Use /sum route for summarization."}'


@app.route("/sum", methods=["POST"])
def get_summary():
    global abs_model
    global abs_model_name

    sum_method = request.form["method"]
    model_name = request.form["model_name"]
    text = request.form["text"]

    summary = {"type": sum_method}

    if sum_method == "abs":
        if abs_model is None or abs_model_name != model_name:
            try:
                abs_model = ndabs.load_model(osp.join(abs_model_dir, model_name))
                abs_model_name = model_name
            except Exception as e:
                print(e)

        if abs_model:
            summary["summary"] = ndabs.make_inference(abs_model, text)
        else:
            return jsonify({"message": "Error: Model not found"})
    elif sum_method == "ext":
        extsum = ndext.get_summary(text, length=7)

        summary["summary"] = extsum.summary
        # summary["reference"] = extsum.reference
        # summary["sentences"] = extsum.sentences
        # summary["paragraphs"] = extsum.paragraphs
    else:
        return jsonify({"message": f"Error: No method called '{sum_method}'"})

    return jsonify(summary)


@app.route("/list")
def list_available_configurations():
    return jsonify(list(available_configurations))


if __name__ == "__main__":
    nd.enable_vram_growth()
    app.run(port=5000, debug=nd.is_debug())

# END OF apiserver.py
