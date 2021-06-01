import os.path as osp
from flask import Flask, render_template, request

import nodetails
import nodetails.abs
import tensorflow.keras.backend as K; K.clear_session()
nodetails.enable_vram_growth()

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        text_body = request.form["content"]
        summary = nodetails.abs.make_inference(model, text_body)

        return render_template("index.html", text=text_body, summary=summary)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    model_dir = "../data/_models"
    model_name = "nodetails--food_reviews--80-10--None.model"
    model = nodetails.abs.load_model(osp.join(model_dir, model_name))

    app.run(debug=nodetails.is_debug())
