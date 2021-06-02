from flask import Flask, render_template, request
import nodetails

app = Flask(__name__)


if __name__ == "__main__":
    app.run(debug=nodetails.is_debug())

# END OF apiserver.py
