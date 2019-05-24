from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return "Home page"


@app.route("/login")
def login():
    return "Login page"


@app.route("/about")
def about():
    return "About page"


@app.route("/<string:username>")
def hello(username):
    # username = request.args.get('username')
    username = username.capitalize()
    return f"<h1>Hello, {username}</h1>"


if __name__ == '__main__':
    app.run(port=9190)
