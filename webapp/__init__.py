from flask import Flask

app = Flask(__name__)
app.config.from_object('web_config')

from webapp import views