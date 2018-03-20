from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField,TextAreaField
from wtforms.validators import DataRequired


class InputTextForm(FlaskForm):
    text = TextAreaField('input_text',validators=[DataRequired()])
