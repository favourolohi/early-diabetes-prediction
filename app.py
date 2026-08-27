import gradio as gr
import pandas as pd
import joblib

# Load the trained Logistic Regression model
model = joblib.load("diabetes_model.pkl")

# The exact feature order used during model training
FEATURES = [
    "Age",
    "Gender",
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity"
]


def predict_diabetes(
    age,
    gender,
    polyuria,
    polydipsia,
    sudden_weight_loss,
    weakness,
    polyphagia,
    genital_thrush,
    visual_blurring,
    itching,
    irritability,
    delayed_healing,
    partial_paresis,
    muscle_stiffness,
    alopecia,
    obesity
):

    input_data = pd.DataFrame([[
        age,
        gender,
        polyuria,
        polydipsia,
        sudden_weight_loss,
        weakness,
        polyphagia,
        genital_thrush,
        visual_blurring,
        itching,
        irritability,
        delayed_healing,
        partial_paresis,
        muscle_stiffness,
        alopecia,
        obesity
    ]], columns=FEATURES)

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = f"""
        <div class="result positive">
            <h2>Prediction: Positive</h2>
            <p>Probability of Positive Classification</p>
            <h1>{probability * 100:.2f}%</h1>
        </div>
        """
    else:
        result = f"""
        <div class="result negative">
            <h2>Prediction: Negative</h2>
            <p>Probability of Positive Classification</p>
            <h1>{probability * 100:.2f}%</h1>
        </div>
        """

    return result


custom_css = """
body {
    background: #faf8fc;
}

.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}

.header {
    text-align: center;
    padding: 30px 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #eee4f7, #ffffff);
    margin-bottom: 20px;
    border: 1px solid #dfcfee;
}

.header h1 {
    color: #76529b;
    font-size: 34px;
    margin-bottom: 10px;
}

.header .student {
    color: #4f3b62;
    font-size: 20px;
    font-weight: bold;
}

.header .school {
    color: #666;
    font-size: 16px;
}

.header .topic {
    background: #ffffff;
    border-radius: 12px;
    padding: 12px;
    margin-top: 18px;
    color: #76529b;
    font-size: 15px;
}

.result {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 10px;
}

.result h1 {
    font-size: 42px;
    margin: 10px 0;
}

.positive {
    background: #f3e8f8;
    border: 2px solid #b58ad3;
}

.negative {
    background: #f1f8f4;
    border: 2px solid #9ac5a9;
}

.disclaimer {
    text-align: center;
    color: #777;
    font-size: 13px;
    padding: 20px;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="purple",
        neutral_hue="slate"
    ),
    css=custom_css
) as app:

    gr.HTML("""
    <div class="header">
        <h1>Early Diabetes Prediction System</h1>

        <div class="student">
            FAVOUR AJEGBA
        </div>

        <div class="school">
            Department of Mathematics<br>
            Ahmadu Bello University, Zaria
        </div>

        <div class="topic">
            <b>Project Topic</b><br>
            Using Mathematical Modelling and Machine Learning
            to Predict Early Diabetes
        </div>
    </div>
    """)

    gr.Markdown("## 👤 Patient Information")

    with gr.Row():
        age = gr.Number(
            label="Age",
            value=40,
            minimum=1,
            maximum=120
        )

        gender = gr.Dropdown(
            choices=[
                ("Female", 0),
                ("Male", 1)
            ],
            label="Gender",
            value=0
        )

    gr.Markdown("## 🩺 Symptoms and Health Indicators")

    with gr.Row():
        polyuria = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Polyuria",
            value=0
        )

        polydipsia = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Polydipsia",
            value=0
        )

        sudden_weight_loss = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Sudden weight loss",
            value=0
        )

    with gr.Row():
        weakness = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Weakness",
            value=0
        )

        polyphagia = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Polyphagia",
            value=0
        )

        genital_thrush = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Genital thrush",
            value=0
        )

    with gr.Row():
        visual_blurring = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Visual blurring",
            value=0
        )

        itching = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Itching",
            value=0
        )

        irritability = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Irritability",
            value=0
        )

    with gr.Row():
        delayed_healing = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Delayed healing",
            value=0
        )

        partial_paresis = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Partial paresis",
            value=0
        )

        muscle_stiffness = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Muscle stiffness",
            value=0
        )

    with gr.Row():
        alopecia = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Alopecia",
            value=0
        )

        obesity = gr.Dropdown(
            [("No", 0), ("Yes", 1)],
            label="Obesity",
            value=0
        )

    predict_button = gr.Button(
        "🔮  PREDICT DIABETES",
        variant="primary",
        size="lg"
    )

    gr.Markdown("## 📊 Prediction Result")

    result = gr.HTML(
        "<div class='result'><h3>Enter patient information and click Predict.</h3></div>"
    )

    predict_button.click(
        fn=predict_diabetes,
        inputs=[
            age, gender, polyuria, polydipsia,
            sudden_weight_loss, weakness, polyphagia,
            genital_thrush, visual_blurring, itching,
            irritability, delayed_healing, partial_paresis,
            muscle_stiffness, alopecia, obesity
        ],
        outputs=result
    )

    gr.HTML("""
    <div class="disclaimer">
        <b>Academic Research Tool</b><br>
        This system was developed as part of a final-year Mathematics project.
        It is intended for research and educational purposes and does not replace
        professional medical diagnosis or clinical judgment.
    </div>
    """)


app.launch()
