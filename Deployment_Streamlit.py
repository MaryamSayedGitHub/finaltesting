import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Amazon Project",
        options=["Home Page", "Avalibality status"],
    )

if selected == "Home Page":
    st.write("# We're glad to see you here....")

    def main():
        # Display the image using the st.image command.
        st.image("/content/images.jpg", use_column_width=True)

    if __name__ == "__main__":
        main()

elif selected == "Avalibality status":
    # Define the `prepare_input_data_for_model()` function.
    def prepare_input_data_for_model(
        title,
        price,
        rating,
        total_reviews,
        manufacturer,
        os,
        ram,
        weight,
        batteries,
        special_features,
        form_factor,
        colour,
        battery_power_rating,
        product_description,
        reviews,
        product_volume,
        reviews_rating,
    ):
        A = [
            title,
            price,
            rating,
            total_reviews,
            manufacturer,
            os,
            ram,
            weight,
            batteries,
            special_features,
            form_factor,
            colour,
            battery_power_rating,
            product_description,
            reviews,
            product_volume,
            reviews_rating,
        ]
        sample = np.array(A).reshape(-1, len(A))
        return sample

    # Load the saved model.
    with open("Amazon.sav", "rb") as f:
        loaded_model = pickle.load(f)

    # Display the title and subheader.
    st.write("# availability status")
    st.write("---")
    st.subheader("Enter The Features to predict Availability")

    # Display the image using the st.image command.
    st.image("/content/tenor.gif", use_column_width=True)

    # Get the user input.
    name = st.text_input("Name:")
    title = st.number_input("Title: ")
    price = st.number_input("Price: ")
    rating=st.number_input('rating : ')
    total_reviews = st.number_input("Total Reviews: ")
    manufacturer = st.number_input("Manufacturer: ")
    os = st.number_input("Operating System: ")
    ram = st.number_input("RAM: ")
    weight = st.number_input("Weight: ")
    batteries = st.number_input("Batteries: ")
    special_features = st.number_input("Special Features: ")
    form_factor = st.number_input("Form Factor: ")
    colour = st.number_input("Colour: ")
    battery_power_rating = st.number_input("Battery Power Rating: ")
    product_description = st.number_input("Product Description: ")
    reviews = st.number_input("Reviews: ")
    product_volume = st.number_input("Product Volume: ")
    reviews_rating = st.number_input("Reviews Rating: ")

    # Prepare the input data for the model.
    sample = prepare_input_data_for_model(
        title,
        price,
        rating,
        total_reviews,
        manufacturer,
        os,
        ram,
        weight,
        batteries,
        special_features,
        form_factor,
        colour,
        battery_power_rating,
        product_description,
        reviews,
        product_volume,
        reviews_rating,
    )

    # Make the prediction.
    pred_Y = loaded_model.predict(sample)

    # Display the prediction.
    if pred_Y == 0:
        st.write("## Predicted Specie ")
        st.write(f"### Congratulations, {name}.\n The Product is Available")
        st.balloons()
    elif pred_Y == 1:
        st.write(f"### Congratulations, {name}.\n The Product is not Available")
        st.balloons()
    else:
        st.write(f"### Congratulations, {name}.\n The Product is un knownen")
        st.balloons()
