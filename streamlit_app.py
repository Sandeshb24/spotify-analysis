import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os # Import os module to handle file paths

def run_analysis(df):
    """
    Performs data cleaning and analysis on the Spotify dataset.
    Displays visualizations using Streamlit.
    """
    st.subheader("1. Data Cleaning Summary")

    # Drop duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    st.write(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")

    # Drop rows with missing values in important columns
    important_cols = ['Popularity', 'Energy', 'Danceability', 'Positiveness', 'Speechiness',
                      'Liveness', 'Acousticness', 'Instrumentalness', 'Tempo', 'Loudness (db)']
    initial_rows_after_duplicates = df.shape[0]
    df.dropna(subset=important_cols, inplace=True)
    st.write(f"Dropped {initial_rows_after_duplicates - df.shape[0]} rows with missing values in important columns.")

    # Handle 'Explicit' column (convert Yes/No or True/False to binary)
    if 'Explicit' in df.columns and df['Explicit'].dtype == object:
        df['Explicit'] = df['Explicit'].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(0)
        st.write("Converted 'Explicit' column to binary (1 for explicit, 0 for non-explicit).")
    elif 'Explicit' in df.columns:
        st.write("'Explicit' column is already numeric or not present.")
    else:
        st.write("'Explicit' column not found.")


    # Create 'Viral' label
    # A song is considered 'viral' if its popularity is 80 or above.
    df['Viral'] = df['Popularity'].apply(lambda x: 1 if x >= 80 else 0)
    st.write("Created 'Viral' label based on Popularity (>= 80).")

    st.success("Data Cleaning Complete!")
    st.write("Cleaned Data Head:")
    st.dataframe(df.head())

    st.subheader("2. Exploratory Data Analysis")
    features = ['Energy', 'Danceability', 'Positiveness', 'Speechiness', 'Liveness',
                'Acousticness', 'Instrumentalness', 'Tempo', 'Loudness (db)']

    # Compare feature distributions
    st.markdown("### Feature Distributions in Viral vs Non-Viral Songs")
    for col in features:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(df[df['Viral'] == 1][col], label='Viral', fill=True, color='green', ax=ax)
        sns.kdeplot(df[df['Viral'] == 0][col], label='Non-Viral', fill=True, color='red', ax=ax)
        ax.set_title(f"{col} Distribution in Viral vs Non-Viral Songs")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent display issues

    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[features + ['Viral']].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    # Genre Analysis (Top genres among viral songs)
    if 'Genre' in df.columns:
        st.markdown("### Top Genres in Viral Songs")
        top_genres = df[df['Viral'] == 1]['Genre'].value_counts().head(10)
        if not top_genres.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_genres.plot(kind='bar', color='purple', title='Top Genres in Viral Songs', ax=ax)
            ax.set_xlabel("Genre")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No viral songs found to analyze genres or 'Genre' column has no data for viral songs.")
    else:
        st.warning("'Genre' column not found in the dataset. Skipping genre analysis.")

    # Save cleaned data for download
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="cleaned_spotify_data.csv",
        mime="text/csv",
    )

    st.success("Analysis complete. You can download the cleaned data above.")


# --- Streamlit App Layout ---
st.set_page_config(page_title="Spotify Song Virality Analyzer", layout="wide")

st.title("🎧 Spotify Song Virality Analyzer")
st.markdown("""
This application helps you analyze key features of Spotify songs to understand
what makes them viral.
""")

# Define the path to your dataset
# IMPORTANT: Adjust this path if your spotify_dataset.csv is in a different location relative to your app.py
DATA_PATH = "spotify_dataset.csv"

# Load the dataset directly
try:
    df_loaded = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "devdope/900k-spotify",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
    st.success(f"Successfully loaded '{DATA_PATH}' for analysis.")
    st.markdown("---")
    run_analysis(df_loaded)
except FileNotFoundError:
    st.error(f"Error: The file '{DATA_PATH}' was not found. Please ensure the CSV file is in the same directory as this Streamlit app or provide the correct absolute/relative path.")
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Python.")
