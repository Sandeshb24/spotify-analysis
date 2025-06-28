import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

def run_analysis(uploaded_file):
    """
    Performs data cleaning and analysis on the Spotify dataset.
    Displays visualizations using Streamlit.
    """
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}. Please ensure it's a valid CSV.")
            return

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
    else:
        st.info("Please upload a CSV file to begin the analysis.")

# --- Streamlit App Layout ---
st.set_page_config(page_title="Spotify Song Virality Analyzer", layout="wide")

st.title("🎧 Spotify Song Virality Analyzer")
st.markdown("""
This application helps you analyze key features of Spotify songs to understand
what makes them viral. Upload your Spotify dataset (CSV format) to get started!
""")

uploaded_file = st.file_uploader("Upload your Spotify dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    st.markdown("---")
    run_analysis(uploaded_file)
else:
    st.info("Please upload a CSV file to proceed with the analysis. Make sure your CSV file contains columns like 'Popularity', 'Energy', 'Danceability', etc. as used in the original script.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Python.")
