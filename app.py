import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration and Model/Data Loading ---
output_dir = 'trained_components'

# Load original DataFrames
gpu_df_original = pd.read_csv('/content/Cleaned_GPU_Data .csv')
motherboard_df_original = pd.read_csv('/content/clean_motherbords_Data.csv')
ram_df_original = pd.read_csv('/content/clean_data_RAM.csv')
cpu_df_original = pd.read_csv('/content/cleaned_cpus.csv')

# Load latent feature DataFrames
gpu_latent_features = pd.read_csv(os.path.join(output_dir, 'gpu_latent_features.csv'))
motherboard_latent_features = pd.read_csv(os.path.join(output_dir, 'motherboard_latent_features.csv'))
ram_latent_features = pd.read_csv(os.path.join(output_dir, 'ram_latent_features.csv'))
cpu_latent_features = pd.read_csv(os.path.join(output_dir, 'cpu_latent_features.csv'))

# Load NCF compatibility model
ncf_model = load_model(os.path.join(output_dir, 'ncf_compatibility_model.keras'))

# Note: Autoencoder encoder models (gpu_encoder, etc.) are not directly used here
# to predict compatibility or recommendations, as we use the pre-computed latent features.
# If we needed to encode new parts on the fly, these would be necessary.

# Re-establish part_type_mapping and related global variables
compatibility_data = pd.read_csv('compatibility_dataset.csv')
all_part_types = pd.concat([compatibility_data['part1_type'], compatibility_data['part2_type']]).unique()
part_type_mapping = {type_name: i for i, type_name in enumerate(all_part_types)}
num_unique_part1_ids = compatibility_data['part1_id'].max() + 1
num_unique_part2_ids = compatibility_data['part2_id'].max() + 1
num_unique_part_types = len(all_part_types)

# Ensure RAM 'Type' column is clean (remove spaces if any) for compatibility inference
ram_df_original['Type'] = ram_df_original['Type'].str.strip()


# --- Backend Logic Functions (copied and adapted from notebook) ---

def predict_compatibility(part1_id, part1_type_str, part2_id, part2_type_str):
    """
    Predicts the compatibility between two parts using the NCF model.
    """
    part1_type_encoded = part_type_mapping.get(part1_type_str)
    part2_type_encoded = part_type_mapping.get(part2_type_str)

    if part1_type_encoded is None or part2_type_encoded is None:
        # st.warning(f"Unknown part type encountered: {part1_type_str} or {part2_type_str}")
        return 0.0 # Treat as incompatible if type is unknown

    model_input = {
        'part1_id_input': np.array([part1_id]),
        'part2_id_input': np.array([part2_id]),
        'part1_type_input': np.array([part1_type_encoded]),
        'part2_type_input': np.array([part2_type_encoded])
    }
    # Suppress verbose output from model.predict
    compatibility_probability = ncf_model.predict(model_input, verbose=0)[0][0]
    return compatibility_probability

def get_price_performance_recommendations(part_type_str, part_index, top_n=5, similarity_threshold=0.8):
    """
    Recommends parts similar to the given part_index from latent_df,
    filtering for those with a lower price from original_df.
    """
    latent_df = None
    original_df = None

    if part_type_str == 'GPU':
        latent_df = gpu_latent_features
        original_df = gpu_df_original
    elif part_type_str == 'Motherboard':
        latent_df = motherboard_latent_features
        original_df = motherboard_df_original
    elif part_type_str == 'RAM':
        latent_df = ram_latent_features
        original_df = ram_df_original
    elif part_type_str == 'CPU':
        latent_df = cpu_latent_features
        original_df = cpu_df_original
    else:
        return pd.DataFrame()

    # Check if part_index is valid. `iloc` is more robust here.
    if part_index >= len(original_df) or part_index < 0:
        return pd.DataFrame()

    input_part_name = original_df.iloc[part_index]['Name']
    input_part_price = original_df.iloc[part_index]['Price']
    input_vector = latent_df.iloc[part_index].values.reshape(1, -1)

    similarities = cosine_similarity(input_vector, latent_df).flatten()
    similarity_series = pd.Series(similarities, index=latent_df.index)

    # Exclude the part itself
    similarity_series = similarity_series.drop(part_index, errors='ignore')

    # Filter for similar parts where similarity is above the threshold
    similar_parts_indices = similarity_series[similarity_series > similarity_threshold].index

    if similar_parts_indices.empty:
        return pd.DataFrame()

    # Further filter for parts with a lower price than the input part
    cheaper_similar_parts_prices = original_df.loc[similar_parts_indices, 'Price']
    cheaper_similar_parts_indices = cheaper_similar_parts_prices[cheaper_similar_parts_prices < input_part_price].index

    if cheaper_similar_parts_indices.empty:
        return pd.DataFrame()

    # Get similarities for the cheaper, similar parts
    final_recommendations_similarity = similarity_series.loc[cheaper_similar_parts_indices]

    # Sort by similarity and return top_n recommendations
    final_recommendations_similarity = final_recommendations_similarity.sort_values(ascending=False)

    recommended_df = original_df.loc[final_recommendations_similarity.head(top_n).index].copy()
    recommended_df['Similarity'] = final_recommendations_similarity.head(top_n)

    return recommended_df[['Name', 'Price', 'Similarity']]

def get_compatible_parts(selected_part_type_str, selected_part_index, target_part_type_str, top_n=5, compatibility_threshold=0.7):
    """
    Recommends compatible parts of a target type for a selected part, using the NCF model.
    """
    target_original_df = None
    if target_part_type_str == 'Motherboard':
        target_original_df = motherboard_df_original
    elif target_part_type_str == 'RAM':
        target_original_df = ram_df_original
    elif target_part_type_str == 'CPU':
        target_original_df = cpu_df_original
    elif target_part_type_str == 'GPU': # Although not explicitly trained for in NCF, keep structure
        target_original_df = gpu_df_original
    else:
        return pd.DataFrame()

    compatible_parts_list = []

    # Iterate through each part in the target DataFrame
    for target_part_idx, target_part_row in target_original_df.iterrows():
        # Call predict_compatibility function
        compatibility_probability = predict_compatibility(
            selected_part_index, selected_part_type_str.lower(), # Convert to lower case for consistency with mapping
            target_part_idx, target_part_type_str.lower()
        )

        # If compatibility is above threshold, store the details
        if compatibility_probability >= compatibility_threshold:
            compatible_parts_list.append({
                'Part_Index': target_part_idx,
                'Name': target_part_row['Name'],
                'Compatibility_Score': compatibility_probability
            })

    # Convert list to DataFrame
    compatible_parts_df = pd.DataFrame(compatible_parts_list)

    if compatible_parts_df.empty:
        return pd.DataFrame()

    # Sort by compatibility score in descending order
    compatible_parts_df = compatible_parts_df.sort_values(by='Compatibility_Score', ascending=False)

    # Return top_n rows
    return compatible_parts_df.head(top_n)

def get_part_details(part_type_str, part_name):
    """
    Retrieves and formats all available details for a given part from its respective original dataframe.
    Returns a DataFrame with 'Feature' and 'Value' columns.
    """
    df = None
    if part_type_str == 'GPU':
        df = gpu_df_original
    elif part_type_str == 'Motherboard':
        df = motherboard_df_original
    elif part_type_str == 'RAM':
        df = ram_df_original
    elif part_type_str == 'CPU':
        df = cpu_df_original
    else:
        return pd.DataFrame(columns=["Feature", "Value"])

    # Find the part by name, allowing for partial matches and case-insensitivity
    part_row = df[df['Name'].str.contains(part_name, case=False, na=False)]

    if part_row.empty:
        return pd.DataFrame(columns=["Feature", "Value"])

    # Take the first match if multiple exist
    part_row_series = part_row.iloc[0].drop(['URL', 'Seller_Product_URL']) # Drop URLs as they might be too long/irrelevant for display

    details = []
    for feature, value in part_row_series.items():
        details.append({'Feature': feature, 'Value': value})

    return pd.DataFrame(details)

# Helper to get example names for placeholder
def get_example_name(part_type):
    if part_type == "GPU":
        return "GeForce RTX 4090" # Example, pick a common one
    elif part_type == "Motherboard":
        return "ASRock B550M/AC" # Example
    elif part_type == "RAM":
        return "Corsair Vengeance RGB 32GB" # Example
    elif part_type == "CPU":
        return "AMD Ryzen 5 7600X" # Example
    return "Part Name"


# --- Streamlit UI Layout ---
st.set_page_config(layout="wide")

st.title("PC Part Recommender System")
st.markdown("""
    Welcome to the PC Part Recommender!
    Select a part type and enter a part name to get cheaper alternatives,
    compatible components, and detailed information.
""")

# Input Section
st.header("1. Select a Part and Enter its Name")

col1, col2 = st.columns(2)

with col1:
    part_type_options = ["CPU", "Motherboard", "RAM", "GPU"]
    selected_part_type = st.selectbox(
        "Select Part Type",
        part_type_options,
        index=0,  # Default to CPU
        key="part_type_selector"
    )

with col2:
    part_name_input = st.text_input(
        f"Enter {selected_part_type} Name",
        placeholder=f"e.g., '{get_example_name(selected_part_type)}'",
        key="part_name_input"
    )

if st.button("Get Recommendations"):
    if not part_name_input:
        st.warning("Please enter a part name.")
    else:
        st.subheader(f"Recommendations for: {part_name_input} ({selected_part_type})")

        # Get the index of the selected part for backend functions
        selected_df_for_index = None
        if selected_part_type == 'GPU':
            selected_df_for_index = gpu_df_original
        elif selected_part_type == 'Motherboard':
            selected_df_for_index = motherboard_df_original
        elif selected_part_type == 'RAM':
            selected_df_for_index = ram_df_original
        elif selected_part_type == 'CPU':
            selected_df_for_index = cpu_df_original

        part_match_for_index = selected_df_for_index[selected_df_for_index['Name'].str.contains(part_name_input, case=False, na=False)]

        if part_match_for_index.empty:
            st.error(f"Part '{part_name_input}' not found in {selected_part_type} database. Please check the name and try again.")
        else:
            selected_part_index = part_match_for_index.index[0] # Take the first match

            # Results Section with Tabs
            tab1, tab2, tab3 = st.tabs(["Cheaper Alternatives", "Compatible Parts", "Part Details"])

            with tab1:
                st.header("Cheaper Alternatives (Similar Performance, Lower Price)")
                # Call the actual price/performance function
                cheaper_df = get_price_performance_recommendations(
                    part_type_str=selected_part_type,
                    part_index=selected_part_index,
                    top_n=5,
                    similarity_threshold=0.8
                )
                if not cheaper_df.empty:
                    st.dataframe(cheaper_df)
                else:
                    st.info(f"No cheaper alternatives found for '{part_name_input}' with similar performance.")

            with tab2:
                st.header("Compatible Parts")
                compatible_target_type = None
                if selected_part_type == "CPU":
                    compatible_target_type = "Motherboard"
                elif selected_part_type == "Motherboard":
                    compatible_target_type = "RAM"
                # GPU-CPU compatibility could be added if NCF was trained for it.

                if compatible_target_type:
                    st.markdown(f"Finding compatible **{compatible_target_type}s** for **{part_name_input}**...")
                    # Call the actual compatibility function
                    compatible_df = get_compatible_parts(
                        selected_part_type_str=selected_part_type,
                        selected_part_index=selected_part_index,
                        target_part_type_str=compatible_target_type,
                        top_n=10, # Display more compatible parts
                        compatibility_threshold=0.99 # High threshold for strong compatibility
                    )
                    if not compatible_df.empty:
                        st.dataframe(compatible_df)
                    else:
                        st.info(f"No highly compatible {compatible_target_type} found for '{part_name_input}'.")
                else:
                    st.info(f"Compatibility recommendations for **{selected_part_type}** are currently limited to CPU-Motherboard and Motherboard-RAM pairs.")


            with tab3:
                st.header("Detailed Information")
                # Call the actual part details function
                details_df = get_part_details(selected_part_type, part_name_input)
                if not details_df.empty:
                    st.dataframe(details_df.set_index('Feature')) # Set Feature as index for better display
                else:
                    st.info(f"Details for '{part_name_input}' not found or an error occurred.")

st.markdown("---")
st.markdown("Developed with Streamlit for PC Part Recommendation System")
