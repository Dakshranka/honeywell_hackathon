"""
Streamlit dashboard for interactive anomaly detection exploration.
Allows filtering by anomaly score, viewing top contributing features, and exporting filtered CSV.
"""


import streamlit as st
import pandas as pd
import numpy as np
import os

def main():
    st.title("Multivariate Time Series Anomaly Detection Dashboard")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "anomalies_output.csv")
    df = pd.read_csv(output_path, index_col=0, parse_dates=True)
    feature_cols = [col for col in df.columns if not col.startswith('abnormality_score') and not col.startswith('top_feature_')]
    score_cols = [col for col in df.columns if col.startswith('abnormality_score')]
    top_feature_cols = [col for col in df.columns if col.startswith('top_feature_')]

    st.sidebar.header("Filter Anomalies")
    score_col = st.sidebar.selectbox("Select abnormality score model", score_cols)
    threshold = st.sidebar.slider("Abnormality score threshold", 0, 100, 80)
    filtered_df = df[df[score_col] > threshold]

    st.write(f"Showing {len(filtered_df)} anomalies with {score_col} > {threshold}")
    st.dataframe(filtered_df[[score_col] + top_feature_cols + feature_cols])

    st.markdown("### Time Series Plot (select feature)")
    feature = st.selectbox("Select feature to plot", feature_cols)
    st.line_chart(df[feature])
    st.scatter_chart(filtered_df[feature])

    st.markdown("### Export Filtered Anomalies")
    if st.button("Export filtered anomalies to CSV"):
        export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "anomalies_filtered.csv")
        filtered_df.to_csv(export_path)
        st.success(f"Filtered anomalies exported to {export_path}")

    st.markdown("### Download Full Output CSV")
    st.download_button(
        label="Download full anomalies_output.csv",
        data=df.to_csv().encode('utf-8'),
        file_name="anomalies_output.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
