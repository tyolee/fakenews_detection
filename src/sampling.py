

# Data Sampling Function
def stratified_sample_df(df, sample_ratio, label_col='2_way_label'):
    """Performs stratified sampling on a DataFrame."""
    if len(df) == 0:
        return df
    
    sample_size = int(len(df) * sample_ratio)
    
    # Handle the edge case where the sample size is smaller than the number of groups
    if sample_size == 0 and len(df) > 0:
        return df.sample(min(1, len(df)), random_state=42)
        
    df_sampled = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(n=int(sample_size * len(x) / len(df)), random_state=42)
    ).reset_index(drop=True)
    return df_sampled