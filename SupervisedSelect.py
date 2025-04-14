import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('../elife/test_simscore_predictions.csv')
    df['original_order'] = df.groupby('id').cumcount()
    top_50_per_id = df.groupby('id').head(50).sort_values(['id', 'original_order'])
    selected_df = top_50_per_id.groupby('id')['paragraph'].apply(list).reset_index()
    selected_df.rename(columns={'paragraph': 'selected_sentences'}, inplace=True)
    selected_df.to_csv('../elife/test_supervised_top50.csv')



