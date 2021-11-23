from scipy.stats import normaltest

def normal_distribution_check(df, drop_columns=True, normal = True):
    df_transformed = df.copy()

    columns       = 0
    normal_columns = 0
    columns_to_drop = []
    for column in df:
        if df[column].dtype == object:
            # optional
            columns_to_drop.append(column)
            continue
        
        columns += 1
        if normaltest(df[column])[1] < 0.05 and normal:
            normal_columns += 1
            if drop_columns:
                columns_to_drop.append(column)
        elif normaltest(df[column])[1] > 0.05 and not normal:
            normal_columns -= 1
            if drop_columns:
                columns_to_drop.append(column)
        print(column, normaltest(df[column]))

    if drop_columns:
        df_transformed.drop(columns=columns_to_drop, inplace=True)
    
    return df_transformed, columns, normal_columns