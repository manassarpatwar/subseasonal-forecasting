import pandas as pd
import config
import os

_, _, filenames = next(os.walk(config.RESULTS_DIR))

target_column = 'test_cosine_similarity'
groupby_column = 'architecture'
quantile_values = [0.25, 0.75]


for filename in filenames:
    result = pd.read_csv(os.path.join(config.RESULTS_DIR, filename))
    result = result[[groupby_column, target_column]]
    result = result.groupby(groupby_column)
    quantiles = result.quantile(quantile_values).reset_index()
    quantiles = quantiles.pivot(index=groupby_column, columns='level_1').droplevel('level_1', axis=1).round(4)
    quantiles.columns = [f"{v} quantile" for v in quantile_values]
    mean = result.mean().rename(columns={target_column: 'mean(std)'}).round(4)
    median = result.median().rename(columns={target_column: 'median'}).round(4)
    std = result.std().rename(columns={target_column: 'std'}).round(2)

    mean['mean(std)'] = mean['mean(std)'].astype(str)+'('+std['std'].astype(str)+')'

    table = mean.merge(median, on='architecture', how='inner')
    table = table.merge(quantiles, on='architecture', how='inner')
    table = table.reset_index()
    table = table.sort_values(['mean(std)', 'median'], ascending=False)

    latexname = os.path.splitext(filename)[0]+'.tex'
    table.to_latex(os.path.join(config.RESULTS_TABLE_DIR, latexname), index=False)
